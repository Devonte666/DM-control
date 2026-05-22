import argparse
import json
import os
import random
import sys
import traceback
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# we recommend students to use conda, like "conda activate dm_control" to activate the environment. 

# =========================
# Utils
# =========================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def flatten_obs(obs_dict):
    return np.concatenate([v.ravel() for v in obs_dict.values()]).astype(np.float32)


def scale_action(action_normalized, action_spec):
    """
    TD3 policy outputs action in [-1, 1].
    dm_control action space may have different bounds.
    """
    minimum = action_spec.minimum
    maximum = action_spec.maximum
    return minimum + (action_normalized + 1.0) * 0.5 * (maximum - minimum)


def random_action_normalized(action_dim):
    return np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)


# =========================
# Reward Functions
# =========================
#
# dm_control has its task rewards inside the installed package. For teaching,
# we keep the reward code here so students can read and modify it directly.
#
# This block reproduces the walker stand / walk / run reward from:
# dm_control.suite.walker.PlanarWalker.get_reward

WALKER_STAND_HEIGHT = 1.3
WALKER_STAND_MARGIN = 0.35
WALKER_WALK_SPEED = 2.0
WALKER_RUN_SPEED = 8.0
WALKER_MAX_KNEE_BEND = 2.0
WALKER_KNEE_MARGIN = 0.6
WALKER_MAX_VERTICAL_SPEED = 1.0
WALKER_VERTICAL_SPEED_MARGIN = 2.0
WALKER_MIN_LEG_ACTIVITY_RATIO = 0.25
WALKER_LEG_ACTIVITY_RATIO_MARGIN = 0.25
WALKER_MIN_FOOT_SEPARATION = 0.15
WALKER_MAX_FOOT_SEPARATION = 0.9
WALKER_FOOT_SEPARATION_MARGIN = 0.4
WALKER_MIN_ALTERNATION_PROGRESS = 0.01
WALKER_ALTERNATION_PROGRESS_MARGIN = 0.08
WALKER_ALTERNATION_WEIGHT = 0.8
WALKER_FRONT_FOOT_THRESHOLD = 0.12
WALKER_CATCHUP_REWARD_WEIGHT = 0.2
WALKER_SWITCH_REWARD_WEIGHT = 0.5
WALKER_FRONT_BALANCE_REWARD_WEIGHT = 0.3
WALKER_UPRIGHT_QPOS = {
    "rootz": 0.0,
    "rootx": 0.0,
    "rooty": 0.0,
    "right_hip": 0.0,
    "right_knee": 0.0,
    "right_ankle": 0.0,
    "left_hip": 0.0,
    "left_knee": 0.0,
    "left_ankle": 0.0,
}


def tolerance(
    x,
    bounds=(0.0, 0.0),
    margin=0.0,
    value_at_margin=0.1,
    sigmoid="gaussian",
):
    """
    Soft version of "is x inside bounds?".

    Returns:
        1.0 when x is inside bounds.
        A value in [0, 1] when x is outside bounds, decreasing with distance.

    For the walker reward below, we only need:
        sigmoid="gaussian" for standing height.
        sigmoid="linear" for walking/running speed.
    """
    lower, upper = bounds

    if lower > upper:
        raise ValueError("Lower bound must be <= upper bound.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")

    in_bounds = lower <= x <= upper
    if in_bounds:
        return 1.0
    if margin == 0:
        return 0.0

    distance = lower - x if x < lower else x - upper
    scaled_distance = distance / margin

    if sigmoid == "gaussian":
        scale = np.sqrt(-2.0 * np.log(value_at_margin))
        return float(np.exp(-0.5 * (scaled_distance * scale) ** 2))

    if sigmoid == "linear":
        value = 1.0 - scaled_distance * (1.0 - value_at_margin)
        return float(np.clip(value, 0.0, 1.0))

    raise ValueError(f"Unsupported sigmoid type: {sigmoid!r}")


def walker_move_speed(task):
    if task == "stand":
        return 0.0
    if task == "walk":
        return WALKER_WALK_SPEED
    if task == "run":
        return WALKER_RUN_SPEED
    raise ValueError(
        f"Local walker reward only supports task='stand', 'walk', or 'run', got {task!r}."
    )


def physics_value(value):
    """Converts a scalar dm_control named array value to a plain Python float."""
    return float(np.asarray(value).reshape(-1)[0])


def leg_activity(physics, side):
    """Returns summed absolute joint speed for one leg."""
    return sum(
        abs(physics_value(physics.named.data.qvel[f"{side}_{joint}"]))
        for joint in ("hip", "knee", "ankle")
    )


def foot_x(physics, side):
    """Returns the horizontal position of one foot."""
    return physics_value(physics.named.data.xpos[f"{side}_foot", "x"])


def foot_diff(physics):
    """Positive means left foot is in front; negative means right foot is in front."""
    return foot_x(physics, "left") - foot_x(physics, "right")


def capture_reward_state(env, domain):
    if domain != "walker":
        return None
    return {"foot_diff": foot_diff(env.physics)}


def new_gait_state():
    return {
        "left_front_steps": 0,
        "right_front_steps": 0,
        "last_front_foot": None,
        "switches": 0,
    }


def front_foot(current_foot_diff):
    if current_foot_diff > WALKER_FRONT_FOOT_THRESHOLD:
        return "left"
    if current_foot_diff < -WALKER_FRONT_FOOT_THRESHOLD:
        return "right"
    return None


def update_gait_state(gait_state, current_foot_diff):
    if gait_state is None:
        return 0.0, 1.0

    current_front_foot = front_foot(current_foot_diff)
    switch_reward = 0.0

    if current_front_foot is not None:
        last_front_foot = gait_state["last_front_foot"]
        if last_front_foot is not None and current_front_foot != last_front_foot:
            switch_reward = 1.0
            gait_state["switches"] += 1

        gait_state["last_front_foot"] = current_front_foot
        if current_front_foot == "left":
            gait_state["left_front_steps"] += 1
        else:
            gait_state["right_front_steps"] += 1

    left_steps = gait_state["left_front_steps"]
    right_steps = gait_state["right_front_steps"]
    max_steps = max(left_steps, right_steps)
    if max_steps == 0:
        front_balance_reward = 1.0
    else:
        front_balance_reward = min(left_steps, right_steps) / max_steps

    return switch_reward, front_balance_reward


def alternating_gait_reward(current_foot_diff, previous_foot_diff, gait_state=None):
    """
    Rewards actual left/right foot alternation.

    The catchup term is only shaping. The main terms reward front-foot switches
    and a balanced amount of time with each foot in front.
    """
    if previous_foot_diff is None or abs(previous_foot_diff) < 1e-6:
        catchup_reward = 1.0
    else:
        foot_diff_delta = current_foot_diff - previous_foot_diff
        rear_catchup_progress = -np.sign(previous_foot_diff) * foot_diff_delta
        catchup_reward = tolerance(
            rear_catchup_progress,
            bounds=(WALKER_MIN_ALTERNATION_PROGRESS, float("inf")),
            margin=WALKER_ALTERNATION_PROGRESS_MARGIN,
            value_at_margin=0.2,
            sigmoid="linear",
        )

    switch_reward, front_balance_reward = update_gait_state(
        gait_state,
        current_foot_diff,
    )

    return (
        WALKER_CATCHUP_REWARD_WEIGHT * catchup_reward
        + WALKER_SWITCH_REWARD_WEIGHT * switch_reward
        + WALKER_FRONT_BALANCE_REWARD_WEIGHT * front_balance_reward
    )


def walker_reward(physics, task, previous_state=None, gait_state=None):
    """
    Reward for dm_control walker tasks, written directly in this file.

    Key physical quantities:
        torso_height: z height of the torso.
        torso_upright: how aligned the torso is with the world z axis.
        horizontal_velocity: forward center-of-mass velocity.
        knee_bend: how deeply the knees are bent.
        vertical_speed: up/down torso speed, used to discourage bouncing.
        leg_activity: joint speed for each leg, used to discourage one-leg gait.
        foot_separation: horizontal distance between feet.
        foot_diff: signed horizontal foot distance, used for left/right alternation.

    Reward structure:
        stand_reward encourages the robot to stay high and upright.
        knee_reward discourages the robot from moving on deeply bent knees.
        smooth_reward discourages repeated jump-and-drop motion.
        gait_reward discourages dragging one inactive leg.
        alternation_reward rewards front-foot switches and balanced front time.
        move_reward encourages forward speed for walk/run tasks.
        final reward is posture_reward for standing, or posture_reward * speed bonus.
    """
    move_speed = walker_move_speed(task)

    torso_height = physics_value(physics.named.data.xpos["torso", "z"])
    torso_upright = physics_value(physics.named.data.xmat["torso", "zz"])
    horizontal_velocity = physics_value(
        physics.named.data.sensordata["torso_subtreelinvel"][0]
    )
    right_knee = abs(physics_value(physics.named.data.qpos["right_knee"]))
    left_knee = abs(physics_value(physics.named.data.qpos["left_knee"]))
    max_knee_bend = max(right_knee, left_knee)
    vertical_speed = abs(physics_value(physics.named.data.qvel["rootz"]))
    right_activity = leg_activity(physics, "right")
    left_activity = leg_activity(physics, "left")
    min_leg_activity = min(right_activity, left_activity)
    max_leg_activity = max(right_activity, left_activity)
    leg_activity_ratio = min_leg_activity / (max_leg_activity + 1e-6)
    current_foot_diff = foot_diff(physics)
    foot_separation = abs(current_foot_diff)

    standing = tolerance(
        torso_height,
        bounds=(WALKER_STAND_HEIGHT, float("inf")),
        margin=WALKER_STAND_MARGIN,
    )
    upright = (1.0 + torso_upright) / 2.0
    stand_reward = standing * upright

    knee_reward = tolerance(
        max_knee_bend,
        bounds=(0.0, WALKER_MAX_KNEE_BEND),
        margin=WALKER_KNEE_MARGIN,
        value_at_margin=0.2,
        sigmoid="linear",
    )

    smooth_reward = tolerance(
        vertical_speed,
        bounds=(0.0, WALKER_MAX_VERTICAL_SPEED),
        margin=WALKER_VERTICAL_SPEED_MARGIN,
        value_at_margin=0.2,
        sigmoid="linear",
    )

    posture_reward = stand_reward * knee_reward * smooth_reward

    if move_speed == 0.0:
        return float(posture_reward)

    move_reward = tolerance(
        horizontal_velocity,
        bounds=(move_speed, float("inf")),
        margin=move_speed / 2.0,
        value_at_margin=0.5,
        sigmoid="linear",
    )

    leg_balance_reward = tolerance(
        leg_activity_ratio,
        bounds=(WALKER_MIN_LEG_ACTIVITY_RATIO, float("inf")),
        margin=WALKER_LEG_ACTIVITY_RATIO_MARGIN,
        value_at_margin=0.3,
        sigmoid="linear",
    )
    foot_separation_reward = tolerance(
        foot_separation,
        bounds=(WALKER_MIN_FOOT_SEPARATION, WALKER_MAX_FOOT_SEPARATION),
        margin=WALKER_FOOT_SEPARATION_MARGIN,
        value_at_margin=0.3,
        sigmoid="linear",
    )
    previous_foot_diff = None
    if previous_state is not None:
        previous_foot_diff = previous_state.get("foot_diff")
    alternation_reward = alternating_gait_reward(
        current_foot_diff,
        previous_foot_diff,
        gait_state=gait_state,
    )
    alternation_factor = (
        1.0
        - WALKER_ALTERNATION_WEIGHT
        + WALKER_ALTERNATION_WEIGHT * alternation_reward
    )
    gait_reward = leg_balance_reward * foot_separation_reward * alternation_factor

    return float(posture_reward * gait_reward * (0.2 + 0.8 * move_reward))


def compute_reward(env, domain, task, previous_state=None, gait_state=None):
    if domain != "walker":
        raise ValueError(
            "This teaching script now computes rewards in run.py, "
            f"but only walker rewards are implemented. Got domain={domain!r}."
        )
    return walker_reward(
        env.physics,
        task,
        previous_state=previous_state,
        gait_state=gait_state,
    )


def set_walker_upright_pose(physics):
    """Overrides dm_control's randomized walker reset with a clear upright pose."""
    for joint_name, value in WALKER_UPRIGHT_QPOS.items():
        physics.named.data.qpos[joint_name] = value

    for joint_name in physics.named.data.qvel.axes.row.names:
        physics.named.data.qvel[joint_name] = 0.0

    physics.forward()


def walker_observation(physics):
    """Rebuilds walker observation after manually changing qpos/qvel."""
    return {
        "orientations": physics.named.data.xmat[1:, ["xx", "xz"]].ravel(),
        "height": physics.named.data.xpos["torso", "z"],
        "velocity": physics.velocity(),
    }


def reset_env(env, domain):
    """
    Resets an environment and returns a flattened observation.

    dm_control's walker task randomizes joint angles on reset. For teaching,
    starting from an upright pose is easier to understand and debug, so we
    overwrite the randomized pose immediately after reset.
    """
    ts = env.reset()
    if domain == "walker":
        set_walker_upright_pose(env.physics)
        return ts, flatten_obs(walker_observation(env.physics))
    return ts, flatten_obs(ts.observation)


def make_output_dir(args):
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        return args.save_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(args.log_root, timestamp)
    output_dir = base_dir
    suffix = 1

    while os.path.exists(output_dir):
        output_dir = f"{base_dir}_{suffix:02d}"
        suffix += 1

    os.makedirs(output_dir)
    return output_dir


def configure_rendering(args):
    if args.save_video:
        os.environ.setdefault("MUJOCO_GL", args.render_backend)


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for file in self.files:
            file.write(data)
            file.flush()

    def flush(self):
        for file in self.files:
            file.flush()

    def close(self):
        for file in self.files:
            try:
                file.flush()
            except ValueError:
                pass


# =========================
# Replay Buffer
# =========================

class ReplayBuffer:
    def __init__(self, obs_dim, action_dim, capacity):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)

        obs = torch.as_tensor(self.obs[idx], device=device)
        actions = torch.as_tensor(self.actions[idx], device=device)
        rewards = torch.as_tensor(self.rewards[idx], device=device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=device)
        dones = torch.as_tensor(self.dones[idx], device=device)

        return obs, actions, rewards, next_obs, dones


# =========================
# Networks
# =========================

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


# =========================
# TD3 Agent
# =========================

class TD3Agent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        self.actor = Actor(obs_dim, action_dim).to(device)
        self.actor_target = Actor(obs_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_updates = 0

    @torch.no_grad()
    def select_action(self, obs, noise_std=0.0):
        obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        action = self.actor(obs_t).cpu().numpy()[0]

        if noise_std > 0.0:
            action += np.random.normal(0.0, noise_std, size=action.shape)

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def update(self, replay_buffer, batch_size):
        self.total_updates += 1

        obs, actions, rewards, next_obs, dones = replay_buffer.sample(
            batch_size, self.device
        )

        with torch.no_grad():
            noise = (
                torch.randn_like(actions) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_actions = (
                self.actor_target(next_obs) + noise
            ).clamp(-1.0, 1.0)

            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + self.gamma * (1.0 - dones) * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None

        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, actor_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)

            actor_loss_value = actor_loss.item()

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
        }

    def soft_update(self, source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(
                self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data
            )

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])


# =========================
# Evaluation
# =========================

def evaluate_policy(agent, domain, task, seed, episodes=5):
    from dm_control import suite

    env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
    action_spec = env.action_spec()

    returns = []

    for _ in range(episodes):
        ts, obs = reset_env(env, domain)
        gait_state = new_gait_state()

        episode_return = 0.0

        while not ts.last():
            action_norm = agent.select_action(obs, noise_std=0.0)
            action_env = scale_action(action_norm, action_spec)

            previous_reward_state = capture_reward_state(env, domain)
            ts = env.step(action_env)
            obs = flatten_obs(ts.observation)

            reward = compute_reward(
                env,
                domain,
                task,
                previous_state=previous_reward_state,
                gait_state=gait_state,
            )
            episode_return += reward

        returns.append(episode_return)

    return float(np.mean(returns))


def record_video(agent, domain, task, seed, output_path, height=480, width=640, fps=30):
    import imageio.v2 as imageio
    from dm_control import suite

    env = suite.load(domain_name=domain, task_name=task, task_kwargs={"random": seed})
    action_spec = env.action_spec()

    frames = []

    ts, obs = reset_env(env, domain)

    while not ts.last():
        frame = env.physics.render(height=height, width=width, camera_id=0)
        frames.append(frame)

        action_norm = agent.select_action(obs, noise_std=0.0)
        action_env = scale_action(action_norm, action_spec)

        ts = env.step(action_env)
        obs = flatten_obs(ts.observation)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


# =========================
# Training
# =========================

def train(args):
    from dm_control import suite

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    print(f"Using device: {device}")
    print(f"Environment: {args.domain} / {args.task}")
    print("Reward source: local functions in run.py")
    print("Initial pose: upright reset in run.py")
    print(f"Output dir: {args.save_dir}")
    if args.save_video:
        print(f"Render backend: {os.environ.get('MUJOCO_GL')}")

    env = suite.load(
        domain_name=args.domain,
        task_name=args.task,
        task_kwargs={"random": args.seed},
    )

    action_spec = env.action_spec()

    ts, obs = reset_env(env, args.domain)

    obs_dim = obs.shape[0]
    action_dim = int(np.prod(action_spec.shape))

    print(f"obs_dim: {obs_dim}")
    print(f"action_dim: {action_dim}")
    print(f"action min: {action_spec.minimum}")
    print(f"action max: {action_spec.maximum}")

    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=args.replay_size,
    )

    agent = TD3Agent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
    )

    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    recent_returns = deque(maxlen=10)

    ts, obs = reset_env(env, args.domain)
    gait_state = new_gait_state()

    for step in range(1, args.train_steps + 1):
        if step < args.start_steps:
            action_norm = random_action_normalized(action_dim)
        else:
            action_norm = agent.select_action(obs, noise_std=args.exploration_noise)

        action_env = scale_action(action_norm, action_spec)

        previous_reward_state = capture_reward_state(env, args.domain)
        next_ts = env.step(action_env)
        next_obs = flatten_obs(next_ts.observation)

        reward = compute_reward(
            env,
            args.domain,
            args.task,
            previous_state=previous_reward_state,
            gait_state=gait_state,
        )
        done = float(next_ts.last())

        replay_buffer.add(obs, action_norm, reward, next_obs, done)

        obs = next_obs
        ts = next_ts

        episode_return += reward
        episode_length += 1

        if replay_buffer.size >= args.batch_size and step >= args.start_steps:
            losses = agent.update(replay_buffer, args.batch_size)
        else:
            losses = None

        if ts.last():
            episode_count += 1
            recent_returns.append(episode_return)

            print(
                f"step={step} "
                f"episode={episode_count} "
                f"return={episode_return:.2f} "
                f"length={episode_length} "
                f"recent_avg={np.mean(recent_returns):.2f}"
            )

            ts, obs = reset_env(env, args.domain)
            gait_state = new_gait_state()

            episode_return = 0.0
            episode_length = 0

        if step % args.eval_interval == 0:
            eval_return = evaluate_policy(
                agent,
                domain=args.domain,
                task=args.task,
                seed=args.seed + 100,
                episodes=args.eval_episodes,
            )

            print("=" * 60)
            print(f"Evaluation at step {step}: avg_return={eval_return:.2f}")
            print("=" * 60)

            save_path = os.path.join(args.save_dir, "td3_walker.pt")
            agent.save(save_path)
            print(f"Saved model to {save_path}")

            if args.save_video:
                video_path = os.path.join(args.save_dir, f"walker_step_{step}.mp4")
                record_video(
                    agent,
                    domain=args.domain,
                    task=args.task,
                    seed=args.seed + 200,
                    output_path=video_path,
                )
                print(f"Saved video to {video_path}")

    final_path = os.path.join(args.save_dir, "td3_walker_final.pt")
    agent.save(final_path)
    print(f"Training finished. Final model saved to {final_path}")

    if args.save_video:
        final_video_path = os.path.join(args.save_dir, "walker_final.mp4")
        record_video(
            agent,
            domain=args.domain,
            task=args.task,
            seed=args.seed + 300,
            output_path=final_video_path,
        )
        print(f"Saved final video to {final_video_path}")


# =========================
# Main
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    # dm_control task
    parser.add_argument("--domain", type=str, default="walker")
    parser.add_argument("--task", type=str, default="walk")

    # training
    parser.add_argument("--train_steps", type=int, default=300_000)
    parser.add_argument("--start_steps", type=int, default=10_000)
    parser.add_argument("--replay_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)

    # TD3 hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_delay", type=int, default=2)

    # eval / save
    parser.add_argument("--eval_interval", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--log_root", type=str, default="./log")
    parser.add_argument("--save_dir", type=str, default=None)
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument("--save_video", dest="save_video", action="store_true")
    video_group.add_argument("--no_save_video", dest="save_video", action="store_false")
    parser.set_defaults(save_video=True)
    parser.add_argument(
        "--render_backend",
        type=str,
        default="egl",
        choices=["egl", "glfw", "osmesa"],
    )

    # misc
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.save_dir = make_output_dir(args)
    configure_rendering(args)

    log_path = os.path.join(args.save_dir, "train.log")
    config_path = os.path.join(args.save_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as config_file:
        json.dump(vars(args), config_file, indent=2, sort_keys=True)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    with open(log_path, "a", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        sys.stderr = Tee(original_stderr, log_file)

        try:
            print(f"Logging to: {log_path}")
            print(f"Config saved to: {config_path}")
            train(args)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
