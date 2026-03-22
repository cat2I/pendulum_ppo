import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

class CartPoleSwingUpEnv(gym.Env):
    def __init__(self, render_mode="none"):
        super(CartPoleSwingUpEnv, self).__init__()
        self.model = mujoco.MjModel.from_xml_path("cartpole_mujoco/urdf/cartpole_native.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.current_step = 0 
        self.prev_action = 0.0 # Init tracking for action smoothing
        # Action space: [-30N, 30N]
        self.action_space = spaces.Box(low=-30.0, high=30.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            high=np.array([5.0, 50.0, 1.0, 1.0, 50.0], dtype=np.float32), 
            dtype=np.float32
        )
        self.max_steps = 1000 
        self.current_step = 0 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = 0.0  
        self.data.qpos[1] = self.np_random.uniform(-0.1, 0.1) 
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.current_step = 0
        if self.render_mode == "human":
            self._render_frame()
            self.current_step = 0
            self.prev_action = 0.0 # Clear history on new episode
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[0] = action[0]
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        cart_x, cart_vel, cos_th, sin_th, pole_vel = obs
        self.current_step += 1
        # Rewards and penalties
        reward_theta = 1.0-cos_th 
        # Penalties: softer scaling, prevent gradient spike
        penalty_x = 1.0 * (cart_x**2)
        penalty_action = 0.001 * (action[0]**2)
        # Action rate penalty: limit high-frequency changes (chattering)
        action_diff = action[0] - self.prev_action
        penalty_action_rate = 0.05 * (action_diff**2) 
        self.prev_action = action[0] # Update for next step
        # Velocity penalty: prevent reward hacking (swing-up by dropping)
        penalty_vx = 0.05 * (cart_vel**2)
        penalty_vth = 0.005 * (pole_vel**2)
        penalty_boundary = 0.0
        # Soft boundary penalty (> 0.3m)
        if abs(cart_x) > 0.3:
            penalty_boundary = 10.0 * (abs(cart_x) - 0.3) #soft penalty
        # Total reward calculation
        reward = float(reward_theta - penalty_x - penalty_action - penalty_action_rate - penalty_boundary - penalty_vx - penalty_vth)
        # Hard termination (0.45m limit)
        terminated = bool(abs(cart_x) > 0.45) 
        if terminated:
            reward -= 5.0 # Hard penalty: reduced from 100 to avoid reward cliff
        truncated = bool(self.current_step >= self.max_steps)
        if self.render_mode == "human":
            self._render_frame()
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        x = self.data.qpos[0]
        theta = self.data.qpos[1]
        v_x = self.data.qvel[0]
        v_th = self.data.qvel[1]
        return np.array([x, v_x, np.cos(theta), np.sin(theta), v_th], dtype=np.float32)

    def _render_frame(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

from typing import Callable

# Linear schedule: progress from 1.0 to 0.0 (LR schedule)
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func
if __name__ == "__main__":
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    env = CartPoleSwingUpEnv(render_mode="none")
    env = Monitor(env)
    # 2. Bọc môi trường để tương thích với SB3 vectorization
    vec_env = DummyVecEnv([lambda: env])
    # 3. Kích hoạt Frame Stacking (Lưu lịch sử 8 frames)
    # Lưu ý: Lúc này Observation Space của bạn sẽ tự động tăng lên gấp 8 lần (5 x 8 = 40 chiều)
    vec_env = VecFrameStack(vec_env, n_stack=8)
    custom_arch = dict(activation_fn=nn.Tanh, net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", vec_env, verbose=1,
                learning_rate=linear_schedule(3e-4), # LR decay
                target_kl=0.015, # KL constraint: early stop policy update (target_kl)
                policy_kwargs=custom_arch, 
                tensorboard_log="./tensorboard/tensorboard_logs/",
                device="cpu")
    print("Training begins...")
    model.learn(total_timesteps=500000, tb_log_name="ppo_vel_control")
    model.save("models/ppo_vel_control")
    print("Done!")