import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer

class OldCartPoleEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(OldCartPoleEnv, self).__init__()
        
        # Load legacy XML. Adjust path if necessary
        self.model = mujoco.MjModel.from_xml_path("cartpole_mujoco/urdf/cartpole_native.xml")
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        
        # Action space for DC Motor (Force in Newtons)
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
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[0] = action[0]
        
        # Frame skipping for 100Hz control loop
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        self.current_step += 1
        
        cart_x = obs[0]
        # Match XML limit: range="-0.47 0.47" to prevent boundary glitch
        terminated = bool(cart_x < -0.47 or cart_x > 0.47)
        truncated = bool(self.current_step >= self.max_steps)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, 0.0, terminated, truncated, {}

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