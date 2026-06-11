from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import os
import time
from train import CartPoleSwingUpEnv

# mode selection flag
USE_SPECIFIC_SEED = True #False for random seed 
SEED = 20 

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    # dynamic path resolution
    if USE_SPECIFIC_SEED:
        stats_path = f"vec/vec_normalize_seed_{SEED}.pkl"
        model_path = f"models/ppo_force_real_seed_{SEED}.zip"
    else:
        stats_path = "vec/vec_normalize.pkl"
        model_path = "models/ppo_force_real.zip"

    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False 
    vec_env.norm_reward = False
    vec_env = VecFrameStack(vec_env, n_stack=8)

    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()

    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, info = vec_env.step(action)
            
            # lock physics fps
            time.sleep(0.01) 
            
            if dones[0]:
                is_truncated = info[0].get("TimeLimit.truncated", False)
                print("Truncated" if is_truncated else "Terminated")
                time.sleep(1) 
                
    except KeyboardInterrupt:
        pass
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()