from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import os
import time
from train_force import CartPoleSwingUpEnv

# mode selection flag
USE_SPECIFIC_SEED = False #False for random seed 
SEED = 40 

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    vec_env = DummyVecEnv([lambda: env])

    # dynamic path resolution
    if USE_SPECIFIC_SEED:
        stats_path = f"vec/stack 1/vec_normalize_stack1_seed_{SEED}.pkl"
        model_path = f"models/stack 1/ppo_force_real_stack1_seed_{SEED}.zip"
    else:
        stats_path = "vec/env/vec_normalize_tune_051107_seed42.pkl"
        model_path = "models/env/tune_051107_seed42.zip"
        #stats_path = "z.vel/vec_normalize_vel_42.pkl"
        #model_path = "z.vel/ppo_vel_42.zip"

    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False 
    vec_env.norm_reward = False
    vec_env = VecFrameStack(vec_env, n_stack=8) #change framestack

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