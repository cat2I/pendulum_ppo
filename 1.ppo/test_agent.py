from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import os
import time
from train_agent import CartPoleSwingUpEnv

def main():
    env = CartPoleSwingUpEnv(render_mode="human")
    # Env Wrapping
    vec_env = DummyVecEnv([lambda: env])

    print("Loading VecNormalize stats...")
    vec_env = VecNormalize.load("vec/vec_normalize.pkl", vec_env)
    
    vec_env.training = False 
    vec_env.norm_reward = False
    # Frame Stacking (8 frames)
    vec_env = VecFrameStack(vec_env, n_stack=8)

    print("Loading pre-trained model...")
    model = PPO.load("models/ppo_force_real.zip", env=vec_env)

    obs = vec_env.reset()
    print("Start testing...")

    try:
        # Infinite loop: Runs until user presses Ctrl+C
        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, dones, info = vec_env.step(action)
            
            time.sleep(0.01) 

            # Auto-reset on crash or timeout, but keeps window open
            if dones[0]:
                print("Episode finished. Restarting...")
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        vec_env.close()

if __name__ == "__main__":
    main()