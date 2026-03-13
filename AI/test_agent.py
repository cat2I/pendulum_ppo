from stable_baselines3 import PPO
import time
from train_agent import CartPoleSwingUpEnv

def main():
    env = CartPoleSwingUpEnv(render_mode="human")

    print("Loading pre-trained model...")
    model = PPO.load("models/ppo_cartpole_swingup.zip", env=env)

    obs, info = env.reset()
    print("Start testing... Press Ctrl+C in terminal to stop.")

    try:
        # Infinite loop: Runs until user presses Ctrl+C
        while True:
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            if hasattr(env, '_render_frame'):
                env._render_frame()
                
            time.sleep(0.01) 

            # Auto-reset on crash or timeout, but keeps window open
            if terminated or truncated:
                print("Episode finished. Restarting...")
                time.sleep(1)
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()