from stable_baselines3 import PPO
import time
# Import isolated legacy environment
from old_env import OldCartPoleEnv

def main():
    env = OldCartPoleEnv(render_mode="human")

    print("Loading pre-trained model...")
    # Load legacy AI brain
    model = PPO.load("sim/ppo_cartpole_swingup.zip", env=env)

    obs, info = env.reset()
    print("Start testing... Press Ctrl+C in terminal to stop.")

    try:
        # Inference loop
        while True:
            # deterministic=True for stable evaluation
            action, _states = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            time.sleep(0.01) 

            # Auto-reset sequence
            if terminated or truncated:
                print("Episode finished. Restarting...")
                time.sleep(1)
                obs, info = env.reset()
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()