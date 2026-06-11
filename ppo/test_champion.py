import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from train import CartPoleSwingUpEnv

# read champion seed from file
def get_champion_seed():
    try:
        with open("test_results/stack 1/champion_seed.txt", "r") as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return 40

CHAMPION_SEED = get_champion_seed()
NUM_EPISODES = 20 

def save_champion_results(episode_num, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome):
    result_dir = f"test_results/stack 1/champion_seed_{CHAMPION_SEED}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{result_dir}/Ep{episode_num}_{outcome}_{timestamp}"
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # format detailed title
    fig.suptitle(f"Champion Seed {CHAMPION_SEED} - {outcome}\n"
                 f"Swing-up: {swing_up} steps | Overshoot: {overshoot if isinstance(overshoot, str) else f'{overshoot:.2f}'}° | Jitter: {jitter:.4f}", 
                 fontsize=12, fontweight='bold')
    
    ax1.plot(history_theta, color='blue')
    # reference lines for target 180 degrees
    ax1.axhline(180, color='red', linestyle='--')
    ax1.axhline(-180, color='red', linestyle='--')
    
    ax2.plot(history_action, color='orange')
    ax3.plot(history_x, color='green')
    
    plt.tight_layout()
    fig.savefig(f"{base_filename}_plot.png", dpi=300)
    plt.close(fig)

def main():
    model_path = f"models/stack 1/ppo_force_real_stack1_seed_{CHAMPION_SEED}.zip"
    stats_path = f"vec/stack 1/vec_normalize_stack1_seed_{CHAMPION_SEED}.pkl"

    raw_env = CartPoleSwingUpEnv(render_mode="none")
    vec_env = DummyVecEnv([lambda: raw_env])
    vec_env = VecNormalize.load(stats_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    vec_env = VecFrameStack(vec_env, n_stack=1)

    model = PPO.load(model_path, env=vec_env, device="cpu")
    
    trunc_count, term_count = 0, 0

    for ep in range(1, NUM_EPISODES + 1):
        obs = vec_env.reset()
        history_theta, history_action, history_x = [], [], []
        done = False
        is_truncated = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            unnorm_obs = vec_env.get_original_obs()[0]
            current_raw_obs = unnorm_obs[-5:]
            
            history_theta.append(np.degrees(np.arctan2(current_raw_obs[3], current_raw_obs[2])))
            history_action.append(action[0][0])
            history_x.append(current_raw_obs[0])

            obs, _, dones, infos = vec_env.step(action)
            
            # force boolean casting
            done = bool(dones[0])
            if done:
                is_truncated = bool(infos[0].get("TimeLimit.truncated", False))

        history_theta = np.array(history_theta)
        history_action = np.array(history_action)
        
        jitter = np.mean(np.abs(np.diff(history_action)))
        
        # calculate error based on 180 degrees target
        theta_error = np.abs(np.abs(history_theta) - 180.0)
        stable_idx = np.where(theta_error < 10)[0]
        swing_up = stable_idx[0] if len(stable_idx) > 0 else "N/A"
        overshoot = np.max(theta_error[swing_up:]) if isinstance(swing_up, (int, np.integer)) else "N/A"

        if is_truncated:
            trunc_count += 1
            outcome = "Truncated"
        else:
            term_count += 1
            outcome = "Terminated"

        save_champion_results(ep, history_theta, history_action, history_x, swing_up, overshoot, jitter, outcome)

    # console summary
    print(f"Truncated: {trunc_count} | Terminated: {term_count}")

if __name__ == "__main__":
    main()