import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
from train import CartPoleSwingUpEnv

SEEDS = [10, 20, 30, 40, 50]
NUM_EPISODES_PER_SEED = 50 

def save_episode_results(seed, episode_num, history_theta, history_action, history_x, swing_up_step, overshoot, jitter_score):
    result_dir = f"test_results/stack 1/sb3_data_seed_{seed}"
    os.makedirs(result_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{result_dir}/SB3_Stack8_Ep{episode_num}_{timestamp}"
    
    data_to_save = np.column_stack((history_theta, history_action, history_x))
    np.savetxt(f"{base_filename}_data.csv", data_to_save, delimiter=",", header="Theta(deg),Action,CartX", comments="")
    
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312, sharex=ax1)
    ax3 = fig.add_subplot(313, sharex=ax1)
    
    fig.suptitle(f"SB3 Baseline (Seed={seed}) - Episode {episode_num}\n"
                 f"Swing-up: {swing_up_step} steps | Overshoot: {overshoot if isinstance(overshoot, str) else f'{overshoot:.2f}'}° | Jitter: {jitter_score:.4f}", 
                 fontsize=12, fontweight='bold')
    
    ax1.plot(history_theta, color='blue', linewidth=1.5)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Pole Angle (deg)')
    
    ax2.plot(history_action, color='orange', linewidth=1)
    ax2.set_ylabel('Action (Force)')
    
    ax3.plot(history_x, color='green', linewidth=1.5)
    ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Cart Position (m)')
    ax3.set_xlabel('Steps')
    
    fig.tight_layout()
    fig.savefig(f"{base_filename}_plot.png", dpi=300)
    plt.close(fig) 

def main():
    global_success_count = 0
    total_episodes_all_seeds = len(SEEDS) * NUM_EPISODES_PER_SEED

    for seed in SEEDS:
        # target final models explicitly
        model_path = f"models/stack 1/ppo_force_real_stack1_seed_{seed}.zip"
        stats_path = f"vec/stack 1/vec_normalize_stack1_seed_{seed}.pkl"

        if not os.path.exists(model_path) or not os.path.exists(stats_path):
            continue

        raw_env = CartPoleSwingUpEnv(render_mode="none")
        vec_env = DummyVecEnv([lambda: raw_env])
        vec_env = VecNormalize.load(stats_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        vec_env = VecFrameStack(vec_env, n_stack=1)

        # [ ... trong hàm main() ... ]
        model = PPO.load(model_path, env=vec_env, device="cpu")
        seed_success_count = 0

        print(f"\n>>> STARTING SEED {seed} <<<")

        for ep in range(1, NUM_EPISODES_PER_SEED + 1):
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
                done = bool(dones[0])
                if done:
                    is_truncated = bool(infos[0].get("TimeLimit.truncated", False))

            if is_truncated: 
                seed_success_count += 1
                global_success_count += 1
                outcome = "Success"
            else:
                outcome = "Fail"
            
            # calculate metrics
            history_theta = np.array(history_theta)
            history_action = np.array(history_action)
            # calculate error based on target 180 degrees
            theta_error = np.abs(np.abs(history_theta) - 180.0)
            stable_indices = np.where(theta_error < 10)[0]
            
            if len(stable_indices) > 0:
                swing_up_step = stable_indices[0]
                overshoot = np.max(theta_error[swing_up_step:])
            else:
                swing_up_step = "N/A"
                overshoot = "N/A"
            jitter_score = np.mean(np.abs(np.diff(history_action)))
            
            save_episode_results(seed, ep, history_theta, history_action, np.array(history_x), swing_up_step, overshoot, jitter_score)
            print(f"Seed {seed} - Ep {ep}/{NUM_EPISODES_PER_SEED} -> {outcome}")

        success_rate = (seed_success_count / NUM_EPISODES_PER_SEED) * 100
        print(f">>> SEED {seed} COMPLETE | SUCCESS RATE: {success_rate:.1f}% <<<\n")
        vec_env.close()

if __name__ == "__main__":
    main()