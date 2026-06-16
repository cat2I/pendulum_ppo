import os
import glob
import numpy as np

BASE_DIR = "test_results/stack 1"
SEEDS = [10, 20, 30, 40, 50]
MAX_STEPS = 1000

def main():
    # buffer report for both console and file output
    report = []
    report.append("="*65)
    report.append(f"{'SEED':<8} | {'SUCCESS RATE':<15} | {'MEAN JITTER':<15} | {'MEAN OVERSHOOT':<15}")
    report.append("-" * 65)

    best_seed = None
    best_success = -1
    best_jitter = float('inf')

    for seed in SEEDS:
        folder = f"{BASE_DIR}/sb3_data_seed_{seed}"
        csv_files = glob.glob(f"{folder}/*_data.csv")
        
        if not csv_files:
            continue
            
        success_count = 0
        seed_jitters = []
        seed_overshoots = []

        for file in csv_files:
            data = np.loadtxt(file, delimiter=',', skiprows=1)
            history_theta = data[:, 0]
            history_action = data[:, 1]
            
            if len(data) >= MAX_STEPS:
                success_count += 1
            
            jitter = np.mean(np.abs(np.diff(history_action)))
            seed_jitters.append(jitter)
            
            # calculate error based on 180 degrees target
            theta_error = np.abs(np.abs(history_theta) - 180.0)
            stable_indices = np.where(theta_error < 10)[0]
            if len(stable_indices) > 0:
                swing_up_step = stable_indices[0]
                overshoot = np.max(theta_error[swing_up_step:])
                seed_overshoots.append(overshoot)

        success_rate = (success_count / len(csv_files)) * 100
        mean_jitter = np.mean(seed_jitters)
        mean_overshoot = np.mean(seed_overshoots) if seed_overshoots else 0.0

        report.append(f"{seed:<8} | {success_rate:>13.1f}% | {mean_jitter:>15.4f} | {mean_overshoot:>14.2f}°")

        if success_rate > best_success:
            best_success = success_rate
            best_jitter = mean_jitter
            best_seed = seed
        elif success_rate == best_success: 
            if mean_jitter < best_jitter:  
                best_jitter = mean_jitter
                best_seed = seed

    report.append("="*65)
    report.append(f"\nCHAMPION SEED: {best_seed}")
    report.append(f"-> Use 'models/stack 1/ppo_force_real_stack1_seed_{best_seed}.zip' for testing.")

    full_report = "\n".join(report)
    print(full_report)

    # export quantitative benchmark table
    os.makedirs(f"{BASE_DIR}/seed_comparison", exist_ok=True)
    with open(f"{BASE_DIR}/seed_comparison/quantitative_benchmark.txt", "w", encoding="utf-8") as f:
        f.write(full_report)

    # save champion seed dynamically
    with open(f"{BASE_DIR}/champion_seed.txt", "w") as f:
        f.write(str(best_seed))

if __name__ == "__main__":
    main()