import torch
import numpy as np
from stable_baselines3 import PPO

# 1. SỬA ĐƯỜNG DẪN Ở ĐÂY: Dùng "../" để lùi ra một thư mục, sau đó vào thư mục models
model_path = "../models/ppo_force_real"

print(f"Loading model: {model_path}...")
model = PPO.load(model_path, device="cpu")
state_dict = model.policy.state_dict()

# Cấu trúc mạng (2 lớp ẩn)
layers = [
    ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.0.bias", "W1", "b1"),
    ("mlp_extractor.policy_net.2.weight", "mlp_extractor.policy_net.2.bias", "W2", "b2"),
    ("action_net.weight", "action_net.bias", "W3", "b3")
]

def format_c_array(name, tensor):
    flat_data = tensor.cpu().numpy().flatten()
    arr_str = ", ".join([f"{x:.6f}f" for x in flat_data])
    return f"const float {name}[{len(flat_data)}] = {{{arr_str}}};\n"

# 2. FILE OUTPUT TỰ ĐỘNG NẰM TRONG STM32: Vì script đang chạy ở trong stm32
output_file = "ppo_weights.h"
with open(output_file, "w") as f:
    f.write("#ifndef PPO_WEIGHTS_H\n#define PPO_WEIGHTS_H\n\n")
    for w_key, b_key, w_name, b_name in layers:
        f.write(format_c_array(w_name, state_dict[w_key]))
        f.write(format_c_array(b_name, state_dict[b_key]))
    f.write("\n#endif\n")

print(f"{output_file} is ready")