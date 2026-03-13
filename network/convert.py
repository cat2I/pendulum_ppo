import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from stable_baselines3 import PPO

# 1. Load mô hình
model = PPO.load("models/ppo_cartpole_swingup.zip", device="cpu")

# 2. BƯỚC QUAN TRỌNG: Tạo một lớp bọc để vứt bỏ phần "xác suất" của PPO
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        # Trích xuất đúng cái bộ não xử lý (chứa 2 lớp ẩn 128)
        self.mlp_extractor = policy.mlp_extractor
        # Trích xuất lớp xuất dữ liệu cuối cùng (chứa 1 node output)
        self.action_net = policy.action_net

    def forward(self, observation):
        # Truyền dữ liệu qua mạng MLP lõi (bỏ qua hàm Value)
        action_hidden, _ = self.mlp_extractor(observation)
        # Tính ra lực đẩy cụ thể
        return self.action_net(action_hidden)

# 3. Khởi tạo mô hình lõi đã được bóc tách
onnxable_model = OnnxablePolicy(model.policy)
onnxable_model.eval()

# 4. Tạo dữ liệu giả lập (vector 5 trạng thái)
dummy_input = torch.randn(1, 5)

# 5. Xuất file ONNX
torch.onnx.export(
    onnxable_model,
    dummy_input,
    "models/ppo_model.onnx",
    export_params=True,
    opset_version=14,
    input_names=['state_input'],
    output_names=['action_output']
)

print("Converted!")