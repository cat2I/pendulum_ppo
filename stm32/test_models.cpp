#include <iostream>
#include <chrono>
#include "ppo_weights.h"

#define IN_FEATURES 5
#define HIDDEN_1 64
#define HIDDEN_2 64
#define OUT_FEATURES 1

void dense_layer(const float* input, const float* weight, const float* bias, 
                 float* output, int in_dim, int out_dim, bool use_relu) {
    for (int i = 0; i < out_dim; i++) {
        output[i] = bias[i];
        for (int j = 0; j < in_dim; j++) {
            output[i] += weight[i * in_dim + j] * input[j];
        }
        if (use_relu && output[i] < 0.0f) {
            output[i] = 0.0f; // Hàm kích hoạt ReLU
        }
    }
}

float ppo_predict(const float* state) {
    float h1[HIDDEN_1];
    float h2[HIDDEN_2];
    float action[OUT_FEATURES];

    dense_layer(state, W1, b1, h1, IN_FEATURES, HIDDEN_1, true);
    dense_layer(h1, W2, b2, h2, HIDDEN_1, HIDDEN_2, true);
    dense_layer(h2, W3, b3, action, HIDDEN_2, OUT_FEATURES, false);

    return action[0]; 
}

int main() {
    // 1. Tạo vector trạng thái giả lập (Ví dụ: con lắc nghiêng 0.1 rad)
    // [vị trí x, vận tốc x, góc nghiêng, vận tốc góc, lực cũ]
    float test_state[5] = {0.0f, 0.0f, 0.1f, 0.0f, 0.0f};

    std::cout << "Dang tinh toan..." << std::endl;

    // 2. Bắt đầu bấm giờ
    auto start = std::chrono::high_resolution_clock::now();
    
    // 3. Đưa trạng thái vào mạng Nơ-ron để lấy kết quả Lực (Force)
    float force_output = ppo_predict(test_state);
    
    // 4. Kết thúc bấm giờ
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::micro> duration = end - start;

    std::cout << "--- KET QUA TEST C++ ---" << std::endl;
    std::cout << "Action (Luc Force) can thuc hien: " << force_output << std::endl;
    std::cout << "Thoi gian chay cua mang No-ron: " << duration.count() << " micro-giay" << std::endl;

    return 0;
}