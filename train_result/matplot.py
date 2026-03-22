import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Thiết lập style đồ thị chuẩn học thuật
sns.set_theme(style="whitegrid", context="paper")

def plot_all_metrics(folder_path):
    # Lấy danh sách tất cả file CSV trong thư mục
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Sắp xếp để rollout lên đầu, train xếp sau cho đúng logic
    csv_files.sort(key=lambda x: 'rollout' not in x.lower()) 

    # Tạo figure gồm lưới 4 hàng x 3 cột
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()

    for i, file_path in enumerate(csv_files):
        if i >= 12: break # Đề phòng có thừa file
        
        df = pd.read_csv(file_path)
        ax = axes[i]
        
        # Trích xuất tên biểu đồ từ tên file (ví dụ: cắt lấy 'ep_len_mean')
        filename = os.path.basename(file_path)
        title_raw = filename.replace('.csv', '').split('tag-')[-1]
        title_clean = title_raw.replace('_', ' ').title()

        # Vẽ dữ liệu thô (màu nhạt)
        ax.plot(df['Step'], df['Value'], color='#1f77b4', alpha=0.2, label='Raw')
        
        # Vẽ dữ liệu đã làm mượt (Smoothing giống hệt TensorBoard)
        # alpha=0.1 tương đương với smoothing 0.6 trên TensorBoard
        smoothed = df['Value'].ewm(alpha=0.1).mean() 
        ax.plot(df['Step'], smoothed, color='#0a3d6e', linewidth=2, label='Smoothed')
        
        # Căn chỉnh tiêu đề và các trục
        ax.set_title(title_clean, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Timesteps', fontsize=11)
        ax.set_ylabel('Value', fontsize=11)
        
        # Định dạng trục X thành ngàn (k) cho dễ nhìn
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}k".format(int(x/1000))))
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Thêm chú thích chung cho toàn bộ figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=14, bbox_to_anchor=(0.5, 1.02))

    # Tự động căn chỉnh khoảng cách giữa các đồ thị
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 1. Xuất file PDF (Chuẩn để chèn vào Word/LaTeX)
    pdf_path = os.path.join(folder_path, "PPO_Training_Metrics_Report.pdf")
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    
    # 2. Xuất thêm file PNG độ phân giải cao 300 DPI
    png_path = os.path.join(folder_path, "PPO_Training_Metrics_Report.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"Đã xuất đồ thị thành công tại:\n- {pdf_path}\n- {png_path}")
    plt.show()

if __name__ == "__main__":
    # Điền đường dẫn tới thư mục chứa 12 file CSV của bạn
    # Dựa vào ảnh bạn gửi, đường dẫn là:
    folder_dir = "/home/cat21/Documents/lab/simulation&code/train_result/vel_control" 
    plot_all_metrics(folder_dir)