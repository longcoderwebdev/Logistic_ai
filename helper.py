import matplotlib
# Đổi backend sang 'Agg' để chạy ngầm, loại bỏ hoàn toàn lỗi Tkinter
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

def plot_training_progress(scores, mean_scores):
    """
    Hàm vẽ biểu đồ và lưu thành ảnh file.
    """
    plt.clf() 
    plt.title('Biểu đồ Huấn luyện Deep Q-Network')
    plt.xlabel('Số trận (Games)')
    plt.ylabel('Điểm số (Score)')
    
    # Vẽ 2 đường đồ thị
    plt.plot(scores, label='Điểm mỗi trận', color='green', alpha=0.6)
    plt.plot(mean_scores, label='Điểm trung bình', color='red', linewidth=2)
    
    plt.ylim(ymin=0)
    
    # Hiển thị con số ở cuối đường đồ thị
    if len(scores) > 0:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
        plt.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
        
    plt.legend(loc='upper left')
    
    # LƯU ẢNH THAY VÌ HIỂN THỊ CỬA SỔ
    plt.savefig('training_chart.png')