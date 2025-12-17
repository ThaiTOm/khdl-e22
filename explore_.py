import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def advanced_eda(file_path):
    print(f"--- BẮT ĐẦU PHÂN TÍCH CHUYÊN SÂU: {file_path} ---\n")
    df = pd.read_csv(file_path)

    # Cấu hình giao diện biểu đồ
    sns.set(style="whitegrid")
    plt.figure(figsize=(20, 15))

    # ==========================================================================
    # 1. XỬ LÝ DỮ LIỆU SƠ BỘ ĐỂ PHÂN TÍCH
    # ==========================================================================
    # Chuyển đổi price_category từ dạng chuỗi "6 (> 15K)" sang số để tính toán
    # Lấy ký tự đầu tiên làm mã phân khúc (1, 2, 3, 4, 5, 6)
    df['price_code'] = df['price_category'].astype(str).str.extract(r'^(\d)').astype(float)

    # Gom nhóm các hãng xe nhỏ lẻ thành 'Other' để biểu đồ đỡ rối
    top_makes = df['make'].value_counts().nlargest(15).index
    df['make_group'] = df['make'].apply(lambda x: x if x in top_makes else 'Other')

    # ==========================================================================
    # 2. PHÂN TÍCH PHÂN PHỐI (UNIVARIATE ANALYSIS)
    # ==========================================================================

    # --- Biểu đồ 1: Phân phối các phân khúc giá ---
    plt.subplot(3, 2, 1)
    ax1 = sns.countplot(y='price_category', data=df, order=df['price_category'].value_counts().index, palette='viridis')
    plt.title('Phân phối số lượng xe theo Phân khúc giá')
    plt.xlabel('Số lượng xe')

    # --- Biểu đồ 2: Top 15 Hãng xe phổ biến nhất ---
    plt.subplot(3, 2, 2)
    sns.countplot(y='make_group', data=df, order=df['make_group'].value_counts().index, palette='magma')
    plt.title('Top 15 Hãng xe phổ biến nhất')
    plt.xlabel('Số lượng xe')

    # --- Biểu đồ 3: Phân phối Năm sản xuất (Year) ---
    plt.subplot(3, 2, 3)
    sns.histplot(df['year'], bins=30, kde=True, color='blue')
    plt.title('Phân phối Năm sản xuất')
    plt.xlabel('Năm')

    # --- Biểu đồ 4: Phân phối Mileage (Số dặm đã đi) ---
    plt.subplot(3, 2, 4)
    sns.histplot(df['mileage'], bins=30, kde=True, color='red')
    plt.title('Phân phối Số dặm (Mileage)')
    plt.xlabel('Mileage')

    # ==========================================================================
    # 3. PHÂN TÍCH TƯƠNG QUAN (BIVARIATE ANALYSIS)
    # ==========================================================================

    # --- Biểu đồ 5: Quan hệ giữa Hãng xe và Phân khúc giá (Boxplot) ---
    # Xem hãng nào thường nằm ở phân khúc giá cao
    plt.subplot(3, 2, 5)

    # Sắp xếp hãng theo giá trung bình giảm dần để dễ nhìn
    sorted_idx = df.groupby('make_group')['price_code'].median().sort_values(ascending=False).index
    sns.boxplot(x='price_code', y='make_group', data=df, order=sorted_idx, palette='coolwarm')
    plt.title('Phân bố Phân khúc giá theo Hãng xe')
    plt.xlabel('Mã giá (1: Rẻ nhất -> 6: Đắt nhất)')

    # --- Biểu đồ 6: Tương quan giữa các chỉ số kỹ thuật (Heatmap) ---
    plt.subplot(3, 2, 6)
    corr_cols = ['year', 'mileage', 'power', 'n_seats', 'price_code', 'displacement', 'dry_weight']
    # Chỉ lấy các cột số và loại bỏ NA
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu', center=0, fmt=".2f")
    plt.title('Ma trận tương quan (Correlation Matrix)')

    plt.tight_layout()
    plt.show()

    # ==========================================================================
    # 4. THỐNG KÊ CHI TIẾT BẰNG SỐ
    # ==========================================================================
    print("\n=== THỐNG KÊ CHI TIẾT ===")

    print("\n1. Tương quan với Giá (Price Code):")
    print(corr_matrix['price_code'].sort_values(ascending=False))
    print("-> Nhận xét: Chỉ số nào ảnh hưởng lớn nhất đến việc xe đắt hay rẻ?")

    print("\n2. Kiểm tra xe đời sâu (Cũ):")
    old_cars = df[df['year'] < 2000]
    print(f"Số lượng xe trước năm 2000: {len(old_cars)} chiếc")
    if len(old_cars) > 0:
        print(old_cars[['make', 'year', 'price_category']].head(5))

    print("\n3. Kiểm tra xe 'Siêu sang' (Phân khúc 6):")
    luxury_cars = df[df['price_code'] == 6]
    print(f"Số lượng xe phân khúc 6 (> 15K): {len(luxury_cars)} chiếc")
    print("Top hãng trong phân khúc này:")
    print(luxury_cars['make'].value_counts().head(5))

    print("\n4. Kiểm tra 'Power' (Mã lực):")
    print(df['power'].describe())

    print("\n5. Logic Hộp số (Transmission):")
    print(df['transmission'].value_counts())


if __name__ == "__main__":
    # Thay đường dẫn file của bạn vào đây
    advanced_eda("dt_car_train_2021.csv")