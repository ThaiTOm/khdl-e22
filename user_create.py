# ==============================================================================
# FILE 1: recommender_engine.py
# CHỨC NĂNG: Xử lý dữ liệu, Train Model (SVD + PyTorch), Tính toán gợi ý
# ==============================================================================

import os
import re
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset as SurpriseDataset, Reader, SVD, dump

# Cấu hình thiết bị (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CarDataProcessor:
    """Xử lý dữ liệu xe từ file CSV thật"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.scaler = MinMaxScaler()

    def process(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"❌ Không tìm thấy file {self.file_path}")

        print(f"1. [Engine] Đang đọc dữ liệu xe từ: {self.file_path}")
        df = pd.read_csv(self.file_path)

        # --- Cleaning ---
        # 1. Price: "5 (10K-15K)" -> 5
        df['price_code'] = df['price_category'].apply(
            lambda x: int(re.search(r'^(\d+)', str(x)).group(1)) if re.search(r'^(\d+)', str(x)) else 3)

        # 2. Transmission
        df['is_automatic'] = df['transmission'].astype(str).str.lower().apply(lambda x: 1 if 'automaat' in x else 0)
        df['clean_trans'] = df['is_automatic'].apply(lambda x: 'Automatic' if x else 'Manual')

        # 3. Clean numeric columns
        num_cols = ['year', 'mileage', 'power', 'n_doors', 'displacement', 'acceleration', 'top_speed', 'n_seats']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())

        # 4. Clean text for display
        df['make'] = df['make'].str.upper()
        df['clean_type'] = df['car_type'].str.capitalize()

        # 5. Normalize features for Content-Based Similarity
        feature_cols = ['price_code', 'year', 'power', 'mileage', 'acceleration', 'n_seats']
        df[feature_cols] = df[feature_cols].fillna(0)
        scaled_features = self.scaler.fit_transform(df[feature_cols])

        # Add normalized columns
        df_features = pd.DataFrame(scaled_features, columns=[f"norm_{c}" for c in feature_cols])
        df = pd.concat([df, df_features], axis=1)

        return df


class PersonaGenerator:
    """Sinh dữ liệu User giả lập để Train model"""

    def __init__(self, df_cars, num_users=1000):
        self.df_cars = df_cars
        self.num_users = num_users

    def generate_ratings(self):
        print("2. [Engine] Đang sinh Ratings giả lập theo Persona...")
        users = []
        ratings = []
        personas = ['Budget_Student', 'Family_Man', 'Luxury_Boss', 'Speed_Racer']

        for uid in range(self.num_users):
            p = np.random.choice(personas)
            users.append({'user_id': uid, 'persona': p})

            # Rate 20 xe
            sample_cars = self.df_cars.sample(n=20)
            for _, car in sample_cars.iterrows():
                score = 3.0
                noise = np.random.uniform(-0.5, 0.5)

                # Logic chấm điểm giả lập
                if p == 'Budget_Student':
                    if car['price_code'] <= 2: score += 2.0
                    if car['mileage'] > 200000: score -= 0.5
                elif p == 'Family_Man':
                    if car['n_seats'] >= 5 and car['car_type'] in ['stationwagon', 'mpv', 'suv']: score += 2.0
                    if car['year'] > 2015: score += 0.5
                elif p == 'Luxury_Boss':
                    if car['make'] in ['MERCEDES-BENZ', 'BMW', 'AUDI', 'LEXUS']: score += 1.5
                    if car['price_code'] >= 5: score += 1.0
                elif p == 'Speed_Racer':
                    if car['power'] > 150: score += 1.5
                    if car['acceleration'] < 8.0: score += 1.0

                final_score = np.clip(score + noise, 1, 5)
                ratings.append({'user_id': uid, 'car_id': car['id'], 'rating': final_score, 'persona': p})

        return pd.DataFrame(ratings), pd.DataFrame(users)


# --- PYTORCH MODEL ---
class TwoTowerNet(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(TwoTowerNet, self).__init__()
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.user_layers = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 32))
        self.item_emb = nn.Embedding(num_items, embedding_dim)
        self.item_layers = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 32))

    def forward(self, u, i):
        return (self.user_layers(self.user_emb(u)) * self.item_layers(self.item_emb(i))).sum(dim=1)


class RatingDataset(Dataset):
    def __init__(self, u, i, r):
        self.u = torch.tensor(u, dtype=torch.long)
        self.i = torch.tensor(i, dtype=torch.long)
        self.r = torch.tensor(r, dtype=torch.float32)

    def __len__(self): return len(self.u)

    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.r[idx]


# --- MAIN RECOMMENDER CLASS ---
class CarRecommendationSystem:
    def __init__(self, csv_path="dt_car_train_2021.csv", checkpoint_dir="checkpoints"):
        self.cp_dir = checkpoint_dir
        if not os.path.exists(self.cp_dir): os.makedirs(self.cp_dir)

        self.path_ratings = f"{self.cp_dir}/ratings_gen.csv"
        self.path_svd = f"{self.cp_dir}/svd.pkl"
        self.path_torch = f"{self.cp_dir}/twotower.pth"

        # 1. Load Data
        self.processor = CarDataProcessor(csv_path)
        self.df_cars = self.processor.process()

        # 2. Ratings & Encoders
        self._prepare_data()

        # 3. Load/Train Models
        self._load_or_train()

    def _prepare_data(self):
        # Load Ratings
        if os.path.exists(self.path_ratings):
            print("2. [Engine] Load ratings cũ từ cache...")
            self.df_ratings = pd.read_csv(self.path_ratings)
        else:
            gen = PersonaGenerator(self.df_cars)
            self.df_ratings, self.df_users = gen.generate_ratings()
            self.df_ratings.to_csv(self.path_ratings, index=False)

        # Encoders
        self.u_enc = LabelEncoder()
        self.i_enc = LabelEncoder()
        self.df_ratings['u_idx'] = self.u_enc.fit_transform(self.df_ratings['user_id'])
        self.i_enc.fit(self.df_cars['id'])  # Fit all cars

        # Valid ratings only
        self.df_ratings = self.df_ratings[self.df_ratings['car_id'].isin(self.df_cars['id'])]
        self.df_ratings['i_idx'] = self.i_enc.transform(self.df_ratings['car_id'])

        self.n_users = len(self.u_enc.classes_)
        self.n_items = len(self.i_enc.classes_)

        # Content Similarity (Optional for cold start items)
        # self.sim_matrix = ... (Có thể thêm nếu cần logic content-based thuần túy)

    def _load_or_train(self):
        if os.path.exists(self.path_svd) and os.path.exists(self.path_torch):
            print("3. [Engine] ✅ Tìm thấy Checkpoint. Load models...")
            _, self.svd = dump.load(self.path_svd)
            self.torch_model = TwoTowerNet(self.n_users, self.n_items).to(device)
            self.torch_model.load_state_dict(torch.load(self.path_torch, map_location=device))
            self.torch_model.eval()
        else:
            print("3. [Engine] ⚠️ Chưa có model. Bắt đầu Train...")
            # SVD
            reader = Reader(rating_scale=(1, 5))
            data = SurpriseDataset.load_from_df(self.df_ratings[['user_id', 'car_id', 'rating']], reader)
            self.svd = SVD(n_factors=50, n_epochs=20)
            self.svd.fit(data.build_full_trainset())
            dump.dump(self.path_svd, algo=self.svd)

            # PyTorch
            self.torch_model = TwoTowerNet(self.n_users, self.n_items).to(device)
            self.torch_model.train()
            ds = RatingDataset(self.df_ratings['u_idx'].values, self.df_ratings['i_idx'].values,
                               self.df_ratings['rating'].values)
            dl = DataLoader(ds, batch_size=64, shuffle=True)
            opt = optim.Adam(self.torch_model.parameters(), lr=0.001)
            cri = nn.MSELoss()

            print("   -> Training Neural Network...")
            for epoch in range(5):
                for u, i, r in dl:
                    u, i, r = u.to(device), i.to(device), r.to(device)
                    opt.zero_grad()
                    loss = cri(self.torch_model(u, i), r)
                    loss.backward()
                    opt.step()
            torch.save(self.torch_model.state_dict(), self.path_torch)
            print("-> Train xong & Đã lưu.")

    def recommend(self, profile_dict, top_k=5):
        """
        Nhận profile dict từ Chatbot, map sang Persona User, và trả về gợi ý
        """
        # 1. Map Gemini Profile -> Internal Persona
        gemini_persona = profile_dict.get('persona', 'Student')
        mapping = {
            'Student': 'Budget_Student',
            'Family': 'Family_Man',
            'Executive': 'Luxury_Boss'
        }
        target_persona = mapping.get(gemini_persona, 'Family_Man')

        # 2. Tìm User đại diện (Proxy User)
        proxy_users = self.df_ratings[self.df_ratings['persona'] == target_persona]['user_id'].unique()
        user_id = proxy_users[0] if len(proxy_users) > 0 else self.df_ratings['user_id'].iloc[0]

        return self._predict_internal(user_id, top_k)

    def _predict_internal(self, user_id, top_k):
        # Lấy ứng viên (loại bỏ xe đã xem)
        viewed = self.df_ratings[self.df_ratings['user_id'] == user_id]['car_id'].tolist()
        candidates = [x for x in self.df_cars['id'].tolist() if x not in viewed]

        # Lấy mẫu 100 xe để dự đoán cho nhanh
        if len(candidates) > 100: candidates = np.random.choice(candidates, 100, replace=False)

        # Predict
        u_idx = self.u_enc.transform([user_id])[0]
        try:
            c_idxs = self.i_enc.transform(candidates)
        except:
            # Fallback nếu có xe lạ chưa từng thấy khi train
            return self.df_cars.sample(top_k)

        u_tensor = torch.tensor([u_idx] * len(candidates)).to(device)
        c_tensor = torch.tensor(c_idxs).to(device)

        with torch.no_grad():
            dl_scores = self.torch_model(u_tensor, c_tensor).cpu().numpy()

        results = []
        for idx, car_id in enumerate(candidates):
            # Hybrid: 50% SVD + 50% Two-Tower
            svd_s = self.svd.predict(user_id, car_id).est
            dl_s = float(dl_scores[idx])
            final = 0.5 * svd_s + 0.5 * dl_s
            results.append({'id': car_id, 'score': final})

        res_df = pd.DataFrame(results).sort_values('score', ascending=False).head(top_k)
        return pd.merge(res_df, self.df_cars, on='id')