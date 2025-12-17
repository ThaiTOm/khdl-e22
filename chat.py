# ==============================================================================
# FILE 2: chatbot_app.py
# CHỨC NĂNG: Giao diện Chat, Kết nối Google Gemini, Gọi Recommender Engine
# ==============================================================================

import os
import json
import numpy as np
import pandas as pd
from google import genai

# IMPORT FILE ENGINE (Phải đặt recommender_engine.py cùng thư mục)
try:
    from user_create import CarRecommendationSystem
except ImportError:
    print("❌ LỖI: Không tìm thấy file 'recommender_engine.py'. Hãy tạo file này trước!")
    exit()


class GeminiCarConsultant:
    def __init__(self, rec_system, api_key):
        self.recsys = rec_system
        os.environ["GEMINI_API_KEY"] = api_key
        self.client = genai.Client(api_key=api_key)
        self.text_model = "gemma-3-27b-it"
        print(f"✅ [Chatbot] Đã kết nối GenAI Client ({self.text_model}).")

    def _extract_user_profile(self, user_chat_text):
        """Dùng Gemini để trích xuất thông tin user từ đoạn chat"""
        prompt = f"""
        Phân tích đoạn chat: "{user_chat_text}"
        Trả về JSON thuần (không markdown) với các trường:
        - age: int
        - salary: int (USD/year, note: 5 trieu VND/month ~ 2500 USD/year)
        - is_married: 0/1
        - persona: Chọn 1 trong ["Student", "Family", "Executive"]

        Ví dụ: {{"age": 22, "salary": 3000, "is_married": 0, "persona": "Student"}}
        """
        try:
            response = self.client.models.generate_content(model=self.text_model, contents=prompt)
            text_res = response.text.replace('```json', '').replace('```', '').strip()
            return json.loads(text_res)
        except Exception as e:
            print(f"⚠️ Lỗi trích xuất profile: {e}")
            return {"age": 30, "salary": 30000, "is_married": 1, "persona": "Family"}

    def _generate_consultation(self, user_chat, cars_df):
        """Dùng Gemini để viết lời tư vấn bán hàng"""
        cars_context = ""
        for _, row in cars_df.iterrows():
            cars_context += f"- {row['make']} {row['clean_type']} ({row['year']}), {row['n_seats']} chỗ, {row['clean_trans']}. Giá code: {row['price_code']}\n"

        prompt = f"""
        Khách hàng: "{user_chat}"
        Hệ thống gợi ý các xe sau:
        {cars_context}

        Đóng vai chuyên gia bán xe, hãy tư vấn ngắn gọn, thuyết phục bằng Tiếng Việt.
        Giải thích tại sao các xe này hợp (dựa vào spec xe và nhu cầu khách).
        """
        print("\n🤖 AI Consultant: ", end="")
        try:
            for chunk in self.client.models.generate_content_stream(model=self.text_model, contents=prompt):
                print(chunk.text, end="")
        except Exception as e:
            print(f"Lỗi GenAI: {e}")
        print("\n")

    def chat(self, user_input):
        print("=" * 60)
        print(f"👤 USER: {user_input}")

        # 1. Trích xuất Profile
        profile = self._extract_user_profile(user_input)
        print(f"🔍 Profile trích xuất: {profile}")

        # 2. Gọi Engine để lấy xe gợi ý
        recommended_cars = self.recsys.recommend(profile, top_k=3)
        print("✅ Xe gợi ý:")
        print(recommended_cars)
        if recommended_cars.empty:
            print("❌ Không tìm thấy xe phù hợp.")
            return

        # 3. Sinh lời tư vấn
        self._generate_consultation(user_input, recommended_cars)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # ⚠️ ĐIỀN API KEY CỦA BẠN VÀO ĐÂY
    MY_API_KEY = "AIzaSyCYdKB0FU82B3m0JJ5zKEP4EVgp1vBgY7Q"

    # --- TẠO DỮ LIỆU GIẢ (NẾU CHƯA CÓ FILE CSV) ---
    csv_file = "dt_car_train_2021.csv"
    # --- KHỞI CHẠY HỆ THỐNG ---
    print("\n--- KHỞI ĐỘNG HỆ THỐNG GỢI Ý XE ---")

    # 1. Khởi tạo Engine (Load Model)
    engine = CarRecommendationSystem(csv_path=csv_file)

    # 2. Khởi tạo Chatbot
    bot = GeminiCarConsultant(engine, api_key=MY_API_KEY)

    # 3. Test Chat
    print("\n--- BẮT ĐẦU CHAT ---")
    bot.chat("Mình là sinh viên mới ra trường, muốn tìm xe nhỏ gọn, tiết kiệm xăng để đi làm.")

    # bot.chat("Tôi cần tìm xe sang trọng cho giám đốc, đời mới, không quan trọng giá.")