import requests
import pandas as pd
import numpy as np
import time
from IPython.display import display

# ============================================================
# 實驗一
# 選用特徵：
#   - confirmed_at（帳號確認/註冊時間）
#   - level1_finished_at（KYC Level1 完成時間）
#   - level2_finished_at（KYC Level2 完成時間）
#   - user_source（用戶來源）
#   - status（帳號狀態）
#   - created_at（帳戶建立時間，併入作為輔助）
#   - kyc_duration_sec：KYC 完成距註冊時間 (level2_finished_at - confirmed_at)
#   - large_trade_after_reg_sec：註冊後多久出現首筆大額交易
#   - high_freq_after_reg_sec：註冊後多久出現首次高頻交易
#   - trade_after_reg_sec：首次交易距註冊時間
# ============================================================

# 設定 API 基礎路徑
BASE_URL = "https://aws-event-api.bitopro.com"

def fetch_table_paginated(name, batch_size=50000):
    """
    分頁抓取指定資料表，每次抓 batch_size 筆，直到無資料為止。
    回傳合併後的完整 DataFrame。
    """
    all_dfs = []
    offset = 0
    while True:
        url = f"{BASE_URL}/{name}?limit={batch_size}&offset={offset}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if len(data) == 0:
            print(f"{name}: 抓取完成，共 {offset} 筆以上。")
            break
        df_batch = pd.DataFrame(data)
        all_dfs.append(df_batch)
        print(f"{name}: 已抓取 {len(df_batch)} 筆, offset={offset}")
        offset += batch_size
        time.sleep(0.2)
    return pd.concat(all_dfs, ignore_index=True)

# ============================================================
# Step 1: 抓取必要的資料表
# ============================================================
user_info = fetch_table_paginated("user_info")
twd_transfer = fetch_table_paginated("twd_transfer")
crypto_transfer = fetch_table_paginated("crypto_transfer")
train_label = fetch_table_paginated("train_label")
predict_label = fetch_table_paginated("predict_label")

# ============================================================
# Step 2: 時間欄位轉型
# ============================================================
# user_info 的時間欄位
for col in ["confirmed_at", "level1_finished_at", "level2_finished_at", "created_at"]:
    if col in user_info.columns:
        user_info[col] = pd.to_datetime(user_info[col], errors="coerce")

# twd_transfer 時間與金額轉換（ori_samount 為整數，需除以 1e8 還原實際金額）
twd_transfer["created_at"] = pd.to_datetime(twd_transfer["created_at"], errors="coerce")
twd_transfer["amount"] = twd_transfer["ori_samount"] * 1e-8

# crypto_transfer 時間與金額轉換
crypto_transfer["created_at"] = pd.to_datetime(crypto_transfer["created_at"], errors="coerce")
crypto_transfer["amount"] = crypto_transfer["ori_samount"] * 1e-8

# ============================================================
# Step 3: 建立基礎 User 特徵（user_info 中的原始欄位）
# 包含 status 與 created_at
# ============================================================
# 先印出 user_info 實際欄位，方便確認
print("user_info columns:", user_info.columns.tolist())

# 只取實際存在的欄位（避免 KeyError）
wanted_cols = ["user_id", "confirmed_at", "level1_finished_at", "level2_finished_at",
               "user_source", "status", "created_at"]
base_cols = [c for c in wanted_cols if c in user_info.columns]
print("使用的欄位:", base_cols)

feature_df = user_info[base_cols].copy()

# ============================================================
# Step 4: 衍生特徵 — KYC 完成距註冊時間
# kyc_duration_sec = level2_finished_at - confirmed_at（秒）
# ============================================================
feature_df["kyc_duration_sec"] = (
    feature_df["level2_finished_at"] - feature_df["confirmed_at"]
).dt.total_seconds()

# ============================================================
# Step 5: 找出每個 user 的首筆交易時間（法幣 & 虛擬幣）
# ============================================================
twd_first = (
    twd_transfer.groupby("user_id")["created_at"]
    .min()
    .reset_index()
    .rename(columns={"created_at": "first_twd_time"})
)
crypto_first = (
    crypto_transfer.groupby("user_id")["created_at"]
    .min()
    .reset_index()
    .rename(columns={"created_at": "first_crypto_time"})
)

# ============================================================
# Step 6: 找出首筆大額交易時間
# 定義大額：TWD 金額 > 100,000
# ============================================================
large_twd = (
    twd_transfer[twd_transfer["amount"] > 100000]
    .groupby("user_id")["created_at"]
    .min()
    .reset_index()
    .rename(columns={"created_at": "first_large_twd"})
)

# ============================================================
# Step 7: 找出首次高頻交易時間
# 定義高頻：同一 user 在 1 小時內有 >= 5 筆交易
# ============================================================
# 對 twd_transfer 按 user_id 與時間排序，計算滾動視窗內的交易筆數
twd_sorted = twd_transfer.sort_values(["user_id", "created_at"]).copy()
twd_sorted = twd_sorted.set_index("created_at")

high_freq_times = []
for uid, group in twd_sorted.groupby("user_id"):
    # 用滾動 1 小時視窗計算每個時間點前 1 小時的交易筆數
    rolling_count = group["amount"].rolling("1h").count()
    # 找出第一次達到高頻門檻（>=5 筆）的時間點
    high_freq_mask = rolling_count >= 5
    if high_freq_mask.any():
        first_high_freq_time = rolling_count[high_freq_mask].index.min()
        high_freq_times.append({"user_id": uid, "first_high_freq_time": first_high_freq_time})

high_freq_df = pd.DataFrame(high_freq_times) if high_freq_times else pd.DataFrame(columns=["user_id", "first_high_freq_time"])

# ============================================================
# Step 8: 合併所有衍生特徵至主表
# ============================================================
feature_df = feature_df.merge(twd_first, on="user_id", how="left")
feature_df = feature_df.merge(crypto_first, on="user_id", how="left")
feature_df = feature_df.merge(large_twd, on="user_id", how="left")
feature_df = feature_df.merge(high_freq_df, on="user_id", how="left")

# 衍生特徵：首次交易距註冊時間（取法幣與虛擬幣中較早的那筆）
feature_df["first_trade_time"] = feature_df[["first_twd_time", "first_crypto_time"]].min(axis=1)
feature_df["trade_after_reg_sec"] = (
    feature_df["first_trade_time"] - feature_df["confirmed_at"]
).dt.total_seconds()

# 衍生特徵：註冊後多久出現首筆大額交易（秒）
feature_df["large_trade_after_reg_sec"] = (
    feature_df["first_large_twd"] - feature_df["confirmed_at"]
).dt.total_seconds()

# 衍生特徵：註冊後多久出現首次高頻交易（秒）
feature_df["high_freq_after_reg_sec"] = (
    feature_df["first_high_freq_time"] - feature_df["confirmed_at"]
).dt.total_seconds()

# ============================================================
# Step 9: 整理最終特徵欄位，移除輔助用的 datetime 欄位
# ============================================================
wanted_final = [
    "user_id",
    "user_source",               # 用戶來源（若存在）
    "status",                    # 帳號狀態（若存在）
    "kyc_duration_sec",          # KYC 完成距註冊時間
    "trade_after_reg_sec",       # 首次交易距註冊時間
    "large_trade_after_reg_sec", # 首筆大額交易距註冊時間
    "high_freq_after_reg_sec",   # 首次高頻交易距註冊時間
]
# 只保留實際存在的欄位
final_features = [c for c in wanted_final if c in feature_df.columns]
print("最終特徵欄位:", final_features)
feature_df_final = feature_df[final_features].fillna(0)

display(feature_df_final.head())

# ============================================================
# Step 10: 合併 Label，產出訓練集與測試集
# ============================================================
train_df = train_label.merge(feature_df_final, on="user_id", how="left").fillna(0)
test_df = predict_label.merge(feature_df_final, on="user_id", how="left").fillna(0)

print(f"訓練集形狀: {train_df.shape}")
print(f"測試集形狀: {test_df.shape}")

# ============================================================
# Step 11: 儲存結果
# ============================================================
train_df.to_csv("train_df.csv", index=False)
test_df.to_csv("test_df.csv", index=False)
feature_df_final.to_csv("feature_df.csv", index=False)

print("資料已存出，請前往 Colab-2 進行模型訓練。")
