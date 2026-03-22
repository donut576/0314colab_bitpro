# -*- coding: utf-8 -*-
"""
build_selected_features.py

整合以下 14 個精選特徵，輸出 train_selected.csv / test_selected.csv：
  1.  first_large_txn_to_confirm_sec  首次大額交易距註冊時間
  2.  first_hf_day_to_confirm_sec     首次高頻交易日距註冊時間
  3.  first_txn_to_confirm_sec        首次交易距註冊時間
  4.  total_amount_twd                總交易金額（台幣）
  5.  avg_gap_sec                     平均交易間隔（秒）
  6.  night_tx_ratio                  夜間交易比例
  7.  avg_from_wallet_share           平均來源錢包共用人數
  8.  to_wallet_reuse_rate            目標錢包重複率
  9.  max_from_wallet_share           最大來源錢包共用人數
  10. ip_share_mean                   平均 IP 共用人數
  11. ip_source_diversity_mean        平均 IP 來源管道多樣性
  12. new_ip_txn_ratio                新 IP 交易比例
  13. trc20_count                     TRC20 交易筆數
  14. overall_small_ratio             全體小額交易比例
  15. trc20_to_wallet_reuse           TRC20 目標錢包重複率
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 1. 載入
# ─────────────────────────────────────────
user_info        = pd.read_csv(f"{DATA_DIR}/user_info.csv")
twd_transfer     = pd.read_csv(f"{DATA_DIR}/twd_transfer.csv")
crypto_transfer  = pd.read_csv(f"{DATA_DIR}/crypto_transfer.csv")
usdt_twd_trading = pd.read_csv(f"{DATA_DIR}/usdt_twd_trading.csv")
usdt_swap        = pd.read_csv(f"{DATA_DIR}/usdt_swap.csv")
train_label      = pd.read_csv(f"{DATA_DIR}/train_label.csv")
predict_label    = pd.read_csv(f"{DATA_DIR}/predict_label.csv")

# 時間轉型
user_info["confirmed_at"]       = pd.to_datetime(user_info["confirmed_at"],       errors="coerce")
twd_transfer["created_at"]      = pd.to_datetime(twd_transfer["created_at"],      errors="coerce")
crypto_transfer["created_at"]   = pd.to_datetime(crypto_transfer["created_at"],   errors="coerce")
usdt_twd_trading["updated_at"]  = pd.to_datetime(usdt_twd_trading["updated_at"],  errors="coerce")
usdt_swap["created_at"]         = pd.to_datetime(usdt_swap["created_at"],         errors="coerce")

# 金額還原（台幣）
twd_transfer["amount_twd"]      = twd_transfer["ori_samount"]        * 1e-8
crypto_transfer["amount_twd"]   = crypto_transfer["ori_samount"]     * 1e-8 * crypto_transfer["twd_srate"]    * 1e-8
usdt_twd_trading["amount_twd"]  = usdt_twd_trading["trade_samount"]  * 1e-8 * usdt_twd_trading["twd_srate"]   * 1e-8
usdt_swap["amount_twd"]         = usdt_swap["twd_samount"]           * 1e-8

# 統一時間欄位
twd_transfer["ts"]      = twd_transfer["created_at"]
crypto_transfer["ts"]   = crypto_transfer["created_at"]
usdt_twd_trading["ts"]  = usdt_twd_trading["updated_at"]
usdt_swap["ts"]         = usdt_swap["created_at"]

# 合併所有交易（含 IP）
all_txn = pd.concat([
    twd_transfer[["user_id", "ts", "amount_twd", "source_ip_hash"]],
    crypto_transfer[["user_id", "ts", "amount_twd", "source_ip_hash"]],
    usdt_twd_trading[["user_id", "ts", "amount_twd", "source_ip_hash"]],
    usdt_swap[["user_id", "ts", "amount_twd"]].assign(source_ip_hash=np.nan),
], ignore_index=True).sort_values(["user_id", "ts"])

feat = user_info[["user_id", "confirmed_at", "user_source"]].copy()

# ─────────────────────────────────────────
# 2. 首次交易距註冊時間
# ─────────────────────────────────────────
first_txn = all_txn.groupby("user_id")["ts"].min().rename("first_txn_ts")
feat = feat.merge(first_txn, on="user_id", how="left")
feat["first_txn_to_confirm_sec"] = (
    feat["first_txn_ts"] - feat["confirmed_at"]
).dt.total_seconds()

# ─────────────────────────────────────────
# 3. 首次大額交易距註冊時間
# ─────────────────────────────────────────
large_threshold = all_txn["amount_twd"].quantile(0.9)
large_txn = all_txn[all_txn["amount_twd"] >= large_threshold]
first_large = large_txn.groupby("user_id")["ts"].min().rename("first_large_ts")
feat = feat.merge(first_large, on="user_id", how="left")
feat["first_large_txn_to_confirm_sec"] = (
    feat["first_large_ts"] - feat["confirmed_at"]
).dt.total_seconds()

# ─────────────────────────────────────────
# 4. 首次高頻交易日距註冊時間
# ─────────────────────────────────────────
all_txn["date"] = all_txn["ts"].dt.date
daily_cnt = all_txn.groupby(["user_id", "date"]).size().reset_index(name="daily_count")
hf_threshold = daily_cnt["daily_count"].quantile(0.9)
hf_days = daily_cnt[daily_cnt["daily_count"] >= hf_threshold].copy()
hf_days["date_dt"] = pd.to_datetime(hf_days["date"])
first_hf = hf_days.groupby("user_id")["date_dt"].min().rename("first_hf_ts")
feat = feat.merge(first_hf, on="user_id", how="left")
feat["first_hf_day_to_confirm_sec"] = (
    feat["first_hf_ts"] - feat["confirmed_at"]
).dt.total_seconds()

# ─────────────────────────────────────────
# 5. total_amount_twd & avg_gap_sec & night_tx_ratio
# ─────────────────────────────────────────
all_txn["hour"]     = all_txn["ts"].dt.hour
all_txn["is_night"] = all_txn["hour"].between(0, 5).astype(int)
all_txn["gap_sec"]  = all_txn.groupby("user_id")["ts"].diff().dt.total_seconds()

txn_agg = all_txn.groupby("user_id").agg(
    total_amount_twd=("amount_twd", "sum"),
    avg_gap_sec     =("gap_sec",    "mean"),
    night_tx_ratio  =("is_night",   "mean"),
).reset_index()
feat = feat.merge(txn_agg, on="user_id", how="left")

# ─────────────────────────────────────────
# 6. 錢包特徵（from/to wallet）
# ─────────────────────────────────────────
ct = crypto_transfer.dropna(subset=["from_wallet_hash"]).copy()

# 每個 from_wallet 被幾個 user 共用
fw_user_cnt = (
    ct.groupby("from_wallet_hash")["user_id"].nunique()
    .rename("fw_global_user_cnt")
)
ct = ct.join(fw_user_cnt, on="from_wallet_hash")

wallet_agg = ct.groupby("user_id").agg(
    avg_from_wallet_share=("fw_global_user_cnt", "mean"),
    max_from_wallet_share=("fw_global_user_cnt", "max"),
).reset_index()

# to_wallet 重複率
ct_to = crypto_transfer.dropna(subset=["to_wallet_hash"]).copy()
to_reuse = ct_to.groupby("user_id")["to_wallet_hash"].apply(
    lambda x: 1 - x.nunique() / (len(x) + 1e-9)
).rename("to_wallet_reuse_rate").reset_index()

feat = feat.merge(wallet_agg, on="user_id", how="left")
feat = feat.merge(to_reuse,   on="user_id", how="left")

# ─────────────────────────────────────────
# 7. IP 特徵
# ─────────────────────────────────────────
ip_txn = all_txn.dropna(subset=["source_ip_hash"]).copy()

# IP 共用人數
ip_user_cnt = (
    ip_txn.groupby("source_ip_hash")["user_id"].nunique()
    .rename("ip_global_user_cnt")
)
ip_txn = ip_txn.join(ip_user_cnt, on="source_ip_hash")

# IP × user_source 多樣性
ip_source = ip_txn.merge(user_info[["user_id", "user_source"]], on="user_id", how="left")
ip_src_div = (
    ip_source.groupby("source_ip_hash")["user_source"].nunique()
    .rename("ip_source_diversity")
)
ip_txn = ip_txn.join(ip_src_div, on="source_ip_hash")

# 新 IP 旗標
ip_txn["ip_first_seen"] = ip_txn.groupby(["user_id", "source_ip_hash"])["ts"].transform("min")
ip_txn["is_new_ip"]     = (ip_txn["ts"] == ip_txn["ip_first_seen"]).astype(int)

ip_agg = ip_txn.groupby("user_id").agg(
    ip_share_mean            =("ip_global_user_cnt",  "mean"),
    ip_source_diversity_mean =("ip_source_diversity", "mean"),
    new_ip_txn_ratio         =("is_new_ip",           "mean"),
).reset_index()

feat = feat.merge(ip_agg, on="user_id", how="left")

# ─────────────────────────────────────────
# 8. TRC20 特徵
# ─────────────────────────────────────────
ct_all = crypto_transfer.copy()
ct_all["is_trc20"] = (ct_all["protocol"] == 4).astype(int)

trc20_count = ct_all.groupby("user_id")["is_trc20"].sum().rename("trc20_count").reset_index()
feat = feat.merge(trc20_count, on="user_id", how="left")

# TRC20 to_wallet 重複率
trc20_only = ct_all[(ct_all["is_trc20"] == 1)].dropna(subset=["to_wallet_hash"])
if not trc20_only.empty:
    trc20_reuse = trc20_only.groupby("user_id")["to_wallet_hash"].apply(
        lambda x: 1 - x.nunique() / (len(x) + 1e-9)
    ).rename("trc20_to_wallet_reuse").reset_index()
    feat = feat.merge(trc20_reuse, on="user_id", how="left")
else:
    feat["trc20_to_wallet_reuse"] = np.nan

# ─────────────────────────────────────────
# 9. 全體小額交易比例
# ─────────────────────────────────────────
small_threshold = all_txn["amount_twd"].quantile(0.25)
all_txn["is_small"] = (all_txn["amount_twd"] <= small_threshold).astype(int)
small_ratio = all_txn.groupby("user_id")["is_small"].mean().rename("overall_small_ratio").reset_index()
feat = feat.merge(small_ratio, on="user_id", how="left")

# ─────────────────────────────────────────
# 10. 整理最終特徵欄位
# ─────────────────────────────────────────
SELECTED_FEATURES = [
    "first_large_txn_to_confirm_sec",
    "first_hf_day_to_confirm_sec",
    "first_txn_to_confirm_sec",
    "total_amount_twd",
    "avg_gap_sec",
    "night_tx_ratio",
    "avg_from_wallet_share",
    "to_wallet_reuse_rate",
    "max_from_wallet_share",
    "ip_share_mean",
    "ip_source_diversity_mean",
    "new_ip_txn_ratio",
    "trc20_count",
    "overall_small_ratio",
    "trc20_to_wallet_reuse",
]

feat_out = feat[["user_id"] + SELECTED_FEATURES].copy()
feat_out[SELECTED_FEATURES] = feat_out[SELECTED_FEATURES].fillna(-1)

print(f"feature_df shape: {feat_out.shape}")
print(feat_out[SELECTED_FEATURES].describe().T[["mean", "min", "max", "count"]])

# ─────────────────────────────────────────
# 11. 輸出
# ─────────────────────────────────────────
train_out = feat_out.merge(train_label,   on="user_id")
test_out  = feat_out.merge(predict_label, on="user_id")

train_out.to_csv(os.path.join(OUT_DIR, "train_selected.csv"), index=False)
test_out.to_csv( os.path.join(OUT_DIR, "test_selected.csv"),  index=False)

print(f"\ntrain_selected.csv : {train_out.shape}")
print(f"test_selected.csv  : {test_out.shape}")
print(f"Saved to {OUT_DIR}")
