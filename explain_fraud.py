# -*- coding: utf-8 -*-
"""
explain_fraud.py

對每個被判定為詐騙的用戶，用 SHAP 解釋「是哪些特徵讓他被判詐騙」。

輸出：
  - fraud_explanation.csv   每個詐騙用戶的 top 風險特徵與說明
  - fraud_shap_detail.csv   完整 SHAP 值明細（所有特徵）
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("[ERROR] 請先安裝 shap: pip install shap")
    exit()

# =========================================================
# 設定
# =========================================================
TRAIN_PATH  = "train_feature.csv"
TEST_PATH   = "test_feature.csv"
TARGET_COL  = "status"
ID_COL      = "user_id"
TOP_N       = 5   # 每個用戶列出前 N 個最重要的風險特徵

# 從 ensemble_detail.csv 或 submission 讀取詐騙名單
FRAUD_SOURCE = "ensemble_detail.csv"   # 優先用 ensemble 結果
FALLBACK_SUB = "output_xgb_v2/full/submission.csv"

DROP_TIME_COLS = [
    "confirmed_at", "level1_finished_at", "level2_finished_at",
    "twd_first_time", "twd_last_time", "crypto_first_time", "crypto_last_time",
    "trade_first_time", "trade_last_time", "swap_first_time", "swap_last_time",
    "overall_first_time", "overall_last_time",
]

# 特徵分類對應
FEATURE_CATEGORY = {
    # 一、用戶基本資訊 [KYC]
    "KYC": [
        "lvl1_minus_confirm_sec", "lvl2_minus_confirm_sec", "lvl2_minus_lvl1_sec",
        "lvl1_minus_confirm_sec_log", "lvl2_minus_confirm_sec_log", "lvl2_minus_lvl1_sec_log",
        "lvl1_minus_confirm_sec_pct", "lvl2_minus_confirm_sec_pct", "lvl2_minus_lvl1_sec_pct",
        "kyc_speed_composite", "has_confirmed", "has_level1", "has_level2",
        "age", "sex", "career", "income_source", "user_source",
    ],
    # 二、台幣出入金 [TWD]
    "TWD": [
        "twd_txn_count", "twd_total_amount", "twd_avg_amount", "twd_max_amount",
        "twd_min_amount", "twd_std_amount", "twd_median_amount",
        "twd_deposit_count", "twd_withdraw_count", "twd_deposit_ratio", "twd_withdraw_ratio",
        "twd_withdraw_deposit_ratio", "twd_out_in_amount_ratio",
        "twd_in_total_amount", "twd_in_avg_amount", "twd_in_max_amount",
        "twd_out_total_amount", "twd_out_avg_amount", "twd_out_max_amount",
        "twd_night_ratio", "twd_weekend_ratio", "twd_unique_ip",
        "twd_hour_entropy", "twd_active_days", "twd_txn_per_active_day",
        "twd_active_span_sec", "twd_burstiness",
        "twd_gap_mean_sec", "twd_gap_std_sec", "twd_gap_min_sec", "twd_gap_max_sec",
        "twd_gap_lt_5min_ratio", "twd_gap_lt_1hr_ratio",
        "twd_avg_amount_pct", "twd_high_amount_ratio",
    ],
    # 三、虛擬貨幣轉帳 [CRYPTO]
    "CRYPTO": [
        "crypto_txn_count", "crypto_total_amount", "crypto_avg_amount", "crypto_max_amount",
        "crypto_std_amount", "crypto_total_twd_value", "crypto_avg_twd_value",
        "crypto_max_twd_value", "crypto_std_twd_value",
        "crypto_deposit_count", "crypto_withdraw_count", "crypto_withdraw_ratio",
        "crypto_external_count", "crypto_internal_count", "crypto_external_ratio", "crypto_internal_ratio",
        "crypto_out_in_amount_ratio", "crypto_out_total_twd_value", "crypto_out_avg_twd_value",
        "crypto_in_total_twd_value", "crypto_in_avg_twd_value",
        "crypto_night_ratio", "crypto_weekend_ratio", "crypto_unique_ip",
        "crypto_hour_entropy", "crypto_active_days", "crypto_txn_per_active_day",
        "crypto_active_span_sec", "crypto_burstiness", "crypto_txn_per_wallet",
        "crypto_unique_currency", "crypto_unique_protocol",
        "crypto_gap_mean_sec", "crypto_gap_std_sec", "crypto_gap_min_sec", "crypto_gap_max_sec",
        "crypto_gap_lt_5min_ratio", "crypto_gap_lt_1hr_ratio",
        "crypto_small_twd_ratio", "crypto_self_wallet_transfer_count", "crypto_self_wallet_transfer_ratio",
        "trc20_ratio", "bsc_ratio", "trc20_tx_count", "bsc_tx_count",
        "trc20_night_ratio", "bsc_night_ratio", "trc20_weekend_ratio", "bsc_weekend_ratio",
        "trc20_unique_to_wallet", "bsc_unique_to_wallet",
        "trc20_addr_reuse_rate", "bsc_addr_reuse_rate",
        "trc20_from_addr_reuse_rate", "bsc_from_addr_reuse_rate",
        "trc20_inflow_ratio", "trc20_outflow_ratio", "bsc_inflow_ratio", "bsc_outflow_ratio",
        "trc20_in_out_imbalance", "bsc_in_out_imbalance",
        "protocol_switch_count", "protocol_switch_rate",
        "crypto_avg_twd_pct", "crypto_high_twd_ratio",
    ],
    # 四、撮合交易 [TRADE]
    "TRADE": [
        "trade_count", "trade_total_amount", "trade_avg_amount", "trade_max_amount", "trade_std_amount",
        "trade_avg_price", "trade_max_price", "trade_std_price",
        "trade_buy_count", "trade_sell_count", "trade_buy_ratio", "trade_sell_ratio",
        "trade_market_count", "trade_market_ratio",
        "trade_unique_ip", "trade_source_nunique",
        "trade_night_ratio", "trade_weekend_ratio", "trade_hour_entropy",
        "trade_active_days", "trade_txn_per_active_day", "trade_txn_per_ip",
        "trade_active_span_sec", "trade_burstiness",
        "trade_gap_mean_sec", "trade_gap_std_sec", "trade_gap_min_sec", "trade_gap_max_sec",
        "trade_gap_lt_5min_ratio", "trade_gap_lt_1hr_ratio",
        "trade_amount_cv", "trade_price_cv",
    ],
    # 五、一鍵買賣 [SWAP]
    "SWAP": [
        "swap_count", "swap_total_twd", "swap_avg_twd", "swap_max_twd", "swap_std_twd",
        "swap_total_crypto", "swap_avg_crypto", "swap_max_crypto", "swap_std_crypto",
        "swap_buy_count", "swap_sell_count", "swap_buy_ratio", "swap_sell_ratio",
        "swap_buy_total_twd", "swap_sell_total_twd", "swap_sell_buy_twd_ratio",
        "swap_night_ratio", "swap_weekend_ratio", "swap_hour_entropy",
        "swap_active_days", "swap_txn_per_active_day", "swap_active_span_sec", "swap_burstiness",
        "swap_gap_mean_sec", "swap_gap_std_sec", "swap_gap_min_sec", "swap_gap_max_sec",
        "swap_gap_lt_5min_ratio", "swap_gap_lt_1hr_ratio",
        "swap_twd_cv", "swap_round_10000_ratio",
        "swap_crypto_twd_ratio", "swap_max_to_avg_ratio", "swap_concentration", "swap_sell_speed",
    ],
    # 六、網路特徵 [NETWORK]
    "NETWORK": [
        "network_out_degree", "network_in_degree", "network_total_degree",
        "network_out_txn_count", "network_in_txn_count",
        "network_out_total_value", "network_in_total_value",
        "network_out_in_degree_ratio", "network_out_in_value_ratio", "network_degree_imbalance",
    ],
    # 七、錢包風險 [WALLET]
    "WALLET": [
        "high_risk_to_wallet_count", "high_risk_from_wallet_count",
        "high_risk_to_wallet_ratio", "high_risk_from_wallet_ratio",
        "addr_reuse_rate", "from_addr_reuse_rate",
        "crypto_unique_from_wallet", "crypto_unique_to_wallet",
    ],
    # 八、IP 特徵 [IP]
    "IP": [
        "total_unique_ips", "total_ip_usage_count", "shared_ip_count", "shared_ip_ratio",
        "ip_source_diversity", "ip_jump_rate",
        "twd_unique_ip", "crypto_unique_ip", "trade_unique_ip",
    ],
    # 九、時間異常 [TEMPORAL]
    "TEMPORAL": [
        "avg_time_entropy", "min_time_entropy",
        "twd_hour_entropy_2", "twd_day_entropy", "twd_midnight_ratio", "twd_weekend_ratio_2",
        "crypto_hour_entropy_2", "crypto_day_entropy", "crypto_midnight_ratio", "crypto_weekend_ratio_2",
        "trade_hour_entropy_2", "trade_day_entropy", "trade_midnight_ratio", "trade_weekend_ratio_2",
        "swap_hour_entropy_2", "swap_day_entropy", "swap_midnight_ratio", "swap_weekend_ratio_2",
    ],
    # 十、金額異常 [AMOUNT]
    "AMOUNT": [
        "twd_round_1000_ratio", "twd_round_10000_ratio", "twd_amount_cv", "twd_amount_skew",
        "crypto_twd_value_cv", "crypto_twd_value_skew",
        "trade_amount_cv", "trade_price_cv",
        "swap_twd_cv", "swap_round_10000_ratio",
        "iforest_score", "iforest_score_behav",
    ],
    # 十一、資金流動 [FLOW]
    "FLOW": [
        "overall_active_span_sec", "twd_to_crypto_gap_sec", "twd_to_trade_gap_sec",
        "is_fast_twd_to_crypto", "is_fast_twd_to_trade",
        "has_twd", "has_crypto", "has_trade", "has_swap", "txn_type_diversity",
    ],
    # 十二、行為序列 [SEQ]
    "SEQ": [
        "total_actions", "repeat_action_ratio", "unique_action_types",
        "suspicious_seq_count", "suspicious_seq_ratio",
    ],
    # 十三、交叉特徵 [CROSS]
    "CROSS": [
        "trade_night_amount_cross", "crypto_night_value_cross",
        "kyc_trade_cross", "kyc_crypto_cross",
        "twd_crypto_value_gap", "twd_crypto_value_ratio",
        "behavior_diversity_score",
        "trc20_night_cross", "bsc_night_cross",
        "protocol_switch_fast_cross",
        "trc20_addr_ip_cross", "bsc_addr_ip_cross",
        "suspicious_fast_cross", "twd_fast_outflow_cross",
    ],
}

CATEGORY_LABEL = {
    "KYC"     : "一、用戶基本資訊特徵 [KYC]",
    "TWD"     : "二、台幣出入金特徵 [TWD]",
    "CRYPTO"  : "三、虛擬貨幣轉帳特徵 [CRYPTO]",
    "TRADE"   : "四、撮合交易特徵 [TRADE]",
    "SWAP"    : "五、一鍵買賣特徵 [SWAP]",
    "NETWORK" : "六、網路特徵 [NETWORK]",
    "WALLET"  : "七、錢包風險特徵 [WALLET]",
    "IP"      : "八、IP 特徵 [IP]",
    "TEMPORAL": "九、時間異常特徵 [TEMPORAL]",
    "AMOUNT"  : "十、金額異常特徵 [AMOUNT]",
    "FLOW"    : "十一、資金流動特徵 [FLOW]",
    "SEQ"     : "十二、行為序列特徵 [SEQ]",
    "CROSS"   : "十三、交叉特徵 [CROSS]",
}

# 建立 feature → category 的反查表
FEAT_TO_CATEGORY = {}
for cat, feats in FEATURE_CATEGORY.items():
    for f in feats:
        FEAT_TO_CATEGORY[f] = cat

# 特徵的中文說明（讓輸出更易讀）
FEATURE_DESC = {
    # KYC
    "lvl1_minus_confirm_sec"      : "Email驗證→手機驗證 間隔(秒)，越短越可疑",
    "lvl2_minus_confirm_sec"      : "Email驗證→身份驗證 間隔(秒)，越短越可疑",
    "lvl2_minus_lvl1_sec"         : "手機驗證→身份驗證 間隔(秒)，越短越可疑",
    "lvl1_minus_confirm_sec_log"  : "KYC手機驗證速度(log)，越小越快越可疑",
    "lvl2_minus_confirm_sec_log"  : "KYC身份驗證速度(log)，越小越快越可疑",
    "kyc_speed_composite"         : "KYC整體完成速度，越小越快越可疑",
    "lvl1_minus_confirm_sec_pct"  : "KYC速度百分位，越低代表完成越快",
    # 台幣
    "twd_txn_count"               : "台幣交易總次數",
    "twd_total_amount"            : "台幣交易總金額",
    "twd_out_in_amount_ratio"     : "台幣出金/入金比，接近1代表快進快出",
    "twd_night_ratio"             : "台幣深夜交易比例(0-5點)",
    "twd_withdraw_deposit_ratio"  : "台幣出金次數/入金次數",
    "twd_unique_ip"               : "台幣交易使用的不同IP數",
    "twd_burstiness"              : "台幣交易突發性指數，越高越異常",
    "twd_fast_outflow_cross"      : "台幣快速出金複合指標",
    "twd_round_1000_ratio"        : "台幣整千元交易比例，高代表被指定金額",
    "twd_round_10000_ratio"       : "台幣整萬元交易比例，高代表被指定金額",
    # 虛幣
    "crypto_txn_count"            : "虛幣交易總次數",
    "crypto_total_twd_value"      : "虛幣交易總台幣價值",
    "crypto_out_in_amount_ratio"  : "虛幣出金/入金比，接近1代表快進快出",
    "crypto_night_ratio"          : "虛幣深夜交易比例",
    "crypto_external_ratio"       : "虛幣鏈上(外部)交易比例",
    "crypto_out_total_twd_value"  : "虛幣出金總台幣價值",
    "crypto_burstiness"           : "虛幣交易突發性指數",
    "crypto_twd_value_cv"         : "虛幣金額變異係數，低代表每次金額固定",
    # TRC20/BSC
    "trc20_ratio"                 : "TRC20鏈交易比例(常見洗錢鏈)",
    "bsc_ratio"                   : "BSC鏈交易比例(常見洗錢鏈)",
    "trc20_tx_count"              : "TRC20鏈交易次數",
    "bsc_tx_count"                : "BSC鏈交易次數",
    "trc20_addr_reuse_rate"       : "TRC20地址重用率，高代表一直提到同一地址",
    "bsc_addr_reuse_rate"         : "BSC地址重用率",
    "trc20_night_ratio"           : "TRC20深夜交易比例",
    "trc20_addr_ip_cross"         : "TRC20地址重用×共用IP複合風險",
    "bsc_addr_ip_cross"           : "BSC地址重用×共用IP複合風險",
    "trc20_in_out_imbalance"      : "TRC20流向不平衡度，高代表單向資金流",
    "bsc_in_out_imbalance"        : "BSC流向不平衡度",
    # IP
    "shared_ip_ratio"             : "共用IP比例(同IP被3+用戶使用)",
    "shared_ip_count"             : "共用IP交易次數",
    "total_unique_ips"            : "使用的不同IP總數",
    "ip_jump_rate"                : "IP跳換率，越高代表每次都換IP",
    "ip_source_diversity"         : "跨渠道IP來源多樣性",
    # Swap
    "swap_max_crypto"             : "單筆最大虛幣Swap量",
    "swap_max_twd"                : "單筆最大台幣Swap量",
    "swap_crypto_twd_ratio"       : "Swap賣幣比例，接近1代表快速變現",
    "swap_concentration"          : "Swap金額集中度，高代表單筆佔比大",
    "swap_sell_speed"             : "每天賣幣次數",
    # 序列/資金流
    "suspicious_seq_count"        : "可疑操作序列數(台幣入金→1小時內虛幣提領)",
    "suspicious_seq_ratio"        : "可疑操作序列比例",
    "is_fast_twd_to_crypto"       : "台幣入金後24小時內開始虛幣提領",
    "suspicious_fast_cross"       : "可疑序列×快進快出複合訊號",
    "repeat_action_ratio"         : "連續重複動作比例(可能分批入金規避申報)",
    # 網路
    "network_in_degree"           : "被多少不同用戶內轉資金給他",
    "network_out_degree"          : "他內轉資金給多少不同用戶",
    "network_degree_imbalance"    : "網路結構不對稱度，高代表只進或只出",
    # 異常分數
    "iforest_score"               : "IsolationForest異常分數，越高越異常",
    "iforest_score_behav"         : "純行為IsolationForest異常分數",
    # 人口學
    "age"                         : "年齡",
    "sex"                         : "性別",
    "career"                      : "職業",
    "income_source"               : "收入來源",
    # 交叉特徵
    "twd_crypto_value_ratio"      : "虛幣價值/台幣入金比，接近1代表全部換成虛幣",
    "behavior_diversity_score"    : "行為多樣性分數",
    "protocol_switch_rate"        : "鏈別切換率，頻繁切換可能規避追蹤",
    "protocol_switch_fast_cross"  : "鏈別切換×高頻操作複合訊號",
    "trc20_night_cross"           : "TRC20×深夜複合風險",
    "bsc_night_cross"             : "BSC×深夜複合風險",
    "kyc_trade_cross"             : "KYC速度×交易量複合訊號",
    "kyc_crypto_cross"            : "KYC速度×虛幣交易量複合訊號",
    "high_risk_to_wallet_ratio"   : "高風險收款地址比例(同地址被5+用戶使用)",
    "high_risk_from_wallet_ratio" : "高風險付款地址比例",
    "addr_reuse_rate"             : "整體地址重用率",
    "twd_high_amount_ratio"       : "台幣大額交易比例(前10%)",
    "crypto_high_twd_ratio"       : "虛幣大額交易比例(前10%)",
}


# =========================================================
# 資料準備
# =========================================================
def prepare_xy(train_df, test_df):
    y      = train_df[TARGET_COL].copy()
    X      = train_df.drop(columns=[TARGET_COL, ID_COL] + DROP_TIME_COLS, errors="ignore")
    test_X = test_df.drop(columns=[ID_COL] + DROP_TIME_COLS, errors="ignore")

    non_num = X.select_dtypes(exclude=["int", "float", "bool"]).columns.tolist()
    X      = X.drop(columns=non_num, errors="ignore")
    test_X = test_X.drop(columns=non_num, errors="ignore")

    for col in X.select_dtypes(include="bool").columns:
        X[col] = X[col].astype(int)
    for col in test_X.select_dtypes(include="bool").columns:
        test_X[col] = test_X[col].astype(int)

    X      = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    test_X = test_X.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in set(X.columns) - set(test_X.columns):
        test_X[col] = 0
    test_X = test_X.drop(columns=set(test_X.columns) - set(X.columns), errors="ignore")
    test_X = test_X[X.columns]

    const  = X.columns[X.nunique() <= 1].tolist()
    X      = X.drop(columns=const, errors="ignore")
    test_X = test_X.drop(columns=const, errors="ignore")

    return X, y, test_X


# =========================================================
# 訓練模型
# =========================================================
def train_model(X, y):
    scale_pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, num_leaves=63,
        max_depth=6, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary", metric="average_precision",
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )

    va_prob = model.predict_proba(X_va)[:, 1]
    best_f1, best_th = 0.0, 0.5
    for th in np.arange(0.01, 0.95, 0.005):
        f1 = f1_score(y_va, (va_prob >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    print(f"[INFO] 模型訓練完成  Val F1={best_f1:.4f} @ th={best_th:.3f}")
    return model, best_th


# =========================================================
# SHAP 解釋
# =========================================================
def explain_with_shap(model, X_fraud, feature_names):
    """計算詐騙用戶的 SHAP 值"""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fraud)

    # LGB 二元分類回傳 list，取 class=1（詐騙）的 SHAP 值
    if isinstance(shap_values, list):
        shap_arr = shap_values[1]
    else:
        shap_arr = shap_values

    return pd.DataFrame(shap_arr, columns=feature_names, index=X_fraud.index)


def build_explanation_by_category(user_id, shap_row, feat_row):
    """
    對單一用戶，把正向 SHAP 特徵按分類彙整。
    每個分類只保留 shap >= 0.05 且最多 top 3 的特徵，代表真正的判定原因。
    """
    SHAP_THRESHOLD = 0.05  # 低於此值視為背景雜訊，不列入原因
    TOP_PER_CAT    = 3

    pos = shap_row[shap_row >= SHAP_THRESHOLD].sort_values(ascending=False)

    result = {}
    for feat, shap_val in pos.items():
        cat = FEAT_TO_CATEGORY.get(feat, "CROSS")
        if cat not in result:
            result[cat] = []
        if len(result[cat]) >= TOP_PER_CAT:
            continue

        raw_val = feat_row.get(feat, np.nan)
        if isinstance(raw_val, float):
            if abs(raw_val) >= 1000:
                val_str = f"{raw_val:,.0f}"
            elif abs(raw_val) >= 1:
                val_str = f"{raw_val:.2f}"
            else:
                val_str = f"{raw_val:.4f}"
        else:
            val_str = str(raw_val)

        # 值為 0 不是真正的原因，跳過
        try:
            if float(raw_val) == 0:
                continue
        except (TypeError, ValueError):
            pass

        result[cat].append({
            "feature"    : feat,
            "description": FEATURE_DESC.get(feat, feat),
            "value"      : val_str,
            "shap"       : round(float(shap_val), 4),
        })

    return result


# =========================================================
# 主流程
# =========================================================
def main():
    # --- 讀資料 ---
    train_df = pd.read_csv(TRAIN_PATH)
    test_df  = pd.read_csv(TEST_PATH)

    for df in [train_df, test_df]:
        cols    = df.columns.tolist()
        to_drop = [c for c in cols if (c.endswith("_x") or c.endswith("_y")) and c[:-2] in cols]
        df.drop(columns=to_drop, inplace=True, errors="ignore")

    X, y, test_X = prepare_xy(train_df, test_df)

    # --- 訓練模型 ---
    print("[INFO] 訓練解釋用模型...")
    model, best_th = train_model(X, y)

    # --- 取得詐騙名單 ---
    if os.path.exists(FRAUD_SOURCE):
        fraud_src = pd.read_csv(FRAUD_SOURCE)
        fraud_ids = fraud_src[fraud_src["status"] == 1][ID_COL].tolist()
        print(f"[INFO] 從 {FRAUD_SOURCE} 讀取詐騙名單：{len(fraud_ids)} 人")
    elif os.path.exists(FALLBACK_SUB):
        fraud_src = pd.read_csv(FALLBACK_SUB)
        fraud_ids = fraud_src[fraud_src["status"] == 1][ID_COL].tolist()
        print(f"[INFO] 從 {FALLBACK_SUB} 讀取詐騙名單：{len(fraud_ids)} 人")
    else:
        # 用模型直接預測
        test_prob = model.predict_proba(test_X)[:, 1]
        fraud_mask = test_prob >= best_th
        fraud_ids  = test_df.loc[fraud_mask, ID_COL].tolist()
        print(f"[INFO] 模型直接預測詐騙：{len(fraud_ids)} 人")

    # --- 對詐騙用戶計算 SHAP ---
    fraud_mask_test = test_df[ID_COL].isin(fraud_ids)
    X_fraud         = test_X[fraud_mask_test.values]
    fraud_user_ids  = test_df.loc[fraud_mask_test, ID_COL].values

    print(f"[INFO] 計算 {len(X_fraud)} 位詐騙用戶的 SHAP 值...")
    shap_df = explain_with_shap(model, X_fraud, X.columns.tolist())

    # 預測機率
    fraud_probs = model.predict_proba(X_fraud)[:, 1]

    # --- 建立說明報告（每個用戶一行，各分類合併）---
    output_rows = []
    for i, uid in enumerate(fraud_user_ids):
        shap_row   = shap_df.iloc[i]
        feat_row   = X_fraud.iloc[i].to_dict()
        cat_result = build_explanation_by_category(uid, shap_row, feat_row)

        row = {
            "user_id"   : uid,
            "fraud_prob": round(float(fraud_probs[i]), 4),
        }

        # 每個分類一個欄位，只列出 shap > 0 的特徵，格式：「說明(值)」
        for cat_key in FEATURE_CATEGORY.keys():
            cat_label = CATEGORY_LABEL[cat_key]
            if cat_key not in cat_result:
                row[cat_label] = ""
            else:
                parts = []
                for f in cat_result[cat_key]:
                    desc = f["description"] if f["description"] != f["feature"] else f["feature"]
                    parts.append(f"{desc}={f['value']}")
                row[cat_label] = " / ".join(parts)

        output_rows.append(row)

    explanation_df = pd.DataFrame(output_rows)

    # --- 輸出 ---
    os.makedirs("output_explanation", exist_ok=True)

    explanation_df.to_csv("output_explanation/fraud_explanation.csv",
                          index=False, encoding="utf-8-sig")

    # SHAP 完整明細（保留原始格式供深入分析）
    shap_full = shap_df.copy()
    shap_full.insert(0, ID_COL, fraud_user_ids)
    shap_full.insert(1, "fraud_prob", fraud_probs)
    shap_full.to_csv("output_explanation/fraud_shap_detail.csv",
                     index=False, encoding="utf-8-sig")

    # 印出前 10 人
    print("\n" + "="*70)
    print(f"詐騙用戶風險說明（共 {len(fraud_user_ids)} 人，顯示前 10 人）")
    print("="*70)

    for row in output_rows[:10]:
        print(f"\n用戶 {row['user_id']}  詐騙機率：{row['fraud_prob']:.1%}")
        for cat_key in FEATURE_CATEGORY.keys():
            cat_label = CATEGORY_LABEL[cat_key]
            content   = row.get(cat_label, "")
            if content:
                print(f"  {cat_label}")
                for item in content.split(" / "):
                    print(f"    • {item}")

    print(f"\n[INFO] 完整報告已儲存：output_explanation/fraud_explanation.csv")
    print(f"[INFO] SHAP 明細已儲存：output_explanation/fraud_shap_detail.csv")


if __name__ == "__main__":
    main()
