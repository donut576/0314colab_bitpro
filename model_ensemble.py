import pandas as pd
import numpy as np

# ===============================
# 1. 讀取三個模型結果
# ===============================

xgb = pd.read_csv("output_xgb/test_scores.csv")
lgb = pd.read_csv("output_lgb/test_scores.csv")
rf  = pd.read_csv("output_rf/test_scores.csv")

# 確保順序一致
df = xgb[["user_id"]].copy()
df["xgb_prob"] = xgb["pred_prob"]
df["lgb_prob"] = lgb["pred_prob"]
df["rf_prob"]  = rf["pred_prob"]

# ===============================
# 2. 各模型 threshold（你可以改）
# ===============================

xgb_th = 0.5
lgb_th = 0.5
rf_th  = 0.5

df["xgb_pred"] = (df["xgb_prob"] >= xgb_th).astype(int)
df["lgb_pred"] = (df["lgb_prob"] >= lgb_th).astype(int)
df["rf_pred"]  = (df["rf_prob"]  >= rf_th ).astype(int)

# ===============================
# 3. baseline（用 LGB）
# ===============================

df["final_pred"] = df["lgb_pred"].copy()

# ===============================
# 4. 找 decision boundary（不確定區）
# ===============================

LOW_TH  = 0.3
HIGH_TH = 0.7

uncertain_mask = (df["lgb_prob"] >= LOW_TH) & (df["lgb_prob"] <= HIGH_TH)

# ===============================
# 5. 多數投票
# ===============================

df["votes"] = df["xgb_pred"] + df["lgb_pred"] + df["rf_pred"]

df.loc[uncertain_mask, "final_pred"] = (
    df.loc[uncertain_mask, "votes"] >= 2
).astype(int)

# ===============================
# 6. 輸出 submission
# ===============================

submission = df[["user_id", "final_pred"]].rename(columns={"final_pred": "status"})
submission.to_csv("submission_ensemble.csv", index=False)

print("submission_ensemble.csv 已產生")
