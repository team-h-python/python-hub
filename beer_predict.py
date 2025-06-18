import numpy as np, pandas as pd, joblib, os, math

MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "beer_model.pkl"))
BEERS = list(MODEL.keys())  # 6銘柄名を自動取得

def _feature_engineering(payload: dict) -> pd.DataFrame:
    """入力 JSON を 1行の特徴量 DataFrame に変換"""
    date = pd.to_datetime(payload["date"])
    feat = {
        "dow": date.dayofweek,
        "doy": date.dayofyear,
        "sin_doy": np.sin(2*np.pi*date.dayofyear/365),
        "cos_doy": np.cos(2*np.pi*date.dayofyear/365),
        **payload["weather"]
    }
    recent = pd.DataFrame(payload["recent_sales"]).sort_values("date")
    for b in BEERS:
        feat[f"{b}_lag1"] = recent.iloc[-1][b]
        feat[f"{b}_lag7"] = recent.iloc[-7][b]
        feat[f"{b}_ma7"]  = recent[b].tail(7).mean()
    return pd.DataFrame([feat])

def predict_one(payload: dict) -> dict:
    X = _feature_engineering(payload)
    preds_raw = {b: float(MODEL[b].predict(X)[0]) for b in BEERS}
    preds_ceil = {b: math.ceil(preds_raw[b]) for b in BEERS}
    
    result = {
        b: f"{preds_ceil[b]}本（予測値: {preds_raw[b]:.2f}）" for b in BEERS
    }
    total_raw = sum(preds_raw.values())
    total_ceil = sum(preds_ceil.values())
    result["総予測杯数"] = f"{total_ceil}本（予測値: {total_raw:.2f}）"
    return result

if __name__ == "__main__":
    import json
    payload = json.load(open("sample_request.json", encoding="utf-8"))
    print(predict_one(payload))