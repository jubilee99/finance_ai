import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

# --- 設定 ---
TICKER = "3936.T"
CSV_PATH = os.path.join("stock_data", f"{TICKER}.csv")
DAYS_TO_FETCH = 200  # simple.py と同じ期間
REQUIRED_ROWS = 130 # モデルが必要とする行数

# simple.py からDQNモデルが使用する特徴量リストをコピー
FEATURE_COLS = [
    "Open", "SMA_5", "SMA_25", "SMA_75", "Upper_3σ", "Upper_2σ", "Upper_1σ",
    "Lower_3σ", "Lower_2σ", "Lower_1σ", "偏差値25", "Upper2_3σ", "Upper2_2σ",
    "Upper2_1σ", "Lower2_3σ", "Lower2_2σ", "Lower2_1σ", "偏差値75",
    "RSI_14", "RSI_22", "MACD", "MACD_signal", "Japan_10Y_Rate", "US_10Y_Rate",
    "ATR_5", "ATR_25", "RCI_9", "RCI_26", "VIX"
]

# --- simple.py から必要な関数をコピー ---

def _safe_calc_rci(series: pd.Series) -> float:
    try:
        if len(series) < 2: return 0.0
        n = len(series)
        order = np.arange(1, n + 1)
        rank = series.rank(method="first").values
        d = order - rank
        rci_value = (1 - 6 * np.sum(d**2) / (n * (n**2 - 1))) * 100
        return float(rci_value) if not (np.isnan(rci_value) or np.isinf(rci_value)) else 0.0
    except:
        return 0.0

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    # 移動平均
    for period in [5, 25, 75]: data[f"SMA_{period}"] = data["Open"].rolling(period, min_periods=1).mean()
    # ボリンジャーバンド（25日）
    std_25 = data["Open"].rolling(25, min_periods=1).std().fillna(1.0).replace(0, 1.0)
    for mult in [1, 2, 3]: data[f"Upper_{mult}σ"] = data["SMA_25"] + mult * std_25
    for mult in [1, 2, 3]: data[f"Lower_{mult}σ"] = data["SMA_25"] - mult * std_25
    data["偏差値25"] = 50 + 10 * ((data["Open"] - data["SMA_25"]) / std_25)
    # ボリンジャーバンド（75日）
    std_75 = data["Open"].rolling(75, min_periods=1).std().fillna(1.0).replace(0, 1.0)
    for mult in [1, 2, 3]: data[f"Upper2_{mult}σ"] = data["SMA_75"] + mult * std_75
    for mult in [1, 2, 3]: data[f"Lower2_{mult}σ"] = data["SMA_75"] - mult * std_75
    data["偏差値75"] = 50 + 10 * ((data["Open"] - data["SMA_75"]) / std_75)
    # RSI
    delta = data["Open"].diff()
    gain = delta.where(delta > 0, 0.0); loss = -delta.where(delta < 0, 0.0)
    for period in [14, 22]:
        avg_gain = gain.rolling(period, min_periods=1).mean()
        avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, 0.001)
        data[f"RSI_{period}"] = 100 - (100 / (1 + (avg_gain / avg_loss)))
    # MACD
    data["EMA12"] = data["Open"].ewm(span=12, min_periods=1).mean()
    data["EMA26"] = data["Open"].ewm(span=26, min_periods=1).mean()
    data["MACD"] = data["EMA12"] - data["EMA26"]
    data["MACD_signal"] = data["MACD"].ewm(span=9, min_periods=1).mean()
    # RCI
    for period in [9, 26]: data[f"RCI_{period}"] = data["Open"].rolling(period).apply(_safe_calc_rci, raw=False).fillna(0.0)
    # ATR
    if all(col in data.columns for col in ['High', 'Low', 'Open']):
        high_low = data["High"] - data["Low"]
        high_close = (data["High"] - data["Open"].shift()).abs()
        low_close = (data["Low"] - data["Open"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    else: # High, Low がない場合のフォールバック
        true_range = data["Open"].diff().abs().fillna(data["Open"] * 0.01)

    for period in [5, 25]: data[f"ATR_{period}"] = true_range.rolling(period, min_periods=1).mean()
    # データクリーニング
    data = data.replace([np.inf, -np.inf], np.nan).fillna(method="ffill").fillna(method="bfill").fillna(0)
    return data

def prepare_observation(data: pd.DataFrame):
    if len(data) < REQUIRED_ROWS: return None
    recent_data = data[FEATURE_COLS].tail(REQUIRED_ROWS).copy()
    obs_arrays = []
    for col in FEATURE_COLS:
        window = recent_data[col].values.astype(np.float32)
        min_val, max_val = np.min(window), np.max(window)
        if max_val - min_val <= 1e-10:
            norm_window = np.zeros_like(window, dtype=np.float32)
        else:
            norm_window = ((window - min_val) / (max_val - min_val)).astype(np.float32)
        norm_window = np.nan_to_num(norm_window, nan=0.0, posinf=1.0, neginf=0.0)
        obs_arrays.append(norm_window.reshape(-1, 1))
    observation = np.concatenate(obs_arrays, axis=1).astype(np.float32)
    return observation.reshape(1, REQUIRED_ROWS, len(FEATURE_COLS))

def process_and_get_observation(df: pd.DataFrame, source_name: str):
    """データ処理のパイプライン全体を実行し、観測データを返す"""
    print(f"\n--- [{source_name}] データの処理開始 ---")
    if df.empty:
        print("データが空です。処理をスキップします。")
        return None, None

    # ===== ★★★ 修正箇所 START ★★★ =====
    # yfinanceがMultiIndexカラムを返す場合があるので、それを平坦化する
    if isinstance(df.columns, pd.MultiIndex):
        print("カラムがMultiIndex形式のため、平坦化します。")
        df.columns = df.columns.get_level_values(0)

    # 重複したカラムがあれば、最初のものだけを残す
    if df.columns.has_duplicates:
        print("カラムに重複があるため、重複を削除します。")
        df = df.loc[:, ~df.columns.duplicated()]
    # ===== ★★★ 修正箇所 END ★★★ =====

    # 外部データを追加
    df['VIX'] = 16.0
    df['Japan_10Y_Rate'] = 1.0
    df['US_10Y_Rate'] = 4.5

    # テクニカル指標を計算
    df_processed = calculate_indicators(df)

    # NaNデータを含む行を削除
    df_processed = df_processed.dropna()
    print(f"指標計算とNaN除去後のデータ行数: {len(df_processed)}")

    if len(df_processed) < REQUIRED_ROWS:
        print(f"❌ データ不足 ({len(df_processed)}/{REQUIRED_ROWS}行)。観測データを生成できません。")
        return None, None

    print("モデルに入力される直前のデータ（末尾3行）:")
    print(df_processed[FEATURE_COLS].tail(3).round(2))

    # モデル用の観測データを準備
    observation = prepare_observation(df_processed)

    if observation is not None:
        print(f"✅ 観測データ生成成功。Shape: {observation.shape}")
    else:
        print("❌ 観測データの生成に失敗しました。")

    return observation, df_processed

def main():
    """メインのデバッグ処理"""
    print(f"--- 【{TICKER}】データパイプライン比較デバッグ ---")

    # 1. ターミナル版のデータ取得 (yfinance)
    print("\n[1] yfinanceからデータを取得 (ターミナル版の動作)")
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=DAYS_TO_FETCH)
    df_yf = yf.download(TICKER, start=start_date, end=end_date, progress=False, auto_adjust=False)
    print(f"yfinanceから {len(df_yf)} 行のデータを取得しました。")

    # 2. API版のデータ取得 (CSV)
    print(f"\n[2] CSVファイルからデータを取得 (API版の動作)")
    if os.path.exists(CSV_PATH):
        df_csv = pd.read_csv(CSV_PATH, index_col='Date', parse_dates=True)
        df_csv = df_csv.tail(DAYS_TO_FETCH)
        print(f"CSVから {len(df_csv)} 行のデータを取得しました。")
    else:
        print(f"❌ CSVファイルが見つかりません: {CSV_PATH}")
        return

    # 3. 各データを処理して観測データを生成
    obs_yf, processed_yf = process_and_get_observation(df_yf.copy(), "ターミナル版 (yfinance)")
    obs_csv, processed_csv = process_and_get_observation(df_csv.copy(), "API版 (CSV)")

    # 4. 最終的な観測データを比較
    print("\n\n--- [最終比較結果] ---")
    if obs_yf is not None and obs_csv is not None:
        if np.array_equal(obs_yf, obs_csv):
            print("✅ 結論: 観測データは完全に一致しました。もし結果が異なるなら、モデルの乱数固定などに問題があるかもしれません。")
        else:
            print("❌ 結論: 観測データが一致しませんでした。これが売買判断が異なる原因です。")
            diff = np.sum(np.abs(obs_yf - obs_csv))
            print(f"   - 観測データの絶対差分の合計: {diff:.4f}")
            print("   - 原因: yfinanceの履歴データとCSVのデータに微細な違い（株式分割の反映タイミング、過去データの修正等）がある可能性があります。")

            flat_diff = np.abs(obs_yf - obs_csv).flatten()
            max_diff_index = np.argmax(flat_diff)
            row_index = max_diff_index // len(FEATURE_COLS)
            col_index = max_diff_index % len(FEATURE_COLS)
            col_name = FEATURE_COLS[col_index]

            print(f"\n   - 最も差分が大きい箇所:")
            print(f"     - 日時: 最新日から{REQUIRED_ROWS - 1 - row_index}日前")
            print(f"     - 指標: {col_name}")
            print(f"     - ターミナル版の値: {obs_yf.flatten()[max_diff_index]:.6f}")
            print(f"     - API版の値     : {obs_csv.flatten()[max_diff_index]:.6f}")
    else:
        print("❌ 片方または両方の観測データが生成できず、比較できませんでした。")

if __name__ == "__main__":
    main()
