import pandas as pd
import numpy as np
from tqdm import tqdm
import json
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ▼▼▼ 条件設定はここで行う ▼▼▼
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Trueにすると、フィルタリングせずに全銘柄のチェック結果を出力します
# どの銘柄がどの条件で除外されているかを確認したい場合に便利です
DEBUG_MODE = False

# 銘柄リストが記載されたJSONファイル名
TICKERS_JSON_FILE = "tickers.json"

# 株価データ（CSVファイル）が保存されているフォルダ名
STOCK_DATA_DIRECTORY = "stock_data"

# --- フィルタリング条件 ---
# 1. データ開始日の条件: この日付以前からデータが存在する銘柄のみを対象とします
REQUIRED_START_DATE_BEFORE = "2022-01-01"

# 2. 最低売買代金の条件(単位: 円): 1日の平均売買代金がこの値以上であること
#    例: 10億円 → 1_000_000_000, 1億円 → 100_000_000
MIN_DAILY_TURNOVER = 100_000_000

# 3. 最低ボラティリティの条件(単位: %): ATRがこの値以上であること
MIN_ATR_PERCENTAGE = 1.0

# 4. 株価の条件(単位: 円): この範囲内の株価の銘柄を対象とします【追加】
MIN_STOCK_PRICE = 50
MAX_STOCK_PRICE = 40000


# --- 出力ファイル名 ---
# 通常モード時の出力ファイル名
OUTPUT_FILENAME = "screener_results.csv"
# デバッグモード時の出力ファイル名
DEBUG_FILENAME = "screener_debug_report.csv"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ▲▲▲ 条件設定はここまで ▲▲▲
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_tickers_from_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tickers_dict = json.load(f)
        print(f"'{file_path}' から {len(tickers_dict)} 件の銘柄を読み込みました。")
        return tickers_dict
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません。")
        return None


def calculate_metrics(df, atr_period=14):
    if "Adj Close" not in df.columns and "Adjusted Close" in df.columns:
        df = df.rename(columns={"Adjusted Close": "Adj Close"})
    df["prev_close"] = df["Close"].shift(1)
    df["tr1"] = df["High"] - df["Low"]
    df["tr2"] = abs(df["High"] - df["prev_close"])
    df["tr3"] = abs(df["Low"] - df["prev_close"])
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    df["atr"] = df["tr"].rolling(window=atr_period).mean()
    latest_data = df.tail(252)
    last_close = latest_data["Close"].iloc[-1]
    avg_volume = latest_data["Volume"].mean()
    avg_turnover = avg_volume * last_close
    atr_percentage = (
        (latest_data["atr"].iloc[-1] / last_close) * 100 if last_close > 0 else 0
    )
    return {
        "最終株価": last_close,
        "平均売買代金(円)": avg_turnover,
        "ATR(%)": atr_percentage,
        "データ開始日": df.index[0].strftime("%Y-%m-%d"),
    }


def screen_stocks(tickers_dict, config):
    results = []
    missing_files = []
    ticker_list = list(tickers_dict.keys())

    print(
        f"ローカルディレクトリ '{config['data_dir']}' 内のCSVファイルをスクリーニングします..."
    )
    for ticker in tqdm(ticker_list, desc="銘柄スクリーニング中"):
        try:
            file_path = os.path.join(config["data_dir"], f"{ticker}.csv")
            if not os.path.exists(file_path):
                missing_files.append(ticker)
                continue

            data = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            if data.empty or len(data) < 252:
                continue

            metrics = calculate_metrics(data)

            required_date = pd.to_datetime(config["required_start_before"]).date()
            actual_date = pd.to_datetime(metrics["データ開始日"]).date()

            # --- フィルタリング条件の判定 ---【更新】
            pass_date = actual_date <= required_date
            pass_turnover = metrics["平均売買代金(円)"] >= config["min_turnover"]
            pass_atr = metrics["ATR(%)"] >= config["min_atr"]
            pass_price = (
                config["min_price"] <= metrics["最終株価"] <= config["max_price"]
            )

            result_entry = {
                "ティッカー": ticker,
                "銘柄名": tickers_dict[ticker],
                **metrics,
            }

            if config["debug"]:
                result_entry["日付OK"] = pass_date
                result_entry["売買代金OK"] = pass_turnover
                result_entry["ATR_OK"] = pass_atr
                result_entry["株価OK"] = pass_price  # デバッグ用に項目追加
                results.append(result_entry)
            else:
                # 全ての条件を満たした場合のみリストに追加
                if pass_date and pass_turnover and pass_atr and pass_price:
                    results.append(result_entry)
        except Exception:
            continue

    if missing_files:
        print(f"\nCSVファイルが見つからなかった銘柄数: {len(missing_files)}件")
    return pd.DataFrame(results)


def main():
    # スクリプト上部の設定を変数としてまとめる【更新】
    config = {
        "debug": DEBUG_MODE,
        "data_dir": STOCK_DATA_DIRECTORY,
        "min_turnover": MIN_DAILY_TURNOVER,
        "min_atr": MIN_ATR_PERCENTAGE,
        "required_start_before": REQUIRED_START_DATE_BEFORE,
        "min_price": MIN_STOCK_PRICE,  # 追加
        "max_price": MAX_STOCK_PRICE,  # 追加
    }

    tickers_dict = get_tickers_from_json(TICKERS_JSON_FILE)
    if not tickers_dict:
        return

    screened_df = screen_stocks(tickers_dict, config)

    if screened_df.empty:
        print(
            "\n基準を満たす銘柄が見つかりませんでした。スクリプト上部の条件を緩和して再試行してください。"
        )
        return

    if config["debug"]:
        print("\n--- デバッグモードレポート ---")
        sorted_df = screened_df.sort_values(
            by="平均売買代金(円)", ascending=False
        ).head(30)
        # デバッグ表示項目を更新
        display_columns = [
            "ティッカー",
            "銘柄名",
            "最終株価",
            "平均売買代金(円)",
            "ATR(%)",
            "データ開始日",
            "日付OK",
            "売買代金OK",
            "ATR_OK",
            "株価OK",
        ]
        print(sorted_df[display_columns].to_string())
        screened_df.to_csv(DEBUG_FILENAME, index=False, encoding="utf-8-sig")
        print(f"\n全銘柄のチェック結果を {DEBUG_FILENAME} に保存しました。")
        return

    screened_df["ATRスコア"] = (screened_df["ATR(%)"] - screened_df["ATR(%)"].min()) / (
        screened_df["ATR(%)"].max() - screened_df["ATR(%)"].min()
    )
    screened_df["売買代金スコア"] = (
        screened_df["平均売買代金(円)"] - screened_df["平均売買代金(円)"].min()
    ) / (screened_df["平均売買代金(円)"].max() - screened_df["平均売買代金(円)"].min())
    screened_df["適性スコア"] = (
        screened_df["ATRスコア"] * 0.6 + screened_df["売買代金スコア"] * 0.4
    ) * 100
    final_df = screened_df.sort_values(by="適性スコア", ascending=False).reset_index(
        drop=True
    )
    display_columns = [
        "ティッカー",
        "銘柄名",
        "適性スコア",
        "ATR(%)",
        "平均売買代金(円)",
        "最終株価",
        "データ開始日",
    ]
    final_df = final_df[display_columns]
    print("\n--- 銘柄スクリーニング結果 ---")
    print(final_df.to_string())
    final_df.to_csv(OUTPUT_FILENAME, index=False, encoding="utf-8-sig")
    print(f"\n結果を {OUTPUT_FILENAME} に保存しました。")


if __name__ == "__main__":
    main()
