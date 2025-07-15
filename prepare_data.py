#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from tqdm import tqdm
import math

# --- 設定 ---
TICKERS_JSON_FILE = "tickers.json"
RAW_DATA_DIR = "stock_data"
CHUNK_SIZE = 20  # 一度にまとめて取得するティッカー数
AUTO_ADJUST = False
START_DATE_FALLBACK = "2018-01-01"


def normalize_ticker(t: str) -> str:
    t = t.strip().upper()
    return t if t.endswith(".T") else t + ".T"


def load_existing_dataframe(ticker: str) -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "Date"
        return df
    except Exception:
        df = pd.read_csv(path, index_col="Date", parse_dates=True)
        df.index.name = "Date"
        return df


def save_dataframe(ticker: str, df: pd.DataFrame):
    path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df = df.sort_index()
    df.index.name = "Date"
    df.to_csv(path, index_label="Date")


def chunked_download(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    複数ティッカーをまとめて yf.download。
    group_by="ticker" で各ティッカー毎に取り出しやすくする。
    """
    return yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",  # 各ティッカー単位で DataFrame がネストされる
        threads=True,
    )


def main():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # ティッカーリスト読み込み & 正規化
    with open(TICKERS_JSON_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    tickers = [normalize_ticker(t) for t in raw]

    # 各ティッカーごとに「次回取得開始日」を計算
    next_starts = {}
    for t in tickers:
        df_exist = load_existing_dataframe(t)
        if df_exist.empty:
            next_starts[t] = START_DATE_FALLBACK
        else:
            last = df_exist.index.max()
            next_starts[t] = (last + timedelta(days=1)).strftime("%Y-%m-%d")

    today = date.today()
    tomorrow = today + timedelta(days=1)
    end_str = tomorrow.strftime("%Y-%m-%d")

    # チャンク処理
    total = len(tickers)
    chunks = math.ceil(total / CHUNK_SIZE)
    for i in range(chunks):
        chunk_list = tickers[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        # このチャンクで最も早い start を使ってまとめ取得
        chunk_start = min(next_starts[t] for t in chunk_list)
        print(
            f"\n[Chunk {i+1}/{chunks}] Fetch {len(chunk_list)} tickers from {chunk_start} to {end_str}"
        )
        df_chunk = chunked_download(chunk_list, chunk_start, end_str)
        # 取得結果を各ティッカー毎に処理
        for t in chunk_list:
            df_exist = load_existing_dataframe(t)
            df_new = pd.DataFrame()
            if t in df_chunk:
                df_new = df_chunk[t].dropna(how="all")
            # フォーマット統一
            if not df_new.empty:
                if isinstance(df_new.columns, pd.MultiIndex):
                    df_new.columns = df_new.columns.get_level_values(0)
                df_new.index = pd.to_datetime(df_new.index).tz_localize(None)
                df_new = df_new.loc[
                    :, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                ]
                # 指定 start 以降のみ抽出
                df_new = df_new[df_new.index >= pd.to_datetime(next_starts[t])]
            # 結合・保存
            if not df_new.empty:
                df_all = pd.concat([df_exist, df_new])
                df_all = df_all[~df_all.index.duplicated(keep="first")]
                save_dataframe(t, df_all)

    print("\n=== 完了 ===")


if __name__ == "__main__":
    main()
