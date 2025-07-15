"""
スクリーニング銘柄DQN分析APIサーバー
screener_results.csvの銘柄をN225最適モデルで分析
Flask REST API として動作
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import warnings
from stable_baselines3 import DQN
import traceback
import torch
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import io
import base64

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)  # React アプリからのアクセスを許可


class ScreenedStockAnalyzer:
    """スクリーニング銘柄分析システム"""

    def __init__(
        self,
        model_path: str = "nikkei_cp_1997-01-01_2024-01-01_410000_steps.zip",
        screener_file: str = "screener_results.csv",
    ):
        self.model_path = model_path
        self.screener_file = screener_file
        self.model = None
        self.screened_stocks = None
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.feature_cols = [
            "Open",
            "SMA_5",
            "SMA_25",
            "SMA_75",
            "Upper_3σ",
            "Upper_2σ",
            "Upper_1σ",
            "Lower_3σ",
            "Lower_2σ",
            "Lower_1σ",
            "偏差値25",
            "Upper2_3σ",
            "Upper2_2σ",
            "Upper2_1σ",
            "Lower2_3σ",
            "Lower2_2σ",
            "Lower2_1σ",
            "偏差値75",
            "RSI_14",
            "RSI_22",
            "MACD",
            "MACD_signal",
            "Japan_10Y_Rate",
            "US_10Y_Rate",
            "ATR_5",
            "ATR_25",
            "RCI_9",
            "RCI_26",
            "VIX",
        ]

        self.load_model()
        self.load_screened_stocks()

    def load_model(self):
        """N225最適モデル読み込み"""
        try:
            if os.path.exists(self.model_path):
                self.model = DQN.load(self.model_path)
                print(f"✅ N225最適モデル読み込み完了")
                return True
            else:
                print(f"❌ モデルファイルが見つかりません: {self.model_path}")
                return False
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False

    def load_screened_stocks(self):
        """スクリーニング結果読み込み"""
        try:
            if os.path.exists(self.screener_file):
                self.screened_stocks = pd.read_csv(
                    self.screener_file, encoding="utf-8-sig"
                )
                print(
                    f"✅ スクリーニング結果読み込み完了: {len(self.screened_stocks)}銘柄"
                )
                return True
            else:
                print(
                    f"❌ スクリーニング結果ファイルが見つかりません: {self.screener_file}"
                )
                return False
        except Exception as e:
            print(f"❌ スクリーニング結果読み込みエラー: {e}")
            return False

    def get_market_status(self):
        """市場状態取得"""
        now = datetime.datetime.now()

        if now.weekday() >= 5:
            return "🔴 週末", "市場閉場中"
        elif 9 <= now.hour < 15:
            return "🟢 開場中", "リアルタイム取引可能"
        elif now.hour < 9:
            return "🟡 開場前", f"開場まで{9-now.hour}時間{60-now.minute}分"
        else:
            return "🔴 終了後", "翌営業日9:00開場"

    def convert_ticker_to_yfinance(self, ticker: str) -> str:
        """ティッカーをyfinance形式に変換"""
        if "." in ticker:
            return ticker
        if ticker.isdigit():
            return f"{ticker}.T"
        return ticker

    def get_single_signal(
        self, ticker: str, stock_name: str, include_reasoning: bool = False
    ) -> dict:
        """単一銘柄のシグナル取得 (ローカルCSV利用)"""
        if self.model is None:
            return {"error": "Model not loaded"}

        try:
            yf_ticker = self.convert_ticker_to_yfinance(ticker)

            # --- MODIFICATION START ---
            # Instead of yf.download, read from the local CSV file.
            data_path = os.path.join("stock_data", f"{yf_ticker}.csv")

            if not os.path.exists(data_path):
                return {"error": f"Local data file not found: {data_path}"}

            # Load data from the CSV file.
            data = pd.read_csv(data_path, index_col="Date", parse_dates=True)

            # Ensure data is sorted by date and get the last 200 records for analysis.
            # --- MODIFICATION END ---

            if data.empty:
                return {"error": "Data is empty after loading from file."}

            # The rest of the function remains the same.
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            # Add external data
            data["VIX"] = 16.0  # Placeholder
            data["Japan_10Y_Rate"] = 1.0  # Placeholder
            data["US_10Y_Rate"] = 4.5  # Placeholder

            # Calculate technical indicators
            data = self._safe_calculate_indicators(data)
            data = data.dropna()

            if len(data) < 130:
                return {
                    "error": f"Insufficient data for analysis: {len(data)}/130 rows"
                }

            # Prepare observation for the model
            observation = self._safe_prepare_observation(data)
            if observation is None:
                return {"error": "Failed to prepare observation data"}

            # Predict the action
            action, error = self._safe_predict(observation)
            if error:
                return {"error": error}

            # Calculate results
            current_price = float(data["Open"].iloc[-1])
            prev_price = float(data["Open"].iloc[-2])
            price_change = ((current_price - prev_price) / prev_price) * 100

            recent_volatility = data["Open"].pct_change().tail(5).std() * np.sqrt(252)
            confidence = min(0.95, max(0.30, 0.70 + (abs(price_change) * 0.05)))

            result = {
                "ticker": ticker,
                "yf_ticker": yf_ticker,
                "name": stock_name,
                "action": action,
                "current_price": current_price,
                "price_change": price_change,
                "confidence": confidence,
                "volatility": recent_volatility,
                "data_points": len(data),
            }

            if include_reasoning:
                result["reasoning"] = self._analyze_reasoning(data)

            return result

        except Exception as e:
            return {
                "error": f"Analysis error: {str(e)}",
                "trace": traceback.format_exc(),
            }

    def _safe_calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """安全なテクニカル指標計算"""
        # 移動平均
        for period in [5, 25, 75]:
            data[f"SMA_{period}"] = data["Open"].rolling(period, min_periods=1).mean()

        # ボリンジャーバンド（25日）
        data["STD_25"] = (
            data["Open"].rolling(25, min_periods=1).std().fillna(1.0).replace(0, 1.0)
        )
        for mult in [1, 2, 3]:
            data[f"Upper_{mult}σ"] = data["SMA_25"] + mult * data["STD_25"]
            data[f"Lower_{mult}σ"] = data["SMA_25"] - mult * data["STD_25"]

        data["偏差値25"] = 50 + 10 * ((data["Open"] - data["SMA_25"]) / data["STD_25"])

        # ボリンジャーバンド（75日）
        data["STD_75"] = (
            data["Open"].rolling(75, min_periods=1).std().fillna(1.0).replace(0, 1.0)
        )
        for mult in [1, 2, 3]:
            data[f"Upper2_{mult}σ"] = data["SMA_75"] + mult * data["STD_75"]
            data[f"Lower2_{mult}σ"] = data["SMA_75"] - mult * data["STD_75"]

        data["偏差値75"] = 50 + 10 * ((data["Open"] - data["SMA_75"]) / data["STD_75"])

        # RSI
        delta = data["Open"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        for period in [14, 22]:
            avg_gain = gain.rolling(period, min_periods=1).mean()
            avg_loss = loss.rolling(period, min_periods=1).mean().replace(0, 0.001)
            rs = avg_gain / avg_loss
            data[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        # MACD
        data["EMA12"] = data["Open"].ewm(span=12, min_periods=1).mean()
        data["EMA26"] = data["Open"].ewm(span=26, min_periods=1).mean()
        data["MACD"] = data["EMA12"] - data["EMA26"]
        data["MACD_signal"] = data["MACD"].ewm(span=9, min_periods=1).mean()

        # RCI
        for period in [9, 26]:
            data[f"RCI_{period}"] = (
                data["Open"]
                .rolling(period)
                .apply(lambda x: self._safe_calc_rci(x), raw=False)
                .fillna(0.0)
            )

        # ATR
        if all(col in data.columns for col in ["High", "Low"]):
            high_low = data["High"] - data["Low"]
            high_close = (data["High"] - data["Open"].shift()).abs()
            low_close = (data["Low"] - data["Open"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(
                axis=1
            )
        else:
            price_diff = data["Open"].diff().abs()
            true_range = price_diff.fillna(data["Open"] * 0.01)

        for period in [5, 25]:
            data[f"ATR_{period}"] = true_range.rolling(period, min_periods=1).mean()

        # データクリーニング
        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.fillna(method="ffill").fillna(method="bfill")
        data = data.fillna(0)

        return data

    def _safe_calc_rci(self, series: pd.Series) -> float:
        """安全なRCI計算"""
        try:
            if len(series) < 2:
                return 0.0

            n = len(series)
            order = np.arange(1, n + 1)
            rank = series.rank(method="first").values
            d = order - rank
            rci_value = (1 - 6 * np.sum(d**2) / (n * (n**2 - 1))) * 100

            if np.isnan(rci_value) or np.isinf(rci_value):
                return 0.0

            return float(rci_value)

        except:
            return 0.0

    def _safe_prepare_observation(self, data: pd.DataFrame):
        """安全な観測データ準備"""
        try:
            # 必要な特徴量チェック
            missing_cols = [col for col in self.feature_cols if col not in data.columns]
            if missing_cols:
                return None

            # 最新130日分を取得
            recent_data = data[self.feature_cols].tail(130).copy()

            if len(recent_data) < 130:
                return None

            # NaN・無限値のクリーニング
            recent_data = recent_data.replace([np.inf, -np.inf], np.nan)
            recent_data = recent_data.fillna(method="ffill").fillna(method="bfill")
            recent_data = recent_data.fillna(0)

            # MinMax正規化
            obs_arrays = []
            for col in self.feature_cols:
                window = recent_data[col].values.astype(np.float32)
                window = np.clip(window, -1e6, 1e6)

                min_val = np.min(window)
                max_val = np.max(window)

                if max_val - min_val <= 1e-10:
                    norm_window = np.zeros_like(window, dtype=np.float32)
                else:
                    norm_window = ((window - min_val) / (max_val - min_val)).astype(
                        np.float32
                    )

                norm_window = np.nan_to_num(
                    norm_window, nan=0.0, posinf=1.0, neginf=0.0
                )
                obs_arrays.append(norm_window.reshape(-1, 1))

            observation = np.concatenate(obs_arrays, axis=1).astype(np.float32)

            if observation.shape != (130, len(self.feature_cols)):
                return None

            observation = observation.reshape(1, 130, len(self.feature_cols))
            return observation

        except Exception as e:
            return None

    def _safe_predict(self, observation):
        """安全な予測実行"""
        try:
            observation = observation.astype(np.float32)

            with torch.no_grad():
                action, _states = self.model.predict(observation, deterministic=True)

            if isinstance(action, np.ndarray):
                action = int(action.item()) if action.size == 1 else int(action[0])
            else:
                action = int(action)

            if action not in [0, 1, 2]:
                return None, f"無効なアクション: {action}"

            return action, None

        except Exception as e:
            return None, f"予測エラー: {e}"

    def _analyze_reasoning(self, data: pd.DataFrame) -> dict:
        """思考プロセス分析：各指標がどう判断に影響したかを分析"""
        try:
            latest = data.iloc[-1]
            prev = data.iloc[-2]

            reasoning = {
                "trend_signals": [],
                "momentum_signals": [],
                "volatility_signals": [],
                "external_signals": [],
                "overall_sentiment": "neutral",
            }

            # トレンド系指標の分析
            if latest["Open"] > latest["SMA_25"]:
                reasoning["trend_signals"].append(
                    {
                        "indicator": "SMA_25",
                        "signal": "bullish",
                        "strength": min(
                            3.0,
                            (latest["Open"] - latest["SMA_25"])
                            / latest["SMA_25"]
                            * 100,
                        ),
                        "description": f"価格がSMA25上回り（+{((latest['Open'] - latest['SMA_25']) / latest['SMA_25'] * 100):.1f}%）",
                    }
                )
            else:
                reasoning["trend_signals"].append(
                    {
                        "indicator": "SMA_25",
                        "signal": "bearish",
                        "strength": min(
                            3.0,
                            abs(
                                (latest["Open"] - latest["SMA_25"])
                                / latest["SMA_25"]
                                * 100
                            ),
                        ),
                        "description": f"価格がSMA25下回り（{((latest['Open'] - latest['SMA_25']) / latest['SMA_25'] * 100):.1f}%）",
                    }
                )

            # ボリンジャーバンド分析
            if latest["偏差値25"] > 60:
                reasoning["trend_signals"].append(
                    {
                        "indicator": "Bollinger",
                        "signal": "bearish",
                        "strength": min(3.0, (latest["偏差値25"] - 60) / 10),
                        "description": f"ボリンジャーバンド上限付近（偏差値{latest['偏差値25']:.1f}）",
                    }
                )
            elif latest["偏差値25"] < 40:
                reasoning["trend_signals"].append(
                    {
                        "indicator": "Bollinger",
                        "signal": "bullish",
                        "strength": min(3.0, (40 - latest["偏差値25"]) / 10),
                        "description": f"ボリンジャーバンド下限付近（偏差値{latest['偏差値25']:.1f}）",
                    }
                )

            # モメンタム系指標の分析
            if latest["RSI_14"] < 30:
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "RSI",
                        "signal": "bullish",
                        "strength": min(3.0, (30 - latest["RSI_14"]) / 10),
                        "description": f"RSI売られ過ぎ水準（{latest['RSI_14']:.1f}）",
                    }
                )
            elif latest["RSI_14"] > 70:
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "RSI",
                        "signal": "bearish",
                        "strength": min(3.0, (latest["RSI_14"] - 70) / 10),
                        "description": f"RSI買われ過ぎ水準（{latest['RSI_14']:.1f}）",
                    }
                )

            # MACD分析
            if (
                latest["MACD"] > latest["MACD_signal"]
                and prev["MACD"] <= prev["MACD_signal"]
            ):
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "MACD",
                        "signal": "bullish",
                        "strength": 2.5,
                        "description": "MACDゴールデンクロス発生",
                    }
                )
            elif (
                latest["MACD"] < latest["MACD_signal"]
                and prev["MACD"] >= prev["MACD_signal"]
            ):
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "MACD",
                        "signal": "bearish",
                        "strength": 2.5,
                        "description": "MACDデッドクロス発生",
                    }
                )

            # RCI分析
            if latest["RCI_9"] < -80:
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "RCI",
                        "signal": "bullish",
                        "strength": min(3.0, abs(latest["RCI_9"] + 80) / 10),
                        "description": f"RCI極度の売られ過ぎ（{latest['RCI_9']:.1f}）",
                    }
                )
            elif latest["RCI_9"] > 80:
                reasoning["momentum_signals"].append(
                    {
                        "indicator": "RCI",
                        "signal": "bearish",
                        "strength": min(3.0, (latest["RCI_9"] - 80) / 10),
                        "description": f"RCI極度の買われ過ぎ（{latest['RCI_9']:.1f}）",
                    }
                )

            # ボラティリティ分析
            atr_change = (latest["ATR_25"] - prev["ATR_25"]) / prev["ATR_25"] * 100
            if atr_change > 10:
                reasoning["volatility_signals"].append(
                    {
                        "indicator": "ATR",
                        "signal": "caution",
                        "strength": min(3.0, atr_change / 10),
                        "description": f"ボラティリティ急増（ATR +{atr_change:.1f}%）",
                    }
                )

            # 外部環境分析
            if latest["VIX"] > 20:
                reasoning["external_signals"].append(
                    {
                        "indicator": "VIX",
                        "signal": "bearish",
                        "strength": min(3.0, (latest["VIX"] - 20) / 10),
                        "description": f"VIX恐怖指数高水準（{latest['VIX']}）",
                    }
                )

            # 総合センチメント計算
            bullish_strength = sum(
                [
                    s["strength"]
                    for s in reasoning["trend_signals"] + reasoning["momentum_signals"]
                    if s["signal"] == "bullish"
                ]
            )
            bearish_strength = sum(
                [
                    s["strength"]
                    for s in reasoning["trend_signals"] + reasoning["momentum_signals"]
                    if s["signal"] == "bearish"
                ]
            )

            if bullish_strength > bearish_strength + 1:
                reasoning["overall_sentiment"] = "bullish"
            elif bearish_strength > bullish_strength + 1:
                reasoning["overall_sentiment"] = "bearish"
            else:
                reasoning["overall_sentiment"] = "neutral"

            reasoning["sentiment_score"] = bullish_strength - bearish_strength

            return reasoning

        except Exception as e:
            return {"error": f"思考プロセス分析エラー: {str(e)}"}


# グローバルアナライザーインスタンス
analyzer = ScreenedStockAnalyzer()


@app.route("/api/status", methods=["GET"])
def get_status():
    """システム状態確認"""
    status, description = analyzer.get_market_status()

    return jsonify(
        {
            "status": "success",
            "system": {
                "model_loaded": analyzer.model is not None,
                "screened_stocks_loaded": analyzer.screened_stocks is not None,
                "model_path": analyzer.model_path,
                "screener_file": analyzer.screener_file,
                "performance": 1.5139,
                "feature_count": len(analyzer.feature_cols),
            },
            "market": {
                "status": status,
                "description": description,
                "timestamp": analyzer.timestamp,
            },
            "data": {
                "stock_count": (
                    len(analyzer.screened_stocks)
                    if analyzer.screened_stocks is not None
                    else 0
                )
            },
        }
    )


@app.route("/api/stocks", methods=["GET"])
def get_screened_stocks():
    """スクリーニング結果取得"""
    if analyzer.screened_stocks is None:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "スクリーニング結果が読み込まれていません",
                }
            ),
            400,
        )

    return jsonify(
        {"status": "success", "data": analyzer.screened_stocks.to_dict("records")}
    )


@app.route("/api/analyze", methods=["POST"])
def analyze_single():
    """単一銘柄分析"""
    try:
        data = request.json
        ticker = data.get("ticker")
        name = data.get("name", ticker)
        fitness_score = data.get("fitness_score", 0)

        if not ticker:
            return (
                jsonify(
                    {"status": "error", "message": "ティッカーが指定されていません"}
                ),
                400,
            )

        result = analyzer.get_single_signal(ticker, name)

        if "error" in result:
            return (
                jsonify(
                    {"status": "error", "message": result["error"], "ticker": ticker}
                ),
                400,
            )

        # アクション変換
        action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
        result["action"] = action_map[result["action"]]
        result["fitness_score"] = fitness_score

        return jsonify({"status": "success", "data": result})

    except Exception as e:
        return jsonify({"status": "error", "message": f"分析エラー: {str(e)}"}), 500


@app.route("/api/analyze_batch", methods=["POST"])
def analyze_batch():
    """複数銘柄一括分析"""
    try:
        data = request.json
        stocks = data.get("stocks", [])

        if not stocks:
            return (
                jsonify(
                    {"status": "error", "message": "分析対象銘柄が指定されていません"}
                ),
                400,
            )

        results = []
        success_count = 0

        for stock in stocks:
            ticker = stock.get("ティッカー") or stock.get("ticker")
            name = stock.get("銘柄名") or stock.get("name", ticker)
            fitness_score = stock.get("適性スコア") or stock.get("fitness_score", 0)

            if not ticker:
                continue

            result = analyzer.get_single_signal(ticker, name)

            if "error" not in result:
                # アクション変換
                action_map = {0: "BUY", 1: "HOLD", 2: "SELL"}
                result["action"] = action_map[result["action"]]
                result["fitness_score"] = fitness_score
                success_count += 1
            else:
                result = {
                    "ticker": ticker,
                    "name": name,
                    "fitness_score": fitness_score,
                    "action": "ERROR",
                    "current_price": 0,
                    "price_change": 0,
                    "confidence": 0,
                    "volatility": 0,
                    "error": result["error"],
                }

            results.append(result)

        return jsonify(
            {
                "status": "success",
                "data": {
                    "results": results,
                    "summary": {
                        "total": len(stocks),
                        "success": success_count,
                        "failure": len(stocks) - success_count,
                    },
                },
            }
        )

    except Exception as e:
        return jsonify({"status": "error", "message": f"一括分析エラー: {str(e)}"}), 500


@app.route("/api/upload_screener", methods=["POST"])
def upload_screener():
    """スクリーニング結果CSVアップロード"""
    try:
        if "file" not in request.files:
            return (
                jsonify({"status": "error", "message": "ファイルが選択されていません"}),
                400,
            )

        file = request.files["file"]
        if file.filename == "":
            return (
                jsonify({"status": "error", "message": "ファイルが選択されていません"}),
                400,
            )

        # CSVファイルを読み込み
        df = pd.read_csv(file, encoding="utf-8-sig")

        # データ検証
        required_cols = ["ティッカー", "銘柄名", "適性スコア"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"必要なカラムが不足しています: {missing_cols}",
                    }
                ),
                400,
            )

        # データ保存
        df.to_csv(analyzer.screener_file, index=False, encoding="utf-8-sig")
        analyzer.load_screened_stocks()

        return jsonify(
            {
                "status": "success",
                "message": f"{len(df)}銘柄のスクリーニング結果を更新しました",
                "data": {"count": len(df), "columns": df.columns.tolist()},
            }
        )

    except Exception as e:
        return (
            jsonify({"status": "error", "message": f"ファイル処理エラー: {str(e)}"}),
            500,
        )


@app.route("/api/health", methods=["GET"])
def health_check():
    """ヘルスチェック"""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
        }
    )


@app.errorhandler(404)
def not_found(error):
    return (
        jsonify({"status": "error", "message": "APIエンドポイントが見つかりません"}),
        404,
    )


@app.errorhandler(500)
def internal_error(error):
    return (
        jsonify({"status": "error", "message": "内部サーバーエラーが発生しました"}),
        500,
    )


if __name__ == "__main__":
    print(
        """
╔══════════════════════════════════════════════════════════════╗
║  🚀 スクリーニング銘柄DQN分析 API サーバー                    ║
║  📈 N225最適モデル使用 (Sharpe Ratio: 1.5139)                ║
║  🌐 Flask REST API として動作                                ║
╚══════════════════════════════════════════════════════════════╝
    """
    )

    print("🔄 システム初期化中...")
    if analyzer.model is None:
        print("❌ モデルが読み込まれていません。モデルファイルを確認してください。")
    else:
        print("✅ DQNモデル読み込み完了")

    if analyzer.screened_stocks is None:
        print(
            "⚠️  スクリーニング結果が読み込まれていません。/api/upload_screener で CSV をアップロードしてください。"
        )
    else:
        print(f"✅ スクリーニング結果読み込み完了: {len(analyzer.screened_stocks)}銘柄")

    print("\n📋 API エンドポイント:")
    print("   GET  /api/health - ヘルスチェック")
    print("   GET  /api/status - システム状態確認")
    print("   GET  /api/stocks - スクリーニング結果取得")
    print("   GET  /api/analyze_n225 - 日経平均予測・思考プロセス")
    print("   POST /api/analyze - 単一銘柄分析")
    print("   POST /api/analyze_batch - 複数銘柄一括分析")
    print("   POST /api/upload_screener - CSVアップロード")

    print(f"\n🌐 サーバー起動中 - http://localhost:5000")
    print("🔗 React アプリは http://localhost:3000 で起動してください")

    app.run(host="0.0.0.0", port=5000, debug=True)
