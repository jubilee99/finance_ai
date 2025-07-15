import matplotlib.pyplot as plt
import pandas as pd
from main import NikkeiEnv
from data import generate_env_data
from collections import Counter
from stable_baselines3 import DQN
from calc_performance import calculate_performance_metrics, compute_sharpe_ratio


# 例: 手動でデータを追加する
manual_data = {
    "Date": ["2025-06-11"],
    "Open": [38420],
    "High": [38420],
    "Low": [38420],
    "Close": [38420],
    "Volume": [100000000],
    "VIX": [16.98],
    "Japan_10Y_Rate": [1.31],
    "US_10Y_Rate": [4.33],
}

# DataFrame に変換
manual_data = pd.DataFrame(manual_data)
manual_data.set_index("Date", inplace=True)

# ──────────────────────────────
# バックテスト（テストデータ上で方策を実行）
start = "2024-01-01"
end = "2025-07-07"
test_data = generate_env_data(start, end, ticker="^N225")  # 日経平均
# test_data = generate_env_data(start, end, ticker="^N225", manual_data=manual_data)
window_size = 130
# バックテスト用（評価用）環境：通常の環境オブジェクトを利用
test_env = NikkeiEnv(
    test_data,
    window_size=window_size,
    transaction_cost=0.001,
    risk_limit=0.5,
    trade_penalty=0.00,
)

# 最後のアクションを保存するリスト
final_actions = []
for i in range(320000, 900001, 10000):
    obs = test_env.reset()
    done = False
    action_history = []

    num_steps = i
    model = DQN.load(
        f"nikkei_cp_1997-01-01_2024-01-01_{num_steps}_steps.zip",
        env=test_env,
    )
    print("## Step", num_steps)
    while not done:
        # 決定論的に行動を選択
        action, _ = model.predict(obs, deterministic=True)
        action_history.append(action)
        obs, reward, done, info = test_env.step(action)
        if done:  # エピソード終了時のアクションを保存
            final_actions.append(int(action))

    # テスト期間中のエクイティカーブを取得
    equity_curve = test_env.get_equity_curve()
    sharpe = compute_sharpe_ratio(equity_curve, yearly_risk_free_rate=0.0065)

    # パフォーマンス指標の計算と表示

    metrics = calculate_performance_metrics(equity_curve, action_history)
    print("=== パフォーマンス指標 ===")
    print(f"年利: {metrics['annual_return']:.2f}%")
    print("年間シャープレシオ:", sharpe)
    print(f"最大ドローダウン: {metrics['max_drawdown']:.2f}%")
    print(f"最大ドローダウン期間: {metrics['max_drawdown_period']}")
    print(f"勝率: {metrics['win_rate']:.2f}%")
    print(f"平均勝ち%: {metrics['avg_win']:.4f}%")
    print(f"平均負け%: {metrics['avg_loss']:.4f}%")
    print(f"W/Lレシオ: {metrics['wl_ratio']:.2f}")
    print(f"期待値: {metrics['expectancy']:.4f}%")
    print(f"プロフィットファクター: {metrics['profit_factor']:.2f}")
    print(f"取引日数: {metrics['total_days']}")
    print(f"平均勝ち期間: {metrics['avg_win_holding_period']}")
    print(f"平均負け期間: {metrics['avg_loss_holding_period']}")
    buy_steps = [i for i, a in enumerate(action_history) if a == 0]
    buy_values = [equity_curve[i] for i in buy_steps]
    steps = range(len(equity_curve))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, equity_curve, label="Equity Curve", color="blue")
    ax.scatter(
        buy_steps, buy_values, color="green", marker="^", s=100, label="BUY Signal"
    )
    ax.set_xlabel("Step (Day)")
    ax.set_ylabel("Asset Balance")
    ax.set_title("Equity Curve with Buy Signals")
    ax.tick_params(axis="y", labelcolor="blue")

    # 株価データを追加（赤）
    stock_prices = test_data["Open"].values
    offset = len(stock_prices) - len(equity_curve)
    stock_prices = stock_prices[
        offset:
    ]  # offsetにはwindow_size, 75のdrop_na区間、その他欠損データが含まれる
    ax2 = ax.twinx()
    ax2.plot(
        steps,
        stock_prices,
        label="Stock Price (N225)",
        color="red",
        linestyle="dashed",
        alpha=0.7,
    )
    ax2.set_ylabel("Stock Price", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    fig.suptitle("Equity Curve & Stock Price with Buy Signals")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    plt.savefig(
        f"equity_curve{num_steps}_{start}_{end}.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

# 最後のアクションの出現回数をカウント
final_action_counts = Counter(final_actions)

# 0, 1, 2 の出現回数を取得（存在しない場合は0）
count_0 = final_action_counts.get(0, 0)
count_1 = final_action_counts.get(1, 0)
count_2 = final_action_counts.get(2, 0)

# 結果を表示
print(f"{end} 買い:{count_0}, 待ち:{count_1}, 売り:{count_2}")
