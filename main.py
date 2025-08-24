import pandas as pd
import numpy as np
import backtrader as bt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.stats import zscore

df = pd.read_csv("dmart_5y_daily.csv", skiprows=2)
df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df.set_index("Date", inplace=True)

start_date = df.index.min() + pd.DateOffset(years=1)
df = df[df.index >= start_date]

df["Lag_Close"] = df["Close"].shift(252)
df["Rolling_Corr"] = df["Close"].rolling(20).corr(df["Lag_Close"])
df["MACD"] = df["Close"].ewm(span=12).mean() - df["Close"].ewm(span=26).mean()
df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
df["EMA100"] = df["Close"].ewm(span=100).mean()
df["Volatility"] = df["Close"].rolling(10).std()
df["ZScore"] = zscore(df["Close"].bfill())

df["Spike"] = ((df["ZScore"].abs() > 2.5) |
               (df["Volatility"] > df["Volatility"].rolling(20).mean() * 2)).astype(int)

df["Trend"] = (df["MACD"] > df["MACD_Signal"]).astype(int)
df["Regime"] = np.where((df["Rolling_Corr"] > 0.6) & (df["Trend"] == 1) & (df["Spike"] == 0), 1, 0)

df.dropna(inplace=True)

features = ["Rolling_Corr", "MACD", "MACD_Signal", "EMA100", "Volatility", "ZScore"]
X = df[features]
y = df["Regime"]

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

df["ML_Regime"] = model.predict(X)

regime_series = df["ML_Regime"]
transition_dates = []

for i in range(1, len(regime_series)):
    prev = regime_series.iloc[i - 1]
    curr = regime_series.iloc[i]
    if prev != curr:
        transition_type = "Start Strategy B (Rallying)" if curr == 1 else "Start Strategy A (Sideways)"
        transition_dates.append((regime_series.index[i], transition_type))

print("\n=== Strategy Transitions ===")
for date, transition in transition_dates:
    print(f"{date.strftime('%Y-%m-%d')} → {transition}")

bt_df = df[["Open", "High", "Low", "Close", "Volume", "ML_Regime"]].copy()
bt_df = bt_df.reset_index()
bt_df.rename(columns={"Date": "datetime"}, inplace=True)
bt_df["datetime"] = pd.to_datetime(bt_df["datetime"], errors="coerce")
bt_df = bt_df.dropna(subset=["datetime"])
assert bt_df["datetime"].dtype == "datetime64[ns]"

class MLFeed(bt.feeds.PandasData):
    lines = ('ml_regime',)
    params = (
        ('datetime', 'datetime'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('ml_regime', 'ML_Regime'),
        ('openinterest', None),
    )

class StrategySwitcher(bt.Strategy):
    def __init__(self):
        self.ema = bt.ind.EMA(period=20)
        self.rsi = bt.ind.RSI(period=14)
        self.trades = []

    def next(self):
        regime = self.data.ml_regime[0]

        if regime == 0:
            if not self.position and self.rsi < 35:
                self.buy()
                self.entry_price = self.data.close[0]
            elif self.position and self.rsi > 65:
                pnl = self.data.close[0] - self.entry_price
                self.trades.append(pnl)
                self.sell()

        elif regime == 1:
            if not self.position and self.data.close[0] > self.ema[0]:
                self.buy()
                self.entry_price = self.data.close[0]
            elif self.position and self.data.close[0] < self.ema[0]:
                pnl = self.data.close[0] - self.entry_price
                self.trades.append(pnl)
                self.sell()

    def stop(self):
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t > 0)
        win_ratio = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        print(f"\n🎯 Total Trades Executed : {total_trades}")
        print(f"✅ Winning Trades        : {winning_trades}")
        print(f"🏆 Strategy Win Ratio    : {win_ratio:.2f}%")

data = MLFeed(dataname=bt_df)
cerebro = bt.Cerebro()
cerebro.addstrategy(StrategySwitcher)
cerebro.adddata(data)
cerebro.addsizer(bt.sizers.AllInSizer)
cerebro.broker.set_cash(10000)

starting_cash = cerebro.broker.get_cash()
cerebro.run()
final_value = cerebro.broker.getvalue()

profit = final_value - starting_cash
percent_gain = (profit / starting_cash) * 100

print("\n=== Backtest Performance Summary ===")
print(f"📈 Starting Capital     : ₹{starting_cash:,.2f}")
print(f"📉 Final Portfolio Value: ₹{final_value:,.2f}")
print(f"💰 Net Profit           : ₹{profit:,.2f}")
print(f"📊 Total Return         : {percent_gain:.2f}%")# === Step 11: Plot Results ===
cerebro.plot(style='candlestick')
