# filename: ytd_stock_gain_plot.py
import matplotlib.pyplot as plt
import yfinance as yf

# List of stock tickers
tickers = ["NVDA", "TLSA"]

# Fetch data for the current year
data = yf.download(tickers, period="ytd")

# Calculate the YTD gain for each stock
ytd_gain = data["Close"].pct_change().iloc[-1] * 100

# Plotting the YTD gain
plt.figure(figsize=(10, 6))
ytd_gain.plot(kind="bar")
plt.title("Stock Gain YTD")
plt.ylabel("Gain %")
plt.xlabel("Stocks")
plt.xticks(rotation=0)

# Save the plot to a file
plt.savefig("ytd_stock_gains.png", bbox_inches="tight")

# Also, printing the gain values to verify
print("YTD Gain for NVDA and TLSA:")
print(ytd_gain)
