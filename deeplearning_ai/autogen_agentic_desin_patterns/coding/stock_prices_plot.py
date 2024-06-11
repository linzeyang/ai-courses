# filename: stock_prices_plot.py
# import pandas as pd
from functions import get_stock_prices, plot_stock_prices

# Set the stock symbols and the date range
stock_symbols = ["NVDA", "TLSA"]
start_date = "2024-01-01"
end_date = "2024-06-10"  # Assuming today's date is 2024-06-10

# Get the stock prices for the given symbols and date range
stock_prices = get_stock_prices(stock_symbols, start_date, end_date)

# Plot the stock prices and save the figure to a file
plot_stock_prices(stock_prices, "stock_prices_YTD_plot.png")
