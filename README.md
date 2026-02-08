# Market Insight Assistant

A lightweight Flask app that evaluates stock market conditions using RSI, moving averages, and a simple chart-image trend detector. Upload a CSV of prices and/or a chart image to receive a buy/hold/sell recommendation.

## Features
- RSI, SMA, and EMA calculations from CSV price data.
- Chart image trend scan to detect a coarse uptrend/downtrend.
- Simple recommendation engine that combines signals.

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` to use the app.

## CSV Format
Provide a CSV with a single column of prices (one price per row). The parser will read the first numeric value on each line.

## Disclaimer
This project is for educational purposes only and does not constitute financial advice.
