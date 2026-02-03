# Stock Signal Analyzer

A lightweight Streamlit app that analyzes stock price data using common indicators (RSI, moving averages, MACD) and provides a basic buy/sell signal from a chart image upload.

## Features
- Upload CSV price data and calculate RSI, SMA, EMA, and MACD.
- Upload a chart image and infer trend direction with a simple pixel-based heuristic.
- Clear visual summaries and signal explanations.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

## CSV format
Include columns named `Date` (or `date`) and `Close` (or `close`).

## Notes
The chart image signal is heuristic-only. It is **not** financial advice.
