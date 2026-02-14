# Market Insight Assistant

A lightweight Flask app that evaluates stock market conditions using RSI, moving averages, and a simple chart-image trend detector. Upload a CSV of prices and/or a chart image to receive a buy/hold/sell recommendation.

## Features
- RSI, SMA, and EMA calculations from CSV price data.
- Chart image trend scan to detect a coarse uptrend/downtrend.
- Simple recommendation engine that combines signals.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Visit `http://localhost:5000` to use the app.

## Run Immediately on Google Colab
If your local environment blocks `pip` via proxy/network policy, run in Colab:

1. Upload project files to Colab (`app.py`, `templates/index.html`, `static/style.css`, `requirements.txt`, `requirements-colab.txt`, `run_colab.py`).
2. Install dependencies:

```bash
pip install -r requirements-colab.txt
```

3. Start the public tunnel + app:

```bash
python run_colab.py
```

4. Open the `Public URL` printed in output.

## CSV Format
Provide a CSV with a single column of prices (one price per row). The parser will read the first numeric value on each line.

## Disclaimer
This project is for educational purposes only and does not constitute financial advice.
