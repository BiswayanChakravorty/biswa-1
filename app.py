from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from flask import Flask, render_template, request
from PIL import Image, ImageStat

app = Flask(__name__)


@dataclass
class IndicatorSummary:
    rsi: Optional[float]
    sma: Optional[float]
    ema: Optional[float]


@dataclass
class ImageTrend:
    direction: str
    confidence: float
    details: str


def parse_prices(csv_bytes: bytes) -> List[float]:
    text_stream = io.StringIO(csv_bytes.decode("utf-8", errors="ignore"))
    reader = csv.reader(text_stream)
    prices: List[float] = []
    for row in reader:
        if not row:
            continue
        for cell in row:
            cell = cell.strip()
            if not cell:
                continue
            try:
                prices.append(float(cell))
                break
            except ValueError:
                continue
    return prices


def compute_sma(prices: Iterable[float], period: int) -> Optional[float]:
    values = list(prices)
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def compute_ema(prices: Iterable[float], period: int) -> Optional[float]:
    values = list(prices)
    if len(values) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for price in values[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def compute_rsi(prices: Iterable[float], period: int) -> Optional[float]:
    values = list(prices)
    if len(values) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        if change >= 0:
            gains.append(change)
        else:
            losses.append(abs(change))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
    if math.isclose(avg_loss, 0.0):
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def analyze_image_trend(image_bytes: bytes) -> Optional[ImageTrend]:
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None

    grayscale = image.convert("L")
    width, height = grayscale.size
    if width < 10 or height < 10:
        return ImageTrend(direction="unknown", confidence=0.0, details="Image too small")

    column_means = []
    for x in range(width):
        column = grayscale.crop((x, 0, x + 1, height))
        stat = ImageStat.Stat(column)
        column_means.append(stat.mean[0])

    midpoint = width // 2
    first_half = sum(column_means[:midpoint]) / midpoint
    second_half = sum(column_means[midpoint:]) / (width - midpoint)

    delta = second_half - first_half
    confidence = min(abs(delta) / 30, 1.0)

    if delta > 2:
        direction = "uptrend"
        detail = "Chart brightness increases from left to right"
    elif delta < -2:
        direction = "downtrend"
        detail = "Chart brightness decreases from left to right"
    else:
        direction = "sideways"
        detail = "No strong brightness trend detected"

    return ImageTrend(direction=direction, confidence=confidence, details=detail)


def build_recommendation(
    indicators: IndicatorSummary,
    image_trend: Optional[ImageTrend],
) -> Tuple[str, List[str]]:
    reasons = []
    if indicators.rsi is not None:
        reasons.append(f"RSI: {indicators.rsi:.2f}")
    if indicators.sma is not None:
        reasons.append(f"SMA: {indicators.sma:.2f}")
    if indicators.ema is not None:
        reasons.append(f"EMA: {indicators.ema:.2f}")
    if image_trend:
        reasons.append(
            f"Image trend: {image_trend.direction} (confidence {image_trend.confidence:.0%})"
        )

    recommendation = "Hold"
    if indicators.rsi is not None:
        if indicators.rsi < 30:
            recommendation = "Buy"
        elif indicators.rsi > 70:
            recommendation = "Sell"

    if image_trend:
        if image_trend.direction == "uptrend" and recommendation == "Buy":
            recommendation = "Strong Buy"
        elif image_trend.direction == "downtrend" and recommendation == "Sell":
            recommendation = "Strong Sell"
        elif image_trend.direction == "downtrend" and recommendation == "Buy":
            recommendation = "Hold"
        elif image_trend.direction == "uptrend" and recommendation == "Sell":
            recommendation = "Hold"

    if not reasons:
        reasons.append("No indicators available. Upload a chart or CSV data for analysis.")

    return recommendation, reasons


@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = None
    reasons: List[str] = []
    indicators = IndicatorSummary(rsi=None, sma=None, ema=None)
    image_trend = None
    error = None

    if request.method == "POST":
        period = request.form.get("period", "14")
        try:
            period_value = max(int(period), 2)
        except ValueError:
            period_value = 14

        if "price_csv" in request.files:
            price_file = request.files["price_csv"]
            if price_file and price_file.filename:
                try:
                    prices = parse_prices(price_file.read())
                    indicators = IndicatorSummary(
                        rsi=compute_rsi(prices, period_value),
                        sma=compute_sma(prices, period_value),
                        ema=compute_ema(prices, period_value),
                    )
                except Exception:
                    error = "Unable to parse price CSV. Please upload a valid CSV."

        if "chart_image" in request.files:
            chart_file = request.files["chart_image"]
            if chart_file and chart_file.filename:
                image_trend = analyze_image_trend(chart_file.read())
                if image_trend is None:
                    error = "Unable to read chart image. Please upload a PNG or JPG file."

        recommendation, reasons = build_recommendation(indicators, image_trend)

    return render_template(
        "index.html",
        recommendation=recommendation,
        reasons=reasons,
        indicators=indicators,
        image_trend=image_trend,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
