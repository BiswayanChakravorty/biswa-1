import io
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image


@dataclass
class ImageSignal:
    action: str
    slope: float
    confidence: float
    details: str


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_macd(series: pd.Series) -> pd.DataFrame:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal
    return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})


def analyze_image(image: Image.Image) -> ImageSignal:
    grayscale = image.convert("L")
    resized = grayscale.resize((320, 240))
    arr = np.array(resized)

    threshold = np.percentile(arr, 25)
    mask = arr < threshold

    ys = []
    xs = []
    height, width = mask.shape

    for x in range(width):
        column = mask[:, x]
        if not column.any():
            continue
        y_positions = np.where(column)[0]
        ys.append(y_positions.mean())
        xs.append(x)

    if len(xs) < 10:
        return ImageSignal(
            action="HOLD",
            slope=0.0,
            confidence=0.0,
            details="Not enough chart pixels detected; try a clearer chart.",
        )

    x_arr = np.array(xs)
    y_arr = np.array(ys)
    x_mean = x_arr.mean()
    y_mean = y_arr.mean()
    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sum((x_arr - x_mean) ** 2)
    slope = numerator / denominator if denominator else 0.0

    slope_direction = -slope
    confidence = min(1.0, abs(slope_direction) / (height / width))
    action = "BUY" if slope_direction > 0.02 else "SELL" if slope_direction < -0.02 else "HOLD"

    details = (
        f"Detected trend slope: {slope_direction:.3f}. "
        "Positive slope suggests upward momentum."
    )

    return ImageSignal(action=action, slope=slope_direction, confidence=confidence, details=details)


def load_price_data(file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [c.strip() for c in df.columns]
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    close_col = next((c for c in df.columns if c.lower() == "close"), None)

    if not date_col or not close_col:
        raise ValueError("CSV must contain Date and Close columns.")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)
    df = df.rename(columns={date_col: "Date", close_col: "Close"})
    return df


def main() -> None:
    st.set_page_config(page_title="Stock Signal Analyzer", layout="wide")
    st.title("Stock Signal Analyzer")
    st.write(
        "Upload price data or a chart image. The app computes indicators like RSI and MACD "
        "and suggests a simple buy/sell/hold signal."
    )

    col_data, col_image = st.columns(2)

    with col_data:
        st.subheader("Price Data Analysis")
        data_file = st.file_uploader("Upload CSV", type=["csv"], key="price_csv")
        if data_file:
            try:
                df = load_price_data(data_file.getvalue())
            except ValueError as exc:
                st.error(str(exc))
            else:
                df["RSI"] = compute_rsi(df["Close"])
                df["SMA_20"] = df["Close"].rolling(window=20).mean()
                df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
                macd = compute_macd(df["Close"])
                df = pd.concat([df, macd], axis=1)

                st.line_chart(df.set_index("Date")[["Close", "SMA_20", "EMA_20"]])
                st.line_chart(df.set_index("Date")["RSI"])
                st.line_chart(df.set_index("Date")[["macd", "signal"]])

                latest = df.iloc[-1]
                signal = "HOLD"
                if latest["RSI"] < 30 and latest["macd"] > latest["signal"]:
                    signal = "BUY"
                elif latest["RSI"] > 70 and latest["macd"] < latest["signal"]:
                    signal = "SELL"

                st.metric("Latest Close", f"{latest['Close']:.2f}")
                st.metric("RSI", f"{latest['RSI']:.2f}")
                st.metric("Signal", signal)

    with col_image:
        st.subheader("Chart Image Signal")
        image_file = st.file_uploader("Upload chart image", type=["png", "jpg", "jpeg"], key="chart_image")
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded chart", use_column_width=True)
            result = analyze_image(image)

            st.metric("Image-based Signal", result.action)
            st.progress(result.confidence)
            st.caption(result.details)

    st.info(
        "This tool provides heuristic signals only. Always validate with additional research "
        "before making financial decisions."
    )


if __name__ == "__main__":
    main()
