"""Run the Flask app in Google Colab and expose it with ngrok.

Usage in Colab:
  pip install -r requirements-colab.txt
  python run_colab.py
"""

from app import app
from pyngrok import ngrok


if __name__ == "__main__":
    tunnel = ngrok.connect(5000)
    print(f"Public URL: {tunnel.public_url}")
    app.run(host="0.0.0.0", port=5000, debug=False)
