import logging
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    filename="crypto.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S"
)

# Fetch historical data from CoinGecko
def fetch_coingecko_historical_data(symbol, currency, days=1):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        params = {"vs_currency": currency, "days": str(days), "interval": "minutely"}
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        data = response.json()

        # Extract 'prices' from the API response
        if "prices" not in data:
            raise ValueError("Unexpected response format from CoinGecko API.")
        prices = data["prices"]

        # Convert data to pandas DataFrame
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except requests.exceptions.RequestException as e:
        print(f"HTTP Error while fetching data: {e}")
        traceback.print_exc()
        return None
    except ValueError as e:
        print(f"Data Error: {e}")
        traceback.print_exc()
        return None


def preprocess_data(historical_data):
    logging.info("Prepping data...")
    try:
        df = pd.DataFrame(historical_data)
        df['moving_avg'] = moving_average_strategy(df['prices'], window=10)
        df['clusters'] = cluster_prices(df['prices'], n_clusters=3)
        logging.info("Data preparation completed successfully.")
    except Exception as e:
        logging.error(f"Error prepping  data: {e}")
        raise
    return df

def train_models(df):
    logging.info("Training models...")
    try:
        lstm_model = predict_with_lstm(input_shape=(len(df['prices']), 1))
        logistic_model = predict_with_logistic_regression(df[['prices']], df['clusters'], df[['prices']])
        knn_model = knn_classifier(df[['prices']], df['clusters'], k=5)
        logging.info("Model training completed successfully.")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise
    return lstm_model, logistic_model, knn_model

def make_predictions(models, live_data):
    logging.info("Making predictions...")
    try:
        lstm_model, logistic_model, knn_model = models
        lstm_prediction = lstm_model.predict(live_data)
        logging.info(f"Prediction completed: {lstm_prediction}")
    except Exception as e:
        logging.error(f"Error during predictions: {e}")
        raise
    return lstm_prediction

def decision_making(prediction, live_data):
    logging.info("Making trading decision...")
    try:
        decision = make_decision(prediction, live_data)
        logging.info(f"Decision made: {decision}")
    except Exception as e:
        logging.error(f"Error during decision making: {e}")
        raise
    return decision

def execute_trades(decision):
    logging.info("Executing trade...")
    try:
        if decision == "buy":
            live_trading()
            logging.info("Executed BUY trade.")
        elif decision == "sell":
            live_trading()
            logging.info("Executed SELL trade.")
        else:
            logging.info("No trade executed. Decision was HOLD.")
    except Exception as e:
        logging.error(f"Error during trade execution: {e}")
        raise

def main():
    logging.info("Trade bot Start")
    try:
        logging.info("Fetching historical data from coingecko")
        historical_data, live_data = fetch_coingecko_historical_data()
        logging.info("Preprocess historical data ...")
        df = preprocess_data(historical_data)
        logging.info("Training ML Models")
        models = train_models(df)
        logging.info("Begin the next price prediction process.")
        prediction = make_predictions(models, live_data)
        logging.info("Decide strategy")
        decision = decision_making(prediction, live_data)
        logging.info("Use all of the above data for the automated trading process.")
        execute_trades(decision)
        logging.info("script completed.")
    except Exception as e:
        logging.error(f"Critical error in trading bot: {e}")

if __name__ == "__main__":
    main()
