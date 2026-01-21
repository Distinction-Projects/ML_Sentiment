from src.ml_sentiment import train_and_cache_models, cache_metrics


def main():
    train_and_cache_models()
    cache_metrics()
    print("Model cache ready.")


if __name__ == "__main__":
    main()
