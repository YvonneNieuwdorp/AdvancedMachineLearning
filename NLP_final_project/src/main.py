"""
main.py
-------
Entry point. Runs both pipelines sequentially.

    python src/main.py

To run a single pipeline:
    python src/genre_pipeline.py
    python src/hit_prediction.py
"""

from genre_pipeline import run_genre_pipeline
from hit_prediction import run_hit_prediction_pipeline


def main() -> None:
    print("=" * 55)
    print("  TASK 1: Genre Classification")
    print("=" * 55)
    run_genre_pipeline()

    print("\n" + "=" * 55)
    print("  TASK 2: Hit Prediction")
    print("=" * 55)
    run_hit_prediction_pipeline()

    print("\nAll done. Results in outputs/")


if __name__ == "__main__":
    main()