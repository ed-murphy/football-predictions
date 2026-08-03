"""NFL totals model — command line entry point.

    python main.py                  # retrain, backtest, and predict this week's slate
    python main.py --train-only     # fit and evaluate, write no predictions
    python main.py --predict-only   # score the slate with the saved model
    python main.py --backtest       # walk-forward evaluation only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime

from src.evaluate import evaluate_holdout, walk_forward_backtest
from src.pipeline import build_features
from src.predictions import (
    build_prediction_table, load_latest_predictions, save_predictions,
)
from src.totals import get_totals
from src.train import load_model, train_model
from src.upcoming import prepare_upcoming_games
from src.weather_forecast import get_forecasted_weather
from config import (
    BACKTEST_PATH, BACKTEST_START_SEASON, EVAL_MODEL_PATH, EVAL_TEST_SEASONS,
    EVAL_TRAIN_SEASONS, PROD_MODEL_PATH, PROD_SEASONS, TRAIN_START_SEASON,
)

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", verbose: bool = False) -> None:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now():%Y%m%d_%H%M%S}.log")
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")],
    )


def run_training(game_frame, backtest: bool = True) -> None:
    """Fit the holdout and production models, and report how they do."""
    eval_model = train_model(game_frame, EVAL_MODEL_PATH, EVAL_TRAIN_SEASONS)
    evaluate_holdout(eval_model, game_frame, EVAL_TEST_SEASONS)

    if backtest:
        results = walk_forward_backtest(
            game_frame,
            start_season=BACKTEST_START_SEASON,
            train_start_season=TRAIN_START_SEASON,
        )
        if not results.empty:
            os.makedirs(os.path.dirname(BACKTEST_PATH) or ".", exist_ok=True)
            results.to_csv(BACKTEST_PATH, index=False)
            logger.info("Backtest written to %s.", BACKTEST_PATH)

    train_model(game_frame, PROD_MODEL_PATH, PROD_SEASONS)
    logger.info("Training complete.")


def run_predictions(bundle, use_cached_totals: bool = False):
    """Score the upcoming slate with the saved production model."""
    model = load_model(PROD_MODEL_PATH)

    totals = get_totals(use_cache_only=use_cached_totals, team_games=bundle.team_games)
    forecast = get_forecasted_weather(totals)
    upcoming = prepare_upcoming_games(totals, bundle, forecast)

    if upcoming.empty:
        logger.info("No upcoming games to score.")
        return None

    new_rows = build_prediction_table(model, upcoming)
    return save_predictions(new_rows, load_latest_predictions(), bundle.games)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train-only", action="store_true",
                      help="fit and evaluate models without predicting")
    mode.add_argument("--predict-only", action="store_true",
                      help="predict using the saved model without retraining")
    mode.add_argument("--backtest", action="store_true",
                      help="run walk-forward evaluation only")
    parser.add_argument("--use-cached-totals", action="store_true",
                        help="skip the odds API and use the cached lines")
    parser.add_argument("--skip-backtest", action="store_true",
                        help="train without the (slow) walk-forward backtest")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(verbose=args.verbose)

    bundle = build_features()

    if args.backtest:
        walk_forward_backtest(
            bundle.game_frame,
            start_season=BACKTEST_START_SEASON,
            train_start_season=TRAIN_START_SEASON,
        )
        return 0

    if not args.predict_only:
        run_training(bundle.game_frame, backtest=not args.skip_backtest)

    if args.train_only:
        return 0

    run_predictions(bundle, use_cached_totals=args.use_cached_totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
