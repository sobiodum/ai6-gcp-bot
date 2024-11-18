#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import List, Optional
import sys
from datetime import datetime
import json

from data_management.dataset_manager import DatasetManager
from trading.training_coordinator import TrainingCoordinator
from trading.model_manager import ModelManager


def setup_logging(log_dir: Path) -> None:
    """Configure logging to both file and console."""
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Forex trading models')

    parser.add_argument(
        '--pairs',
        nargs='+',
        help='List of currency pairs to train (e.g., EUR_USD GBP_USD). If not specified, trains all pairs.'
    )

    parser.add_argument(
        '--timesteps',
        type=int,
        default=1_000_000,
        help='Total timesteps for training (default: 1,000,000)'
    )

    parser.add_argument(
        '--eval-freq',
        type=int,
        default=15_000,
        help='Frequency of evaluation during training (default: 15,000 steps)'
    )

    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Train multiple pairs in parallel'
    )

    parser.add_argument(
        '--continue-training',
        action='store_true',
        help='Continue training from existing models if available'
    )

    return parser.parse_args()


def train_models(
    pairs: Optional[List[str]] = None,
    total_timesteps: int = 1_000_000,
    eval_freq: int = 15_000,
    parallel: bool = False,
    continue_training: bool = False
) -> None:
    """
    Main training function.

    Args:
        pairs: List of currency pairs to train. If None, trains all supported pairs.
        total_timesteps: Total timesteps for training
        eval_freq: Frequency of evaluation during training
        parallel: Whether to train pairs in parallel
        continue_training: Whether to continue training from existing models
    """
    logger = logging.getLogger(__name__)

    # Initialize components
    dataset_manager = DatasetManager()
    training_coordinator = TrainingCoordinator()
    model_manager = ModelManager()

    # Get pairs to train
    if not pairs:
        pairs = dataset_manager.get_currency_pairs()
        logger.info(f"No pairs specified. Training all {len(pairs)} pairs")

    logger.info(f"Starting training for pairs: {pairs}")
    logger.info(
        f"Training config: {total_timesteps:,} steps, evaluate every {eval_freq:,} steps")

    # Load and prepare data for all pairs
    pair_data = {}
    for pair in pairs:
        try:
            logger.info(f"Loading data for {pair}")
            df = dataset_manager.load_and_update_dataset(
                currency_pair=pair,
                normalize=True  # Enable feature normalization
            )
            pair_data[pair] = df
        except Exception as e:
            logger.error(f"Error loading data for {pair}: {str(e)}")
            pairs.remove(pair)

    if not pairs:
        logger.error("No valid pairs to train. Exiting.")
        return

    # Training loop
    try:
        if parallel:
            logger.info("Starting parallel training")
            training_coordinator.train_pairs(
                pairs=pairs,
                parallel=True
            )
        else:
            logger.info("Starting sequential training")
            for pair in pairs:
                logger.info(f"\nTraining {pair}")
                df = pair_data[pair]

                try:
                    # Train model
                    model, metrics = model_manager.train_model(
                        df=df,
                        pair=pair,
                        total_timesteps=total_timesteps,
                        checkpoint_freq=eval_freq
                    )

                    # Log training results
                    logger.info(f"Training completed for {pair}")
                    logger.info(
                        f"Final metrics: {json.dumps(metrics.__dict__, indent=2)}")

                except Exception as e:
                    logger.error(f"Error training {pair}: {str(e)}")
                    continue

        logger.info("Training completed for all pairs")

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error during training: {str(e)}")
        raise


def main():
    """Main entry point."""
    # Setup logging
    log_dir = Path("logs")
    setup_logging(log_dir)

    # Parse command line arguments
    args = parse_args()

    # Start training
    train_models(
        pairs=args.pairs,
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        parallel=args.parallel,
        continue_training=args.continue_training
    )


if __name__ == "__main__":
    main()
