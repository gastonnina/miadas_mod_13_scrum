"""
Incremental pipeline runner.

Checkpoint logic
----------------
pipeline_state.json stores `last_processed_order` (ISO date string, e.g. "2018-07-01").

On each run:
  - If no checkpoint exists → full rebuild.
  - If a checkpoint exists and no orders are newer than it → skip (nothing to process).
  - If new orders exist → full rebuild of the master table (all history, not just the delta)
    and update the checkpoint to the latest order date.

Full rebuild is correct here because the master table is aggregated at the customer level:
a "new" order for an existing customer changes their aggregate metrics, so we must
re-aggregate from scratch rather than merging a partial delta.

Docker / Cron
-------------
Run with:
    python -m scripts.pipeline               # incremental (default)
    python -m scripts.pipeline --full        # force full rebuild ignoring checkpoint

Environment variables (for Docker):
    PROJECT_ROOT, RAW_DIR, PROCESSED_DIR, PIPELINE_STATE_FILE
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from .config import PIPELINE_STATE_FILE, PROCESSED_DIR, RAW_DIR
from .data_loader import load_raw_datasets
from .feature_engineering import build_master_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def load_pipeline_state(state_file: Path = PIPELINE_STATE_FILE) -> dict:
    if state_file.exists():
        with open(state_file) as f:
            return json.load(f)
    return {"last_processed_order": None}


def save_pipeline_state(state: dict, state_file: Path = PIPELINE_STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)
    logger.info(f"Checkpoint saved → {state}")


def _has_new_orders(orders: pd.DataFrame, last_processed: str | None) -> bool:
    """Return True if there are orders newer than the checkpoint date."""
    if last_processed is None:
        return True
    ts = pd.to_datetime(orders["order_purchase_timestamp"], errors="coerce")
    return bool((ts > pd.Timestamp(last_processed)).any())


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    state_file: Path = PIPELINE_STATE_FILE,
    incremental: bool = True,
    premium_quantile: float = 0.80,
) -> pd.DataFrame | None:
    """
    Build and persist the customer master table.

    Parameters
    ----------
    raw_dir : path to raw parquet/csv files
    processed_dir : destination for master_table.parquet
    state_file : path to pipeline_state.json
    incremental : when True, skip if no new orders since last checkpoint
    premium_quantile : passed to build_master_table

    Returns
    -------
    master_table DataFrame, or None if pipeline was skipped.
    """
    state = load_pipeline_state(state_file)
    last_processed = state.get("last_processed_order")

    logger.info("Loading raw datasets…")
    datasets = load_raw_datasets(raw_dir)

    if incremental:
        if not _has_new_orders(datasets["orders"], last_processed):
            logger.info(f"No new orders since {last_processed}. Pipeline skipped.")
            return None
        logger.info(
            f"New orders detected (checkpoint: {last_processed}). Running full rebuild."
        )

    logger.info("Building master table…")
    master_table, threshold = build_master_table(datasets, premium_quantile=premium_quantile)
    logger.info(
        f"Master table: {len(master_table):,} customers | "
        f"premium threshold (p{int(premium_quantile * 100)}): {threshold:.2f}"
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_dir / "master_table.parquet"
    master_table.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    logger.info(f"Saved → {output_path}  ({output_path.stat().st_size / 1_048_576:.2f} MB)")

    latest_order = str(
        pd.to_datetime(datasets["orders"]["order_purchase_timestamp"], errors="coerce").max()
    )[:10]
    save_pipeline_state({"last_processed_order": latest_order}, state_file)

    return master_table


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Olist premium-customer pipeline.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force a full rebuild, ignoring the checkpoint.",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.80,
        help="Percentile threshold for the is_premium label (default: 0.80).",
    )
    args = parser.parse_args()

    run_pipeline(incremental=not args.full, premium_quantile=args.quantile)
