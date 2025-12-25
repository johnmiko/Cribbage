#!/usr/bin/env python3
"""
Benchmark script to compare _new vs _old model versions.
For each model type (LinearB, NonLinearB, DeepPeg), pit the _new version
against the _old version and report win percentages.
"""

import sys
import argparse
import logging
import io
import os
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from Arena import Arena
from LinearB import LinearB
from NonLinearB import NonLinearB
from DeepPeg import DeepPeg
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
)


@contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout (keeps stderr for logger output)."""
    try:
        with open(os.devnull, 'wb') as devnull_bin:
            devnull = io.TextIOWrapper(devnull_bin, encoding='utf-8', errors='ignore')
            try:
                with redirect_stdout(devnull):
                    yield
            finally:
                try:
                    devnull.flush()
                except Exception:
                    pass
    except Exception:
        yield


def load_linear_b_model(version: str, number: int):
    """Load LinearB model from _new or _old files."""
    models_dir = Path(__file__).parent / "models" / "linear_b"
    
    throw_weights_path = models_dir / f"throw_weights_{version}.npy"
    peg_weights_path = models_dir / f"peg_weights_{version}.npy"
    
    if not throw_weights_path.exists() or not peg_weights_path.exists():
        raise FileNotFoundError(f"LinearB {version} model not found at {models_dir}")
    
    player = LinearB(number=number, alpha=0.3, Lambda=0.7, verboseFlag=False)
    player.throwingWeights = np.load(throw_weights_path)
    player.peggingWeights = np.load(peg_weights_path)
    
    return player


def load_non_linear_b_model(version: str, number: int):
    """Load NonLinearB model from _new or _old files."""
    models_dir = Path(__file__).parent / "models" / "linear_b"
    
    throw_weights_path = models_dir / f"nlb_throw_weights_{version}.npy"
    peg_weights_path = models_dir / f"nlb_peg_weights_{version}.npy"
    
    if not throw_weights_path.exists() or not peg_weights_path.exists():
        raise FileNotFoundError(f"NonLinearB {version} model not found at {models_dir}")
    
    player = NonLinearB(number=number, alpha=0.3, Lambda=0.7, verboseFlag=False)
    player.throwingWeights = np.load(throw_weights_path)
    player.peggingWeights = np.load(peg_weights_path)
    
    return player


def load_deep_peg_model(version: str, number: int):
    """Load DeepPeg model from _new or _old files."""
    models_dir = Path(__file__).parent / "models" / "deep_peg"
    
    peg_brain_path = models_dir / f"pegging_brain_{version}.pkl"
    throw_brain_path = models_dir / f"throwing_brain_{version}.pkl"
    
    if not peg_brain_path.exists() or not throw_brain_path.exists():
        raise FileNotFoundError(f"DeepPeg {version} model not found at {models_dir}")
    
    player = DeepPeg(number=number, softmaxFlag=False, saveBrains=False, verbose=False)
    player.peggingBrain = joblib.load(peg_brain_path)
    player.throwingBrain = joblib.load(throw_brain_path)
    
    return player


def play_match(player1, player1_name: str, player2, player2_name: str, num_games: int = 100) -> dict:
    """
    Play multiple games between two players using Arena.
    
    Args:
        player1: Player 1 instance
        player1_name: Name of player 1
        player2: Player 2 instance
        player2_name: Name of player 2
        num_games: Number of games to play
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\nPlaying {num_games} games: {player1_name} vs {player2_name}")
    logger.info("-" * 60)
    
    try:
        # Use Arena to play games
        arena = Arena([player1, player2], repeatDeck=False, verboseFlag=False)
        # Suppress verbose gameplay output from Arena/engine
        with _suppress_stdout():
            results = arena.playHands(num_games)
        
        # results[0] = pegging diffs, results[1] = hands diffs, results[2] = total points diffs
        total_diffs = results[2]
        
        # Count wins for player1 (positive diff = player1 wins)
        p1_wins = sum(1 for diff in total_diffs if diff > 0)
        p2_wins = sum(1 for diff in total_diffs if diff < 0)
        ties = num_games - p1_wins - p2_wins
        
        # Calculate average point diff (player1 perspective)
        avg_point_diff = sum(total_diffs) / num_games if num_games > 0 else 0
        
        # Calculate win percentages
        p1_win_pct = (p1_wins / num_games * 100) if num_games > 0 else 0
        p2_win_pct = (p2_wins / num_games * 100) if num_games > 0 else 0
        
        logger.info(f"Results: {player1_name} {p1_wins}W-{p2_wins}L-{ties}T")
        logger.info(f"  {player1_name}: {p1_win_pct:.1f}% win rate")
        logger.info(f"  {player2_name}: {p2_win_pct:.1f}% win rate")
        logger.info(f"  Avg point diff: {avg_point_diff:+.1f}")
        
        return {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "ties": ties,
            "p1_win_pct": p1_win_pct,
            "p2_win_pct": p2_win_pct,
            "avg_point_diff": avg_point_diff,
            "games": num_games,
        }
    except Exception as e:
        logger.error(f"Error during match: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            "p1_wins": 0,
            "p2_wins": 0,
            "ties": 0,
            "p1_win_pct": 0,
            "p2_win_pct": 0,
            "avg_point_diff": 0,
            "games": 0,
        }


def main(num_games_per_match: int = 100):
    """
    Compare _new vs _old versions of each model.
    
    Args:
        num_games_per_match: Number of games to play per matchup (default 100)
    """
    logger.info("=" * 60)
    logger.info("NEW vs OLD MODEL COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Each match: {num_games_per_match} games")
    logger.info("")
    
    all_results = {}
    
    # Test LinearB
    try:
        logger.info("\n" + "=" * 60)
        logger.info("LINEARB: _new vs _old")
        logger.info("=" * 60)
        
        linear_b_new = load_linear_b_model("new", 1)
        linear_b_old = load_linear_b_model("old", 2)
        
        results = play_match(linear_b_new, "LinearB_new", linear_b_old, "LinearB_old", num_games_per_match)
        all_results["linearb"] = results
        
        if results["p1_win_pct"] > results["p2_win_pct"]:
            logger.info(f"\n✓ NEW model is BETTER ({results['p1_win_pct']:.1f}% > {results['p2_win_pct']:.1f}%)")
        elif results["p1_win_pct"] < results["p2_win_pct"]:
            logger.info(f"\n✗ NEW model is WORSE ({results['p1_win_pct']:.1f}% < {results['p2_win_pct']:.1f}%)")
        else:
            logger.info(f"\n= NEW and OLD models are TIED ({results['p1_win_pct']:.1f}%)")
    except FileNotFoundError as e:
        logger.warning(f"\nSkipping LinearB: {e}")
    
    # Test NonLinearB
    try:
        logger.info("\n" + "=" * 60)
        logger.info("NONLINEARB: _new vs _old")
        logger.info("=" * 60)
        
        nlb_new = load_non_linear_b_model("new", 1)
        nlb_old = load_non_linear_b_model("old", 2)
        
        results = play_match(nlb_new, "NonLinearB_new", nlb_old, "NonLinearB_old", num_games_per_match)
        all_results["nonlinearb"] = results
        
        if results["p1_win_pct"] > results["p2_win_pct"]:
            logger.info(f"\n✓ NEW model is BETTER ({results['p1_win_pct']:.1f}% > {results['p2_win_pct']:.1f}%)")
        elif results["p1_win_pct"] < results["p2_win_pct"]:
            logger.info(f"\n✗ NEW model is WORSE ({results['p1_win_pct']:.1f}% < {results['p2_win_pct']:.1f}%)")
        else:
            logger.info(f"\n= NEW and OLD models are TIED ({results['p1_win_pct']:.1f}%)")
    except FileNotFoundError as e:
        logger.warning(f"\nSkipping NonLinearB: {e}")
    
    # Test DeepPeg
    try:
        logger.info("\n" + "=" * 60)
        logger.info("DEEPPEG: _new vs _old")
        logger.info("=" * 60)
        
        deep_peg_new = load_deep_peg_model("new", 1)
        deep_peg_old = load_deep_peg_model("old", 2)
        
        results = play_match(deep_peg_new, "DeepPeg_new", deep_peg_old, "DeepPeg_old", num_games_per_match)
        all_results["deeppeg"] = results
        
        if results["p1_win_pct"] > results["p2_win_pct"]:
            logger.info(f"\n✓ NEW model is BETTER ({results['p1_win_pct']:.1f}% > {results['p2_win_pct']:.1f}%)")
        elif results["p1_win_pct"] < results["p2_win_pct"]:
            logger.info(f"\n✗ NEW model is WORSE ({results['p1_win_pct']:.1f}% < {results['p2_win_pct']:.1f}%)")
        else:
            logger.info(f"\n= NEW and OLD models are TIED ({results['p1_win_pct']:.1f}%)")
    except FileNotFoundError as e:
        logger.warning(f"\nSkipping DeepPeg: {e}")
    
    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    for model_name, results in all_results.items():
        if results["games"] > 0:
            status = "BETTER" if results["p1_win_pct"] > results["p2_win_pct"] else "WORSE" if results["p1_win_pct"] < results["p2_win_pct"] else "TIED"
            logger.info(f"{model_name.upper()}: NEW is {status} ({results['p1_win_pct']:.1f}% vs {results['p2_win_pct']:.1f}%)")
    
    logger.info("\n" + "=" * 60)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare new vs old model versions")
    parser.add_argument(
        "--games",
        type=int,
        default=500,
        help="Number of games to play per matchup (default: 100)"
    )
    args = parser.parse_args()
    
    main(num_games_per_match=args.games)
