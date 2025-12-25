#!/usr/bin/env python3
"""
Benchmark script to compare opponent difficulty levels using Arena.
Instead of full round-robin, pit every model against the current best model
(starting with Myrmidon). If a challenger beats the best, it becomes the new
best for subsequent matches.
"""

import sys
import argparse
import logging
import io
import os
from contextlib import contextmanager, redirect_stdout

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from Arena import Arena
import numpy as np
from pathlib import Path
from Myrmidon import Myrmidon
from LinearB import LinearB
from NonLinearB import NonLinearB
from Perceptron import Perceptron

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
)


@contextmanager
def _suppress_stdout():
    """Temporarily suppress stdout (keeps stderr for logger output).
    Uses a UTF-8 text wrapper with errors ignored to avoid encoding issues.
    """
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
        # If suppression fails for any reason, do not block execution
        yield


def play_match(player1_factory, player1_name: str, player2_factory, player2_name: str, num_games: int = 10) -> dict:
    """
    Play multiple games between two players using Arena.
    
    Args:
        player1_factory: Function that creates player 1
        player1_name: Name of player 1
        player2_factory: Function that creates player 2
        player2_name: Name of player 2
        num_games: Number of games to play
        
    Returns:
        Dictionary with results
    """
    logger.info(f"\nPlaying {num_games} games: {player1_name} vs {player2_name}")
    logger.info("-" * 60)
    
    try:
        # Create players
        player1 = player1_factory(1)
        player2 = player2_factory(2)
        
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
        
        logger.info(f"Results: {player1_name} {p1_wins}W-{p2_wins}L-{ties}T (Avg diff: {avg_point_diff:+.1f})")
        
        return {
            "p1_wins": p1_wins,
            "p2_wins": p2_wins,
            "ties": ties,
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
            "avg_point_diff": 0.0,
            "games": 0,
        }


def calculate_difficulty(win_rate: float, avg_point_diff: float) -> float:
    """
    Calculate difficulty rating from 1-10 based on win rate and point differential.
    
    Args:
        win_rate: Win rate (0-1)
        avg_point_diff: Average points per game difference
        
    Returns:
        Difficulty rating 1-10
    """
    # Win rate contributes 60% to difficulty
    win_score = win_rate * 6  # Max 6 points
    
    # Point differential contributes 40% to difficulty
    # Average of 20+ points per game = 4 points
    point_score = min(max(avg_point_diff / 5, 0), 4)  # Max 4 points
    
    difficulty = win_score + point_score
    return max(1, min(10, difficulty))  # Clamp to 1-10


def main(num_games_per_match: int = 10):
    """
    Run benchmark tournament using Arena.
    
    Args:
        num_games_per_match: Number of games to play per matchup (default 10)
        
    Returns:
        Dictionary with difficulty ratings and statistics
    """
    # Define players with their factories
    opponents = {
        "myrmidon": lambda num: Myrmidon(number=num, numSims=10, verboseFlag=False),
        "linearb": lambda num: LinearB(number=num, alpha=0.3, Lambda=0.9, verboseFlag=False),
        "nonlinearb": lambda num: NonLinearB(number=num, alpha=0.3, Lambda=0.7, verboseFlag=False),
        "perceptron": lambda num: Perceptron(number=num, alpha=0.1, verboseFlag=False),
    }
    
    logger.info("=" * 60)
    logger.info("CRIBBAGE OPPONENT DIFFICULTY BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Each match: {num_games_per_match} games")
    logger.info("")
    
    # Filter to available opponents only
    available_opponents = {}
    for name, factory in opponents.items():
        try:
            _ = factory(1)  # Test instantiation
            available_opponents[name] = factory
            logger.info(f"  ✓ {name} available")
        except Exception as e:
            logger.info(f"  ✗ {name} skipped: {e}")
    
    if not available_opponents:
        logger.error("No opponents available for benchmark!")
        return {"best_model": None, "all_results": {}}
    
    # Track results
    all_results = {}
    best_current = "myrmidon" if "myrmidon" in available_opponents else list(available_opponents.keys())[0]
    best_factory = available_opponents[best_current]
    logger.info(f"\nStarting benchmark: baseline best = {best_current}")

    # Challengers are everyone except the initial best
    challengers = [name for name in available_opponents.keys() if name != best_current]

    for challenger in challengers:
        logger.info("\n" + "-" * 60)
        logger.info(f"Challenger: {challenger} vs current best: {best_current}")
        results = play_match(
            available_opponents[challenger], challenger,
            best_factory, best_current,
            num_games_per_match,
        )
        all_results[f"{challenger}_vs_{best_current}"] = results

        # Decide if challenger becomes new best
        challenger_better = results["p1_wins"] > results["p2_wins"]
        if challenger_better:
            logger.info(f"\n{challenger} BEATS {best_current}! Updating best_current to {challenger}.")
            best_current = challenger
            best_factory = available_opponents[challenger]
        else:
            logger.info(f"\n{best_current} remains best (wins: {results['p2_wins']} vs {results['p1_wins']}).")
    
    # Summarize results
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Best model after benchmark: {best_current}")

    # Simple difficulty proxy: best at top, others ordered by win margin vs best
    summary = []
    for match, res in all_results.items():
        challenger, _, baseline = match.partition("_vs_")
        summary.append((challenger, res["p1_wins"] - res["p2_wins"], res))

    # Sort challengers by margin descending
    summary.sort(key=lambda x: x[1], reverse=True)

    for challenger, margin, res in summary:
        logger.info(f"\n{challenger.upper()} vs {best_current if challenger != best_current else 'previous best'}")
        logger.info(f"  Wins/Losses/Ties: {res['p1_wins']}/{res['p2_wins']}/{res['ties']}")
        logger.info(f"  Avg Point Diff: {res['avg_point_diff']:+.1f}")
        logger.info(f"  Win Margin: {margin:+}")
    
    logger.info("\n" + "=" * 60)
    # Ensure this section appears at the very bottom
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    
    return {
        "best_model": best_current,
        "all_results": all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cribbage opponent difficulty levels")
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to play per matchup (default: 10)"
    )
    args = parser.parse_args()
    
    main(num_games_per_match=args.games)
