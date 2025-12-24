#!/usr/bin/env python3
"""
Benchmark script to compare opponent difficulty levels using Arena.
Runs round-robin matches between LinearB, Myrmidon using TRAINED models.
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
from LinearB import LinearB
from Myrmidon import Myrmidon
from PlayerRandom import PlayerRandom
import numpy as np

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
            "p1_total_points": 0,
            "p2_total_points": 0,
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
        "linearb": lambda num: LinearB(number=num, alpha=0.3, Lambda=0.7, verboseFlag=False),
        "myrmidon": lambda num: Myrmidon(number=num, numSims=10, verboseFlag=False),
        "random": lambda num: PlayerRandom(number=num, verboseFlag=False),
    }
    
    logger.info("=" * 60)
    logger.info("CRIBBAGE OPPONENT DIFFICULTY BENCHMARK")
    logger.info("=" * 60)
    logger.info(f"Each match: {num_games_per_match} games")
    logger.info("")
    
    # Store all results
    all_results = {}
    opponent_stats = {opp: {"wins": 0, "total_point_diff": 0, "games": 0} for opp in opponents.keys()}
    
    # Run round-robin tournament
    opponent_list = list(opponents.keys())
    for i, opp1 in enumerate(opponent_list):
        for opp2 in opponent_list[i+1:]:
            results = play_match(
                opponents[opp1], opp1,
                opponents[opp2], opp2,
                num_games_per_match
            )
            
            # Store results both ways (p1 vs p2 and p2 vs p1)
            all_results[f"{opp1}_vs_{opp2}"] = results
            
            if results["games"] > 0:
                # Update opponent stats
                opponent_stats[opp1]["wins"] += results["p1_wins"]
                opponent_stats[opp1]["total_point_diff"] += results["avg_point_diff"] * num_games_per_match
                opponent_stats[opp1]["games"] += num_games_per_match
                
                opponent_stats[opp2]["wins"] += results["p2_wins"]
                opponent_stats[opp2]["total_point_diff"] -= results["avg_point_diff"] * num_games_per_match
                opponent_stats[opp2]["games"] += num_games_per_match
    
    # Calculate and display difficulty ratings
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    
    difficulty_ratings = {}
    
    for opponent in opponents.keys():
        stats = opponent_stats[opponent]
        games = stats["games"]
        if games > 0:
            win_rate = stats["wins"] / games
            avg_point_diff = stats["total_point_diff"] / games
        else:
            win_rate = 0
            avg_point_diff = 0
        
        difficulty = calculate_difficulty(win_rate, avg_point_diff)
        difficulty_ratings[opponent] = difficulty
        
        logger.info(f"\n{opponent.upper()}")
        logger.info(f"  Wins: {stats['wins']}/{games} ({win_rate*100:.1f}%)")
        logger.info(f"  Avg Point Diff: {avg_point_diff:+.1f}")
        logger.info(f"  Difficulty Rating: {difficulty:.1f}/10")
    
    # Display suggested names
    logger.info("\n" + "=" * 60)
    logger.info("SUGGESTED OPPONENT NAMES (by difficulty)")
    logger.info("=" * 60)
    for opponent in sorted(opponents.keys(), key=lambda x: difficulty_ratings[x], reverse=True):
        difficulty = difficulty_ratings[opponent]
        base_name = opponent.capitalize()
        logger.info(f"{base_name} ({difficulty:.1f}/10)")
    
    logger.info("\n" + "=" * 60)
    # Ensure this section appears at the very bottom
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    
    return {
        "difficulty_ratings": difficulty_ratings,
        "opponent_stats": opponent_stats,
        "all_results": all_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Cribbage opponent difficulty levels")
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of games to play per matchup (default: 10)"
    )
    args = parser.parse_args()
    
    main(num_games_per_match=args.games)
