#!/usr/bin/env python3
"""
Train AI models, save artifacts locally under ./models as _new, then copy them into
crib_back/models. Previous _new becomes _old if it exists.
"""

import shutil
from pathlib import Path
import json
import numpy as np
import joblib
import sys
import os
import io
import random
from contextlib import contextmanager, redirect_stdout

# Cribbage imports
from Arena import Arena

# Player imports
from Myrmidon import Myrmidon
from LinearB import LinearB
from NonLinearB import NonLinearB
from Perceptron import Perceptron
from DeepPeg import DeepPeg

# Training configuration
TRAINING_ROUNDS = 50000  # hands to play per iteration
TRAINING_ITERATIONS = 0  # 0 = infinite, >0 = specific number of iterations
BENCHMARK_GAMES = 3000  # games to compare candidate vs old before publishing
SEED = None  # set to an int for determinism, or leave None
UPDATE_BEST = "--do-not-update-best" not in sys.argv
GENERATE_REPORT = "--no-report" not in sys.argv

# CLI overrides: --seed=INT, --benchmark-games=INT, --iterations=INT
for arg in sys.argv:
    if arg.startswith("--seed="):
        try:
            SEED = int(arg.split("=", 1)[1])
        except Exception:
            pass
    if arg.startswith("--benchmark-games="):
        try:
            BENCHMARK_GAMES = int(arg.split("=", 1)[1])
        except Exception:
            pass
    if arg.startswith("--iterations="):
        try:
            TRAINING_ITERATIONS = int(arg.split("=", 1)[1])
        except Exception:
            pass

# Paths
BASE_DIR = Path(__file__).parent
local_models = BASE_DIR / "models"
crib_back_models = BASE_DIR.parent / "crib_back" / "models"

# Ensure directories exist
local_models.mkdir(parents=True, exist_ok=True)
crib_back_models.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to hide verbose game output."""
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
        # If suppression fails, continue normally
        yield

def get_best_opponent(player_number: int):
    """Determine the best available opponent for training.
    
    Checks for a best_opponent.txt file. If not found, defaults to Myrmidon.
    Returns an opponent player instance.
    """
    best_opponent_file = local_models / "best_opponent.txt"
    
    if best_opponent_file.exists():
        opponent_type = best_opponent_file.read_text().strip().lower()
        print(f"Training against current best: {opponent_type}")
        
        if opponent_type == "linearb":
            linear_b_dir = local_models / "linear_b"
            throw_path = linear_b_dir / "throw_weights_new.npy"
            peg_path = linear_b_dir / "peg_weights_new.npy"
            if throw_path.exists() and peg_path.exists():
                opponent = LinearB(number=player_number, alpha=0.3, Lambda=0.7, verboseFlag=False)
                opponent.throwingWeights = np.load(throw_path)
                opponent.peggingWeights = np.load(peg_path)
                return opponent
        elif opponent_type == "nonlinearb":
            linear_b_dir = local_models / "linear_b"
            throw_path = linear_b_dir / "nlb_throw_weights_new.npy"
            peg_path = linear_b_dir / "nlb_peg_weights_new.npy"
            if throw_path.exists() and peg_path.exists():
                opponent = NonLinearB(number=player_number, alpha=0.3, Lambda=0.7, verboseFlag=False)
                opponent.throwingWeights = np.load(throw_path)
                opponent.peggingWeights = np.load(peg_path)
                return opponent

    
    # Default to Myrmidon
    print(f"Training against Myrmidon (baseline)")
    return Myrmidon(number=player_number, numSims=10, verboseFlag=False)

def archive_and_save(new_path: Path, old_path: Path, data, save_fn):
    """Archive existing _new to _old, then save new data as _new."""
    # If _new exists, move it to _old
    if new_path.exists():
        if old_path.exists():
            old_path.unlink()
        shutil.move(str(new_path), str(old_path))
        print(f"  Archived previous model: {new_path.name} -> {old_path.name}")
    # Save new data (np.save expects (file, arr), joblib.dump expects (obj, file))
    save_fn(new_path, data) if save_fn == np.save else save_fn(data, new_path)
    print(f"  Saved new model: {new_path.name}")

def run_benchmark(candidate_player, old_loader_fn, games: int = BENCHMARK_GAMES) -> bool:
    """Compare candidate vs old model; return True if candidate wins more games.
    If old model cannot be loaded, accept candidate by default.
    """
    try:
        old_player = old_loader_fn(2)
    except FileNotFoundError:
        # No old baseline, accept candidate
        print("No _old baseline found; accepting candidate by default")
        return True

    # Play games: candidate (1) vs old (2)
    arena = Arena([candidate_player, old_player], repeatDeck=False, verboseFlag=False)
    with suppress_stdout():
        results = arena.playHands(games)
    total_diffs = results[2]
    p1_wins = sum(1 for d in total_diffs if d > 0)
    p2_wins = sum(1 for d in total_diffs if d < 0)
    ties = games - p1_wins - p2_wins
    p1_pct = (p1_wins / games) * 100.0
    p2_pct = (p2_wins / games) * 100.0
    print(f"Benchmark: candidate vs old — {p1_wins}W-{p2_wins}L-{ties}T | {p1_pct:.1f}% vs {p2_pct:.1f}%")
    return p1_wins > p2_wins

def prefer_new_else_old(new_path: Path, old_path: Path) -> Path:
    if new_path.exists():
        return new_path
    if old_path.exists():
        return old_path
    raise FileNotFoundError(f"No published artifacts found: {new_path.name}/{old_path.name}")

def generate_model_report(games: int = BENCHMARK_GAMES):
    """Round-robin report comparing published models. Writes JSON to models/model_report.json."""
    print("\nGenerating cross-model report...")
    candidates = []
    report = {"games": games, "pairs": []}

    # Build candidate loaders that prefer _new else _old
    def load_linearb(player_number: int):
        p = LinearB(number=player_number, alpha=0.3, Lambda=0.9, verboseFlag=False)
        p.throwingWeights = np.load(prefer_new_else_old(throw_weights_new, throw_weights_old))
        p.peggingWeights = np.load(prefer_new_else_old(peg_weights_new, peg_weights_old))
        return p

    def load_nlb(player_number: int):
        p = NonLinearB(number=player_number, alpha=0.3, Lambda=0.7, verboseFlag=False)
        p.throwingWeights = np.load(prefer_new_else_old(nlb_throw_weights_new, nlb_throw_weights_old))
        p.peggingWeights = np.load(prefer_new_else_old(nlb_peg_weights_new, nlb_peg_weights_old))
        return p

    def load_perceptron(player_number: int):
        p = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
        p.throwingWeights = np.load(prefer_new_else_old(perc_throw_weights_new, perc_throw_weights_old))
        p.peggingWeights = np.load(prefer_new_else_old(perc_peg_weights_new, perc_peg_weights_old))
        return p

    # DeepPeg loader disabled (model not in use)
    # def load_deeppeg(player_number: int):
    #     p = DeepPeg(number=player_number, softmaxFlag=False, saveBrains=False, verbose=False)
    #     p.peggingBrain = joblib.load(prefer_new_else_old(peg_brain_new, peg_brain_old))
    #     p.throwingBrain = joblib.load(prefer_new_else_old(throw_brain_new, throw_brain_old))
    #     return p

    loaders = []
    # Attempt to include each model if artifacts exist
    for name, loader in [("LinearB", load_linearb), ("NonLinearB", load_nlb), ("Perceptron", load_perceptron), ("Myrmidon", lambda n: Myrmidon(number=n, numSims=10, verboseFlag=False))]:
        try:
            # quick instantiation to verify availability
            _ = loader(1)
            candidates.append((name, loader))
        except FileNotFoundError:
            print(f"  Skipping {name}: no published artifacts found")
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    # Round-robin comparisons
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            name_a, loader_a = candidates[i]
            name_b, loader_b = candidates[j]
            p1 = loader_a(1)
            p2 = loader_b(2)
            arena = Arena([p1, p2], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(games)
            diffs = results[2]
            a_wins = sum(1 for d in diffs if d > 0)
            b_wins = sum(1 for d in diffs if d < 0)
            ties = games - a_wins - b_wins
            avg_diff = float(np.mean(diffs))
            print(f"  {name_a} vs {name_b}: {a_wins}W-{b_wins}L-{ties}T | avg diff {avg_diff:.2f}")
            report["pairs"].append({
                "a": name_a,
                "b": name_b,
                "a_wins": a_wins,
                "b_wins": b_wins,
                "ties": ties,
                "avg_diff": avg_diff,
            })

    # Aggregate totals per model
    totals = {}
    def ensure_model(name):
        if name not in totals:
            totals[name] = {"wins": 0, "losses": 0, "ties": 0, "point_diff": 0.0}
    for pair in report["pairs"]:
        a = pair["a"]; b = pair["b"]
        ensure_model(a); ensure_model(b)
        a_w = int(pair["a_wins"]); b_w = int(pair["b_wins"]); t = int(pair["ties"])
        totals[a]["wins"] += a_w
        totals[a]["losses"] += b_w
        totals[a]["ties"] += t
        totals[b]["wins"] += b_w
        totals[b]["losses"] += a_w
        totals[b]["ties"] += t
        # Use total point differential = avg_diff * games (a perspective)
        pd = float(pair["avg_diff"]) * games
        totals[a]["point_diff"] += pd
        totals[b]["point_diff"] -= pd

    # Compute win% and ranking
    ranked = []
    for name, stats in totals.items():
        played = stats["wins"] + stats["losses"]
        win_pct = (stats["wins"] / played * 100.0) if played > 0 else 0.0
        stats["win_pct"] = win_pct
        ranked.append((name, win_pct, stats["point_diff"], stats))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)

    print("\nRanking (by win% then point diff):")
    for idx, (name, win_pct, pd, stats) in enumerate(ranked, start=1):
        w = stats["wins"]; l = stats["losses"]; t = stats["ties"]
        print(f"  {idx}. {name}: {win_pct:.1f}% | {w}W-{l}L-{t}T | point diff {pd:+.1f}")

    report["totals"] = totals
    report["ranking"] = [{"name": n, "win_pct": wp, "point_diff": pd, "wins": s["wins"], "losses": s["losses"], "ties": s["ties"]} for (n, wp, pd, s) in ranked]

    # Underperforming models to consider deprioritizing
    deprioritize = []
    for (n, wp, pd, s) in ranked:
        if wp < 50.0 and pd < 0.0:
            deprioritize.append({
                "name": n,
                "win_pct": wp,
                "point_diff": pd,
                "wins": s["wins"],
                "losses": s["losses"],
                "ties": s["ties"],
            })
    if deprioritize:
        print("\nCandidates to deprioritize (win% < 50 and negative point diff):")
        for d in deprioritize:
            print(f"  - {d['name']}: {d['win_pct']:.1f}% | {d['wins']}W-{d['losses']}L-{d['ties']}T | point diff {d['point_diff']:+.1f}")
    report["deprioritize"] = deprioritize

    # Save JSON report
    out_path = local_models / "model_report.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Saved cross-model report to {out_path}")
    except Exception as e:
        print(f"Failed to write report: {e}")

def update_best_opponent_ladder(games: int = BENCHMARK_GAMES):
    """Run ladder benchmark and update best_opponent.txt if any candidate beats current best."""
    print("\nUpdating best opponent via ladder benchmark...")
    best_file = local_models / "best_opponent.txt"
    current_best = "Myrmidon"
    try:
        if best_file.exists():
            current_best = best_file.read_text(encoding="utf-8").strip() or current_best
    except Exception:
        pass

    def loader_for_name(name, player_number):
        if name == "Myrmidon":
            return Myrmidon(number=player_number, numSims=10, verboseFlag=False)
        if name == "LinearB":
            return LinearB(number=player_number, alpha=0.3, Lambda=0.9, verboseFlag=False)
        if name == "NonLinearB":
            return NonLinearB(number=player_number, alpha=0.3, Lambda=0.7, verboseFlag=False)
        if name == "Perceptron":
            return Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
        raise ValueError(name)

    # Build candidate list (available models)
    candidates = ["LinearB", "NonLinearB", "Perceptron", "Myrmidon"]
    available = []
    for name in candidates:
        try:
            _ = loader_for_name(name, 1)
            available.append(name)
        except Exception:
            pass

    # Challenge current best
    for cand in available:
        if cand == current_best:
            continue
        p1 = loader_for_name(cand, 1)
        p2 = loader_for_name(current_best, 2)
        arena = Arena([p1, p2], repeatDeck=False, verboseFlag=False)
        with suppress_stdout():
            results = arena.playHands(games)
        diffs = results[2]
        c_wins = sum(1 for d in diffs if d > 0)
        b_wins = sum(1 for d in diffs if d < 0)
        ties = games - c_wins - b_wins
        print(f"  Challenge: {cand} vs {current_best} — {c_wins}W-{b_wins}L-{ties}T")
        if c_wins > b_wins:
            print(f"  Promoting {cand} to best opponent")
            current_best = cand

    try:
        best_file.write_text(current_best, encoding="utf-8")
        print(f"Best opponent is now: {current_best} (written to {best_file})")
    except Exception as e:
        print(f"Failed to update best opponent: {e}")

print("=" * 80)
print("TRAINING CRIBBAGE AI MODELS")
print("=" * 80)

# Optional determinism
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)

# ============================================================================
# TRAIN LINEAR B
# ============================================================================
print("\n[1/3] Training LinearB...")
print("-" * 80)

linear_b_dir = ensure_dir(local_models / "linear_b")
throw_weights_new = linear_b_dir / "throw_weights_new.npy"
throw_weights_old = linear_b_dir / "throw_weights_old.npy"
peg_weights_new = linear_b_dir / "peg_weights_new.npy"
peg_weights_old = linear_b_dir / "peg_weights_old.npy"

# Choose learning rate based on whether continuing training
alpha_linearb = 0.3
continuing_linearb = throw_weights_old.exists() and peg_weights_old.exists()
if continuing_linearb:
    alpha_linearb = 0.1
    print("Loaded existing _old weights; continuing training with decayed alpha=0.1")
else:
    print("No existing weights found; training from scratch with alpha=0.3")
linear_b = LinearB(number=1, alpha=alpha_linearb, Lambda=0.9, verboseFlag=False)

# Load existing _old weights if they exist (incremental training)
if continuing_linearb:
    linear_b.throwingWeights = np.load(throw_weights_old)
    linear_b.peggingWeights = np.load(peg_weights_old)

opponent = get_best_opponent(2)

def load_linearb_old(player_number: int):
    if not (throw_weights_old.exists() and peg_weights_old.exists()):
        raise FileNotFoundError("LinearB _old weights not found")
    p = LinearB(number=player_number, alpha=alpha_linearb, Lambda=0.9, verboseFlag=False)
    p.throwingWeights = np.load(throw_weights_old)
    p.peggingWeights = np.load(peg_weights_old)
    return p

# Training loop with incremental saves
iteration = 0
try:
    while TRAINING_ITERATIONS == 0 or iteration < TRAINING_ITERATIONS:
        iteration += 1
        print(f"\nLinearB Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
        arena = Arena([linear_b, opponent], repeatDeck=False, verboseFlag=False)
        with suppress_stdout():
            results = arena.playHands(TRAINING_ROUNDS)
        avg_diff = np.mean(results[2])
        print(f"  Iteration {iteration} complete. Avg diff: {avg_diff:.2f}")
        
        # Benchmark and save after each iteration
        if run_benchmark(linear_b, load_linearb_old, BENCHMARK_GAMES):
            archive_and_save(throw_weights_new, throw_weights_old, linear_b.throwingWeights, np.save)
            archive_and_save(peg_weights_new, peg_weights_old, linear_b.peggingWeights, np.save)
            print(f"  ✓ Published LinearB iteration {iteration}")
        else:
            print(f"  ✗ LinearB iteration {iteration} did not beat _old; continuing training")
except KeyboardInterrupt:
    print(f"\n⚠ LinearB training interrupted at iteration {iteration}. Saving current state...")
    if run_benchmark(linear_b, load_linearb_old, BENCHMARK_GAMES):
        archive_and_save(throw_weights_new, throw_weights_old, linear_b.throwingWeights, np.save)
        archive_and_save(peg_weights_new, peg_weights_old, linear_b.peggingWeights, np.save)
        print("  ✓ Published interrupted LinearB weights")
    else:
        print("  ✗ Interrupted weights did not beat _old; discarded")

# ============================================================================
# TRAIN NON-LINEAR B
# ============================================================================
print("\n[2/3] Training NonLinearB...")
print("-" * 80)

nlb_throw_weights_new = linear_b_dir / "nlb_throw_weights_new.npy"
nlb_throw_weights_old = linear_b_dir / "nlb_throw_weights_old.npy"
nlb_peg_weights_new = linear_b_dir / "nlb_peg_weights_new.npy"
nlb_peg_weights_old = linear_b_dir / "nlb_peg_weights_old.npy"

# Create NonLinearB player
alpha_nlb = 0.3
continuing_nlb = nlb_throw_weights_old.exists() and nlb_peg_weights_old.exists()
if continuing_nlb:
    alpha_nlb = 0.1
    print("Loaded existing _old weights; continuing training with decayed alpha=0.1")
else:
    print("No existing weights found; training from scratch with alpha=0.3")
non_linear_b = NonLinearB(number=1, alpha=alpha_nlb, Lambda=0.7, verboseFlag=False)

# Load existing _old weights if they exist (incremental training)
if continuing_nlb:
    non_linear_b.throwingWeights = np.load(nlb_throw_weights_old)
    non_linear_b.peggingWeights = np.load(nlb_peg_weights_old)

opponent = get_best_opponent(2)

def load_nlb_old(player_number: int):
    if not (nlb_throw_weights_old.exists() and nlb_peg_weights_old.exists()):
        raise FileNotFoundError("NonLinearB _old weights not found")
    p = NonLinearB(number=player_number, alpha=alpha_nlb, Lambda=0.7, verboseFlag=False)
    p.throwingWeights = np.load(nlb_throw_weights_old)
    p.peggingWeights = np.load(nlb_peg_weights_old)
    return p

# Training loop with incremental saves
iteration = 0
try:
    while TRAINING_ITERATIONS == 0 or iteration < TRAINING_ITERATIONS:
        iteration += 1
        print(f"\nNonLinearB Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
        arena = Arena([non_linear_b, opponent], repeatDeck=False, verboseFlag=False)
        with suppress_stdout():
            results = arena.playHands(TRAINING_ROUNDS)
        avg_diff = np.mean(results[2])
        print(f"  Iteration {iteration} complete. Avg diff: {avg_diff:.2f}")
        
        # Benchmark and save after each iteration
        if run_benchmark(non_linear_b, load_nlb_old, BENCHMARK_GAMES):
            archive_and_save(nlb_throw_weights_new, nlb_throw_weights_old, non_linear_b.throwingWeights, np.save)
            archive_and_save(nlb_peg_weights_new, nlb_peg_weights_old, non_linear_b.peggingWeights, np.save)
            print(f"  ✓ Published NonLinearB iteration {iteration}")
        else:
            print(f"  ✗ NonLinearB iteration {iteration} did not beat _old; continuing training")
except KeyboardInterrupt:
    print(f"\n⚠ NonLinearB training interrupted at iteration {iteration}. Saving current state...")
    if run_benchmark(non_linear_b, load_nlb_old, BENCHMARK_GAMES):
        archive_and_save(nlb_throw_weights_new, nlb_throw_weights_old, non_linear_b.throwingWeights, np.save)
        archive_and_save(nlb_peg_weights_new, nlb_peg_weights_old, non_linear_b.peggingWeights, np.save)
        print("  ✓ Published interrupted NonLinearB weights")
    else:
        print("  ✗ Interrupted weights did not beat _old; discarded")

# ============================================================================
# TRAIN PERCEPTRON
# ============================================================================
print("\n[3/3] Training Perceptron...")
print("-" * 80)

perc_dir = ensure_dir(local_models / "perceptron")
perc_throw_weights_new = perc_dir / "throw_weights_new.npy"
perc_throw_weights_old = perc_dir / "throw_weights_old.npy"
perc_peg_weights_new = perc_dir / "peg_weights_new.npy"
perc_peg_weights_old = perc_dir / "peg_weights_old.npy"

# Create Perceptron player
continuing_perc = perc_throw_weights_old.exists() and perc_peg_weights_old.exists()
if continuing_perc:
    print("Loaded existing _old weights; continuing training")
else:
    print("No existing weights found; training from scratch")
perceptron = Perceptron(number=1, alpha=0.1, verboseFlag=False)

# Load existing _old weights if they exist (incremental training)
if continuing_perc:
    perceptron.throwingWeights = np.load(perc_throw_weights_old)
    perceptron.peggingWeights = np.load(perc_peg_weights_old)

opponent = get_best_opponent(2)

def load_perc_old(player_number: int):
    if not (perc_throw_weights_old.exists() and perc_peg_weights_old.exists()):
        raise FileNotFoundError("Perceptron _old weights not found")
    p = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
    p.throwingWeights = np.load(perc_throw_weights_old)
    p.peggingWeights = np.load(perc_peg_weights_old)
    return p

# Training loop with incremental saves
iteration = 0
try:
    while TRAINING_ITERATIONS == 0 or iteration < TRAINING_ITERATIONS:
        iteration += 1
        print(f"\nPerceptron Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
        arena = Arena([perceptron, opponent], repeatDeck=False, verboseFlag=False)
        with suppress_stdout():
            results = arena.playHands(TRAINING_ROUNDS)
        avg_diff = np.mean(results[2])
        print(f"  Iteration {iteration} complete. Avg diff: {avg_diff:.2f}")
        
        # Benchmark and save after each iteration
        if run_benchmark(perceptron, load_perc_old, BENCHMARK_GAMES):
            archive_and_save(perc_throw_weights_new, perc_throw_weights_old, perceptron.throwingWeights, np.save)
            archive_and_save(perc_peg_weights_new, perc_peg_weights_old, perceptron.peggingWeights, np.save)
            print(f"  ✓ Published Perceptron iteration {iteration}")
        else:
            print(f"  ✗ Perceptron iteration {iteration} did not beat _old; continuing training")
except KeyboardInterrupt:
    print(f"\n⚠ Perceptron training interrupted at iteration {iteration}. Saving current state...")
    if run_benchmark(perceptron, load_perc_old, BENCHMARK_GAMES):
        archive_and_save(perc_throw_weights_new, perc_throw_weights_old, perceptron.throwingWeights, np.save)
        archive_and_save(perc_peg_weights_new, perc_peg_weights_old, perceptron.peggingWeights, np.save)
        print("  ✓ Published interrupted Perceptron weights")
    else:
        print("  ✗ Interrupted weights did not beat _old; discarded")

# ============================================================================
# TRAIN DEEP PEG (SKIPPED - underperforms)
# ============================================================================
# print("\n[2/2] Training DeepPeg...")
# print("-" * 80)
# DeepPeg training skipped (model underperforms)

deep_peg_dir = ensure_dir(local_models / "deep_peg")
peg_brain_new = deep_peg_dir / "pegging_brain_new.pkl"
peg_brain_old = deep_peg_dir / "pegging_brain_old.pkl"
throw_brain_new = deep_peg_dir / "throwing_brain_new.pkl"
throw_brain_old = deep_peg_dir / "throwing_brain_old.pkl"

# Optional post-training tasks
if UPDATE_BEST:
    update_best_opponent_ladder(BENCHMARK_GAMES)

if GENERATE_REPORT:
    generate_model_report(BENCHMARK_GAMES)

# ============================================================================
# CREATE MYRMIDON CONFIG
# ============================================================================
print("\n[3/3] Creating Myrmidon config...")
print("-" * 80)

myrmidon_dir = ensure_dir(local_models / "myrmidon")
config_path = myrmidon_dir / "config.txt"

with open(config_path, 'w') as f:
    f.write(f"# Myrmidon Configuration\n")
    f.write(f"# Myrmidon is heuristic-based and requires no trained parameters\n")
    f.write(f"# Default simulations: 5\n")

print(f"✓ Created config at: {config_path}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)

# Manifest of required artifacts for each model
manifest = {
    "linear_b": {
        "throw_weights": "throw_weights_new.npy",
        "peg_weights": "peg_weights_new.npy",
        "nlb_throw_weights": "nlb_throw_weights_new.npy",
        "nlb_peg_weights": "nlb_peg_weights_new.npy",
    },
    "deep_peg": {
        "pegging_brain": "pegging_brain_new.pkl",
        "throwing_brain": "throwing_brain_new.pkl",
    },
    "myrmidon": {
        "config": "config.txt",
    },
}

manifest_path = local_models / "manifest.json"
with open(manifest_path, "w", encoding="utf-8") as mf:
    json.dump(manifest, mf, indent=2)

# Verify required artifacts exist and are non-empty
required_paths = [
    throw_weights_new,
    peg_weights_new,
    nlb_throw_weights_new,
    nlb_peg_weights_new,
    peg_brain_new,
    throw_brain_new,
    config_path,
    manifest_path,
]

missing = [p for p in required_paths if not p.exists() or p.stat().st_size == 0]
if missing:
    raise FileNotFoundError(f"Missing model artifacts: {missing}")

# Copy the local models directory into crib_back/models
print("\nCopying models to crib_back/models...")
shutil.copytree(local_models, crib_back_models, dirs_exist_ok=True)
print(f"✓ Copied models to {crib_back_models}")

print(f"\nModels saved locally to: {local_models}")
print(f"Copied to crib_back at: {crib_back_models}")
print(f"\nModel files created:")
print(f"  LinearB:")
print(f"    - throw_weights_new.npy (and _old if previous existed)")
print(f"    - peg_weights_new.npy (and _old if previous existed)")
print(f"    - nlb_throw_weights_new.npy (and _old if previous existed)")
print(f"    - nlb_peg_weights_new.npy (and _old if previous existed)")
print(f"  DeepPeg:")
print(f"    - pegging_brain_new.pkl (and _old if previous existed)")
print(f"    - throwing_brain_new.pkl (and _old if previous existed)")
print(f"  Myrmidon:")
print(f"    - config.txt")
print(f"  Manifest:")
print(f"    - manifest.json")
print("\nNext steps:")
print("  1. Run benchmark to compare _new vs _old models")
print("  2. If _new performs worse, restore _old and continue training it")
print("  3. Update crib_back/cribbage/opponents.py to load _new models")
print("=" * 80)
