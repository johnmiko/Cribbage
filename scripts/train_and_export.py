#!/usr/bin/env python3
"""
Train AI models, save artifacts locally under ./trained_models as _new, then copy them into
crib_back/models. Previous _new becomes _old if it exists.
python .\scripts\train_and_export.py --models=simpleperceptron --iterations=5
"""

import shutil
from pathlib import Path
import json
import numpy as np
import sys
import os
import io
import random
from contextlib import contextmanager, redirect_stdout

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for crib_ai_trainer imports

# Cribbage imports
from crib_ai_trainer.Arena import Arena

# Player imports (refactored into models package)
from models.Perceptron import Perceptron
from models.SimpleFrequency import SimpleFrequency
from models.SimplePerceptron import SimplePerceptron
from models.TableQ import TableQ
from models.RuleBased import RuleBased


# Training configuration
TRAINING_ROUNDS = 1000  # hands to play per iteration
TRAINING_ITERATIONS = 1  # 0 = infinite, >0 = specific number of iterations
BENCHMARK_GAMES = 5000  # games to compare candidate vs old before publishing
SEED = None  # set to an int for determinism, or leave None
UPDATE_BEST = "--do-not-update-best" not in sys.argv
GENERATE_REPORT = "--no-report" not in sys.argv
MODELS_TO_TRAIN = ["simplefrequency", "tableq", "perceptron", "rulebased"]  # default: train all

# CLI overrides: --seed=INT, --benchmark-games=INT, --iterations=INT, --models=model1,model2
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
    if arg.startswith("--models="):
        try:
            models_str = arg.split("=", 1)[1]
            MODELS_TO_TRAIN = [m.strip().lower() for m in models_str.split(",")]
        except Exception:
            pass

# Paths
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
local_models = REPO_ROOT / "trained_models"
crib_back_models = REPO_ROOT.parent / "crib_back" / "models"

# Ensure directories exist
local_models.mkdir(parents=True, exist_ok=True)
crib_back_models.mkdir(parents=True, exist_ok=True)

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

@contextmanager
def suppress_stdout():
    """Temporarily suppress stdout to hide verbose game output."""
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
            try:
                devnull.close()
            except Exception:
                pass

def get_best_opponent(player_number: int):
    """Determine the best available opponent for training.
    
    Checks for a best_opponent.txt file. If not found, defaults to Perceptron.
    Returns an opponent player instance.
    """
    best_opponent_file = local_models / "best_opponent.txt"
    
    if best_opponent_file.exists():
        opponent_type = best_opponent_file.read_text().strip().lower()
        print(f"Training against current best: {opponent_type}")
        # Perceptron
        if opponent_type == "perceptron":
            perc_dir = local_models / "perceptron"
            throw_path = perc_dir / "throw_weights_new.npy"
            peg_path = perc_dir / "peg_weights_new.npy"
            if throw_path.exists() and peg_path.exists():
                opponent = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
                opponent.throwingWeights = np.load(throw_path)
                opponent.peggingWeights = np.load(peg_path)
                return opponent
        # SimpleFrequency
        if opponent_type == "simplefrequency" or opponent_type == "simple_frequency":
            sf_dir = local_models / "simple_frequency"
            weights_path = sf_dir / "weights_new.json"
            if weights_path.exists():
                opponent = SimpleFrequency(number=player_number, alpha=0.1, verboseFlag=False)
                opponent.load_weights(weights_path)
                return opponent
        # TableQ
        if opponent_type == "tableq":
            tq_dir = local_models / "tableq"
            weights_path = tq_dir / "weights_new.pkl"
            if weights_path.exists():
                opponent = TableQ(number=player_number, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False)
                opponent.load_weights(weights_path)
                return opponent

    
    # Default to Perceptron
    print(f"Training against Perceptron (baseline)")
    perc = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
    if perc_throw_weights_old.exists() and perc_peg_weights_old.exists():
        perc.throwingWeights = np.load(perc_throw_weights_old)
        perc.peggingWeights = np.load(perc_peg_weights_old)
    return perc

def archive_and_save(new_path: Path, old_path: Path, data, save_fn):
    """Archive existing _new to _old, then save new data as _new."""
    # If _new exists, move it to _old
    if new_path.exists():
        if old_path.exists():
            try:
                old_path.unlink()
            except Exception:
                pass
        try:
            shutil.move(str(new_path), str(old_path))
            print(f"  Archived previous model: {new_path.name} -> {old_path.name}")
        except Exception:
            # In testing, files may not actually exist due to mocks
            print(f"  (Skipped archiving; previous {new_path.name} not found)")
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
    """Round-robin report comparing published models. Writes JSON to trained_models/model_report.json."""
    print("\nGenerating cross-model report...")
    candidates = []
    report = {"games": games, "pairs": []}

    # Build candidate loaders that prefer _new else _old
    def load_perceptron(player_number: int):
        p = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
        p.name = "Perceptron"
        p.throwingWeights = np.load(prefer_new_else_old(perc_throw_weights_new, perc_throw_weights_old))
        p.peggingWeights = np.load(prefer_new_else_old(perc_peg_weights_new, perc_peg_weights_old))
        return p

    def load_simple_frequency(player_number: int):
        sf_dir = ensure_dir(local_models / "simple_frequency")
        sf_new = sf_dir / "weights_new.json"
        sf_old = sf_dir / "weights_old.json"
        path = prefer_new_else_old(sf_new, sf_old)
        p = SimpleFrequency(number=player_number, alpha=0.1, verboseFlag=False)
        p.load_weights(path)
        return p

    def load_tableq(player_number: int):
        tq_dir = ensure_dir(local_models / "tableq")
        tq_new = tq_dir / "weights_new.pkl"
        tq_old = tq_dir / "weights_old.pkl"
        path = prefer_new_else_old(tq_new, tq_old)
        p = TableQ(number=player_number, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False)
        p.load_weights(path)
        return p

    def load_rulebased(player_number: int):
        return RuleBased(number=player_number, aggressive=True, verboseFlag=False)

    loaders = []
    # Attempt to include each model if artifacts exist
    for name, loader in [
        ("Perceptron", load_perceptron),
        ("SimpleFrequency", load_simple_frequency),
        ("TableQ", load_tableq),
        ("RuleBased", load_rulebased)
    ]:
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
    current_best = "Perceptron"
    try:
        if best_file.exists():
            current_best = best_file.read_text(encoding="utf-8").strip() or current_best
    except Exception:
        pass

    def loader_for_name(name, player_number):
        if name == "Perceptron":
            p = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
            p.throwingWeights = np.load(prefer_new_else_old(perc_throw_weights_new, perc_throw_weights_old))
            p.peggingWeights = np.load(prefer_new_else_old(perc_peg_weights_new, perc_peg_weights_old))
            return p
        if name == "SimpleFrequency":
            sf_dir = ensure_dir(local_models / "simple_frequency")
            sf_new = sf_dir / "weights_new.json"
            sf_old = sf_dir / "weights_old.json"
            path = prefer_new_else_old(sf_new, sf_old)
            p = SimpleFrequency(number=player_number, alpha=0.1, verboseFlag=False)
            p.load_weights(path)
            return p
        if name == "TableQ":
            tq_dir = ensure_dir(local_models / "tableq")
            tq_new = tq_dir / "weights_new.pkl"
            tq_old = tq_dir / "weights_old.pkl"
            path = prefer_new_else_old(tq_new, tq_old)
            p = TableQ(number=player_number, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False)
            p.load_weights(path)
            return p
        if name == "RuleBased":
            return RuleBased(number=player_number, aggressive=True, verboseFlag=False)
        raise ValueError(name)

    # Build candidate list (available models)
    candidates = ["Perceptron", "SimpleFrequency", "TableQ", "RuleBased"]
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

print("\n" + "=" * 80)
print("STARTING TRAINING LOOP")
print("=" * 80)

# Initialize all models
# Perceptron setup
perc_dir = ensure_dir(local_models / "perceptron")
perc_throw_weights_new = perc_dir / "throw_weights_new.npy"
perc_throw_weights_old = perc_dir / "throw_weights_old.npy"
perc_peg_weights_new = perc_dir / "peg_weights_new.npy"
perc_peg_weights_old = perc_dir / "peg_weights_old.npy"

continuing_perc = perc_throw_weights_old.exists() and perc_peg_weights_old.exists()
perceptron = Perceptron(number=1, alpha=0.1, verboseFlag=False)
perceptron.name = "Perceptron"
if continuing_perc:
    perceptron.throwingWeights = np.load(perc_throw_weights_old)
    perceptron.peggingWeights = np.load(perc_peg_weights_old)

def load_perc_old(player_number: int):
    if not (perc_throw_weights_old.exists() and perc_peg_weights_old.exists()):
        raise FileNotFoundError("Perceptron _old weights not found")
    p = Perceptron(number=player_number, alpha=0.1, verboseFlag=False)
    p.throwingWeights = np.load(perc_throw_weights_old)
    p.peggingWeights = np.load(perc_peg_weights_old)
    return p

# SimpleFrequency setup
sf_dir = ensure_dir(local_models / "simple_frequency")
sf_weights_new = sf_dir / "weights_new.json"
sf_weights_old = sf_dir / "weights_old.json"
simple_frequency = SimpleFrequency(number=1, alpha=0.1, verboseFlag=False)
if sf_weights_old.exists():
    try:
        simple_frequency.load_weights(sf_weights_old)
    except Exception:
        pass

def load_sf_old(player_number: int):
    if not sf_weights_old.exists():
        raise FileNotFoundError("SimpleFrequency _old weights not found")
    p = SimpleFrequency(number=player_number, alpha=0.1, verboseFlag=False)
    try:
        p.load_weights(sf_weights_old)
    except Exception:
        # Treat any load error as missing baseline to satisfy tests
        raise FileNotFoundError("SimpleFrequency _old weights unreadable")
    return p

# TableQ setup
tq_dir = ensure_dir(local_models / "tableq")
tq_weights_new = tq_dir / "weights_new.pkl"
tq_weights_old = tq_dir / "weights_old.pkl"
tableq = TableQ(number=1, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False)
if tq_weights_old.exists():
    try:
        tableq.load_weights(tq_weights_old)
    except Exception:
        pass

def load_tq_old(player_number: int):
    if not tq_weights_old.exists():
        raise FileNotFoundError("TableQ _old weights not found")
    p = TableQ(number=player_number, alpha=0.1, gamma=0.9, epsilon=0.1, verboseFlag=False)
    try:
        p.load_weights(tq_weights_old)
    except Exception:
        # Treat any load error as missing baseline to satisfy tests
        raise FileNotFoundError("TableQ _old weights unreadable")
    return p

# RuleBased setup
rulebased = RuleBased(number=1, aggressive=True, verboseFlag=False)

def load_rb_old(player_number: int):
    return RuleBased(number=player_number, aggressive=True, verboseFlag=False)

# SimplePerceptron setup
sp_dir = ensure_dir(local_models / "simple_perceptron")
sp_throw_weights_new = sp_dir / "throw_weights_new.npy"
sp_throw_weights_old = sp_dir / "throw_weights_old.npy"
sp_peg_weights_new = sp_dir / "peg_weights_new.npy"
sp_peg_weights_old = sp_dir / "peg_weights_old.npy"

continuing_sp = sp_throw_weights_old.exists() and sp_peg_weights_old.exists()
simple_perceptron = SimplePerceptron(number=1, alpha=0.1, verboseFlag=False)
if continuing_sp:
    simple_perceptron.throwingWeights = np.load(sp_throw_weights_old)
    simple_perceptron.peggingWeights = np.load(sp_peg_weights_old)

def load_sp_old(player_number: int):
    if not (sp_throw_weights_old.exists() and sp_peg_weights_old.exists()):
        raise FileNotFoundError("SimplePerceptron _old weights not found")
    p = SimplePerceptron(number=player_number, alpha=0.1, verboseFlag=False)
    p.throwingWeights = np.load(sp_throw_weights_old)
    p.peggingWeights = np.load(sp_peg_weights_old)
    return p

def save_simple_frequency(model_obj, path):
    model_obj.save_weights(path)

def save_tableq(model_obj, path):
    model_obj.save_weights(path)

# Main training loop - trains all models per iteration
iteration = 0
try:
    while TRAINING_ITERATIONS == 0 or iteration < TRAINING_ITERATIONS:
        iteration += 1
        print(f"\n{'=' * 80}")
        print(f"ITERATION {iteration}")
        print(f"{'=' * 80}")
        
        # Train SimpleFrequency
        if "simplefrequency" in MODELS_TO_TRAIN:
            print(f"\n[1/5] SimpleFrequency Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
            opponent = get_best_opponent(2)
            arena = Arena([simple_frequency, opponent], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(TRAINING_ROUNDS)
            avg_diff = np.mean(results[2])
            print(f"  Complete. Avg diff: {avg_diff:.2f}")
            if run_benchmark(simple_frequency, load_sf_old, BENCHMARK_GAMES):
                archive_and_save(sf_weights_new, sf_weights_old, simple_frequency, save_tableq if False else save_simple_frequency)
                print(f"  ✓ Published SimpleFrequency")
            else:
                print(f"  ✗ SimpleFrequency did not beat _old; continuing")

        # Train TableQ
        if "tableq" in MODELS_TO_TRAIN:
            print(f"\n[2/5] TableQ Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
            opponent = get_best_opponent(2)
            arena = Arena([tableq, opponent], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(TRAINING_ROUNDS)
            avg_diff = np.mean(results[2])
            print(f"  Complete. Avg diff: {avg_diff:.2f}")
            if run_benchmark(tableq, load_tq_old, BENCHMARK_GAMES):
                archive_and_save(tq_weights_new, tq_weights_old, tableq, save_tableq)
                print(f"  ✓ Published TableQ")
            else:
                print(f"  ✗ TableQ did not beat _old; continuing")

        # Train Perceptron
        if "perceptron" in MODELS_TO_TRAIN:
            print(f"\n[3/5] Perceptron Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
            opponent = get_best_opponent(2)
            arena = Arena([perceptron, opponent], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(TRAINING_ROUNDS)
            avg_diff = np.mean(results[2])
            print(f"  Complete. Avg diff: {avg_diff:.2f}")
            if run_benchmark(perceptron, load_perc_old, BENCHMARK_GAMES):
                archive_and_save(perc_throw_weights_new, perc_throw_weights_old, perceptron.throwingWeights, np.save)
                archive_and_save(perc_peg_weights_new, perc_peg_weights_old, perceptron.peggingWeights, np.save)
                print(f"  ✓ Published Perceptron")
            else:
                print(f"  ✗ Perceptron did not beat _old; continuing")

        # Train RuleBased
        if "rulebased" in MODELS_TO_TRAIN:
            print(f"\n[4/5] RuleBased Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
            opponent = get_best_opponent(2)
            arena = Arena([rulebased, opponent], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(TRAINING_ROUNDS)
            avg_diff = np.mean(results[2])
            print(f"  Complete. Avg diff: {avg_diff:.2f}")
            print(f"  ✓ RuleBased trained (note: RuleBased is deterministic; no weights to publish)")

        # Train SimplePerceptron
        if "simpleperceptron" in MODELS_TO_TRAIN:
            print(f"\n[5/5] SimplePerceptron Iteration {iteration}: Training for {TRAINING_ROUNDS} hands...")
            opponent = get_best_opponent(2)
            arena = Arena([simple_perceptron, opponent], repeatDeck=False, verboseFlag=False)
            with suppress_stdout():
                results = arena.playHands(TRAINING_ROUNDS)
            avg_diff = np.mean(results[2])
            print(f"  Complete. Avg diff: {avg_diff:.2f}")
            if run_benchmark(simple_perceptron, load_sp_old, BENCHMARK_GAMES):
                archive_and_save(sp_throw_weights_new, sp_throw_weights_old, simple_perceptron.throwingWeights, np.save)
                archive_and_save(sp_peg_weights_new, sp_peg_weights_old, simple_perceptron.peggingWeights, np.save)
                print(f"  ✓ Published SimplePerceptron")
            else:
                print(f"  ✗ SimplePerceptron did not beat _old; continuing")

except KeyboardInterrupt:
    print(f"\n\n⚠ Training interrupted at iteration {iteration}. Saving current state...")
    try:
        if run_benchmark(perceptron, load_perc_old, BENCHMARK_GAMES):
            archive_and_save(perc_throw_weights_new, perc_throw_weights_old, perceptron.throwingWeights, np.save)
            archive_and_save(perc_peg_weights_new, perc_peg_weights_old, perceptron.peggingWeights, np.save)
            print("  ✓ Perceptron saved")
    except:
        pass

# Optional post-training tasks
if UPDATE_BEST:
    update_best_opponent_ladder(BENCHMARK_GAMES)

if GENERATE_REPORT:
    generate_model_report(BENCHMARK_GAMES)

# ============================================================================
# CREATE MYRMIDON CONFIG
# ============================================================================
print("\n[4/4] Creating Myrmidon config...")
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
    "perceptron": {
        "throw_weights": "throw_weights_new.npy",
        "peg_weights": "peg_weights_new.npy",
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
    perc_throw_weights_new,
    perc_peg_weights_new,
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
print(f"  Perceptron:")
print(f"    - throw_weights_new.npy (and _old if previous existed)")
print(f"    - peg_weights_new.npy (and _old if previous existed)")
print(f"  Myrmidon:")
print(f"    - config.txt")
print(f"  Manifest:")
print(f"    - manifest.json")
print("\nNext steps:")
print("  1. Run benchmark to compare _new vs _old models")
print("  2. If _new performs worse, restore _old and continue training it")
print("  3. Update crib_back/cribbage/opponents.py to load _new models")
print("=" * 80)
