"""Analyse Chess960 experiment logs written by run_multiple_games_960.py.

Usage:
    uv run python analyze_960.py                                # all runs under _logs/chess960
    uv run python analyze_960.py _logs/chess960/A1-pos518_B1-legal
    uv run python analyze_960.py --stockfish /opt/homebrew/bin/stockfish --depth 14
    uv run python analyze_960.py --no-engine                    # skip move-quality analysis

Writes to --out (default _analysis/):
    games.csv            one row per LLM side per game (all metrics below)
    conditions.csv       aggregated per model x condition (A start position, B legal moves, C opponent)
    moves.csv            one row per analysed LLM move (ply, phase, centipawn loss, best-move match);
                         used for the opening-gap (difference-in-differences) test in the notebook
    illegal_attempts.csv one row per illegal move attempt, with the standard-pattern classification

Metrics:
    - result / win rate, and loss cause (checkmate, illegal_move_forfeit, format_forfeit, ...)
    - comprehension check (stated back rank correct?); summary also reported on passing games only
    - illegal attempts per LLM turn, split by type (castling, piece_movement, wrong_piece_on_square, ...)
    - first-attempt illegal rate (share of turns whose first make_move was illegal)
    - format normalizations (replies rewritten from <invoke> tool-call syntax)
    - move quality with Stockfish (LLM moves only): average centipawn loss, blunder (>=300cp),
      mistake (100-299cp), inaccuracy (50-99cp) rates and best-move match, split into
      opening (plies 1-15 of the game) and later plies
    - standard-pattern errors (Chess960 games): an illegal attempt is a "phantom piece" move when it
      starts on the mover's back rank and would be a valid move for the piece the *standard* setup has
      on that square (e.g. g8f6 when g8 holds a rook). Its chance rate is the same test against 100
      random other Chess960 arrangements; standard castling notation (e1g1, e8c8, ...) is counted apart
    - performance Elo vs Dragon, using the repo's anchor Elo = 125 * (level + 1) and a 35-point
      white advantage for the engine (only for Dragon conditions with mixed results)
"""

import argparse
import csv
import io
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import chess
import chess.engine
import chess.pgn

LLM_TYPES = {"LLM_WHITE", "LLM_BLACK", "LLM_NON"}
OPENING_PLIES = 15
CP_CLAMP = 1000  # evaluations are clamped to +-1000cp before computing loss (mate = +-1000)
BLUNDER_CP, MISTAKE_CP, INACCURACY_CP = 300, 100, 50
DRAGON_WHITE_ADVANTAGE = 35.0
ILLEGAL_CATEGORIES = [
    "castling",
    "piece_movement",
    "wrong_piece_on_square",
    "leaves_king_in_check",
    "missing_promotion",
    "unparseable",
]


STANDARD_BACK_RANK = "RNBQKBNR"
STANDARD_CASTLING_UCI = {"e1g1", "e1c1", "e8g8", "e8c8"}
CHANCE_ARRANGEMENTS = 100  # random other Chess960 setups used for the chance baseline


# ---------------------------------------------------------------------------
# Standard-pattern errors
# ---------------------------------------------------------------------------


def back_rank_letters(board: chess.Board, color: chess.Color) -> str:
    rank = 0 if color == chess.WHITE else 7
    out = []
    for file in range(8):
        piece = board.piece_at(chess.square(file, rank))
        out.append(piece.symbol().upper() if piece else "-")
    return "".join(out)


def arrangement_of_position(position: int) -> str:
    return back_rank_letters(chess.Board.from_chess960_pos(position), chess.WHITE)


_CHANCE_CACHE: Dict[int, List[str]] = {}


def chance_arrangements(exclude: Optional[int]) -> List[str]:
    """A fixed (seeded) set of other Chess960 arrangements, excluding #518 and the game's own position."""
    key = -1 if exclude is None else exclude
    if key not in _CHANCE_CACHE:
        import random

        pool = [p for p in range(960) if p not in (518, exclude)]
        sample = random.Random(960).sample(pool, CHANCE_ARRANGEMENTS)
        _CHANCE_CACHE[key] = [arrangement_of_position(p) for p in sample]
    return _CHANCE_CACHE[key]


def fits_arrangement(board: chess.Board, move: chess.Move, arrangement: str, start_rank: str) -> bool:
    """Would `move` be a valid (pseudo-legal) move if the back-rank square it starts from held the piece
    that `arrangement` puts there? Only counted when that piece differs from this game's own start piece,
    so playing the real setup never counts."""
    color = board.turn
    home_rank = 0 if color == chess.WHITE else 7
    if chess.square_rank(move.from_square) != home_rank:
        return False
    file = chess.square_file(move.from_square)
    hyp_letter, start_letter = arrangement[file], start_rank[file]
    if hyp_letter == start_letter:
        return False
    hyp = board.copy(stack=False)
    hyp.set_piece_at(move.from_square, chess.Piece(chess.Piece.from_symbol(hyp_letter).piece_type, color))
    try:
        return hyp.is_pseudo_legal(move)
    except Exception:
        return False


def classify_standard_pattern(attempt: Dict[str, Any], start_fen: Optional[str], chess960: bool,
                              position: Optional[int]) -> Dict[str, Any]:
    out = {"standard_castling_notation": attempt.get("move", "") in STANDARD_CASTLING_UCI,
           "phantom_standard": None, "phantom_chance": None}
    if not chess960 or position == 518 or not start_fen or not attempt.get("fen"):
        return out  # only defined when the real setup differs from the standard one
    try:
        board = chess.Board(attempt["fen"], chess960=True)
        move = chess.Move.from_uci(attempt["move"])
        start_rank = back_rank_letters(chess.Board(start_fen, chess960=True), board.turn)
    except ValueError:
        return out
    out["phantom_standard"] = fits_arrangement(board, move, STANDARD_BACK_RANK, start_rank)
    others = chance_arrangements(position)
    out["phantom_chance"] = sum(fits_arrangement(board, move, arr, start_rank) for arr in others) / len(others)
    return out


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def iter_game_logs(paths: Iterable[str]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for base in paths:
        if os.path.isfile(base):
            candidates = [base]
        else:
            candidates = []
            for root, _dirs, files in os.walk(base):
                for name in files:
                    if name.endswith(".json") and not name.startswith("_"):
                        candidates.append(os.path.join(root, name))
        for path in sorted(candidates):
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(data, dict) and "settings" in data and "pgn" in data:
                yield path, data


def condition_of(game: Dict[str, Any], side: str) -> Dict[str, str]:
    s = game["settings"]
    if not s.get("chess960"):
        start = "standard-chess"
    elif s.get("chess960_position") == 518:
        start = "A1-standard-pos518"
    else:
        start = "A2-chess960"
    if s.get("chess960") and s.get("chess960_prompt_hint") is False:
        start += "-uninformed"  # prompt did not say it is Chess960
    legal = "B1-legal-moves" if s.get("legal_moves_enabled", True) else "B2-no-legal-moves"

    opp_type = s["black_player_type"] if side == "white" else s["white_player_type"]
    if opp_type == "RANDOM_PLAYER":
        opponent = "C1-random"
    elif opp_type == "CHESS_ENGINE_DRAGON":
        opponent = f"C2-dragon-lvl{s.get('dragon_level')}"
    elif opp_type == "CHESS_ENGINE_STOCKFISH":
        opponent = f"C2-stockfish-lvl{s.get('stockfish_level')}"
    else:
        opponent = opp_type.lower()
    if opponent.startswith("C2-") and s.get("engine_depth") is not None:
        opponent += f"-depth{s['engine_depth']}"  # a depth-limited engine is a different opponent
    return {"start": start, "legal_moves": legal, "opponent": opponent}


# ---------------------------------------------------------------------------
# Move quality
# ---------------------------------------------------------------------------


def _cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    value = score.pov(color).score(mate_score=100_000)
    return max(-CP_CLAMP, min(CP_CLAMP, value))


def move_quality(
    game: Dict[str, Any], side: str, engine: Optional[chess.engine.SimpleEngine], limit: chess.engine.Limit
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    empty = {f"{phase}_{k}": None for phase in ("opening", "later", "all")
             for k in ("moves", "acpl", "blunder_rate", "mistake_rate", "inaccuracy_rate", "best_move_rate")}
    if engine is None:
        return empty, []

    pgn_game = chess.pgn.read_game(io.StringIO(game["pgn"]))
    if pgn_game is None:
        return empty, []
    board = pgn_game.board()
    color = chess.WHITE if side == "white" else chess.BLACK

    buckets: Dict[str, List[Dict[str, Any]]] = {"opening": [], "later": []}
    for ply, move in enumerate(pgn_game.mainline_moves(), start=1):
        if board.turn != color:
            board.push(move)
            continue
        before = engine.analyse(board, limit)
        best_move = before.get("pv", [None])[0]
        best_cp = _cp(before["score"], color)
        board.push(move)
        if board.is_game_over():
            outcome = board.outcome()
            if outcome and outcome.winner == color:
                played_cp = CP_CLAMP
            elif outcome and outcome.winner is None:
                played_cp = 0
            else:
                played_cp = -CP_CLAMP
        else:
            after = engine.analyse(board, limit)
            played_cp = _cp(after["score"], color)
        loss = max(0, best_cp - played_cp)
        buckets["opening" if ply <= OPENING_PLIES else "later"].append(
            {"ply": ply, "phase": "opening" if ply <= OPENING_PLIES else "later",
             "loss": loss, "best": best_move is not None and move == best_move}
        )

    out: Dict[str, Any] = {}
    buckets["all"] = buckets["opening"] + buckets["later"]
    for phase, rows in buckets.items():
        n = len(rows)
        out[f"{phase}_moves"] = n
        if n == 0:
            for k in ("acpl", "blunder_rate", "mistake_rate", "inaccuracy_rate", "best_move_rate"):
                out[f"{phase}_{k}"] = None
            continue
        losses = [r["loss"] for r in rows]
        out[f"{phase}_acpl"] = sum(losses) / n
        out[f"{phase}_blunder_rate"] = sum(l >= BLUNDER_CP for l in losses) / n
        out[f"{phase}_mistake_rate"] = sum(MISTAKE_CP <= l < BLUNDER_CP for l in losses) / n
        out[f"{phase}_inaccuracy_rate"] = sum(INACCURACY_CP <= l < MISTAKE_CP for l in losses) / n
        out[f"{phase}_best_move_rate"] = sum(r["best"] for r in rows) / n
    per_move = [{"ply": r["ply"], "phase": r["phase"], "cp_loss": r["loss"], "best_move": r["best"]}
                for r in buckets["all"]]
    return out, per_move


# ---------------------------------------------------------------------------
# Per-game rows
# ---------------------------------------------------------------------------


def game_rows(
    path: str, game: Dict[str, Any], engine: Optional[chess.engine.SimpleEngine], limit: chess.engine.Limit,
    moves_out: Optional[List[Dict[str, Any]]] = None, attempts_out: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    settings = game["settings"]
    rows = []
    for side in ("white", "black"):
        if settings.get(f"{side}_player_type") not in LLM_TYPES:
            continue
        player = game[f"player_{side}"]
        outcome = game.get("outcome", {})
        losing_side = outcome.get("losing_side")
        winner = game.get("winner")
        if losing_side == side:
            result = "loss"
        elif losing_side is not None:
            result = "win"
        elif winner in (None, "NONE"):
            result = "draw"
        else:  # older logs without losing_side
            result = "win" if winner == player.get("name") else "loss"

        comp = game.get("comprehension", {}).get(side)
        turns = player.get("turns") or 0
        by_type = player.get("illegal_attempts_by_type", {}) or {}
        illegal_total = sum(by_type.values())
        first_total = player.get("first_attempt_total") or 0

        row: Dict[str, Any] = {
            "log_file": path,
            "model": player.get("model", ""),
            "side": side,
            **condition_of(game, side),
            "chess960_position": settings.get("chess960_position"),
            "result": result,
            "score": {"win": 1.0, "draw": 0.5, "loss": 0.0}[result],
            "loss_cause": outcome.get("loss_cause") if result == "loss" else "",
            "termination_reason": game.get("reason"),
            "plies": game.get("number_of_moves"),
            "llm_turns": turns,
            "comprehension_checked": comp is not None,
            "comprehension_correct": (comp or {}).get("correct") if comp is not None else "",
            "comprehension_parsed": (comp or {}).get("parsed") if comp is not None else "",
            "comprehension_expected": (comp or {}).get("expected") if comp is not None else "",
            "illegal_attempts": illegal_total,
            "illegal_per_turn": illegal_total / turns if turns else None,
            "first_attempt_illegal_rate": (player.get("first_attempt_illegal", 0) / first_total) if first_total else None,
            "wrong_actions": player.get("wrong_actions", 0),
            "format_normalizations": player.get("format_normalizations", 0),
            "disallowed_action_requests": player.get("disallowed_action_requests", 0),
        }
        for cat in ILLEGAL_CATEGORIES:
            row[f"illegal_{cat}"] = by_type.get(cat, 0)
            row[f"illegal_{cat}_per_turn"] = by_type.get(cat, 0) / turns if turns else None
        quality, per_move = move_quality(game, side, engine, limit)
        row.update(quality)

        key = {k: row[k] for k in ("log_file", "model", "side", "start", "legal_moves", "opponent", "chess960_position")}
        if moves_out is not None:
            moves_out.extend({**key, **m} for m in per_move)

        sp_counts = {"phantom_standard": 0, "phantom_chance": 0.0, "standard_castling_notation": 0}
        for attempt in player.get("illegal_attempts", []) or []:
            sp = classify_standard_pattern(attempt, settings.get("starting_fen"), bool(settings.get("chess960")),
                                           settings.get("chess960_position"))
            sp_counts["phantom_standard"] += int(bool(sp["phantom_standard"]))
            sp_counts["phantom_chance"] += sp["phantom_chance"] or 0.0
            sp_counts["standard_castling_notation"] += int(sp["standard_castling_notation"])
            if attempts_out is not None:
                attempts_out.append({**key, "ply": attempt.get("ply"), "move": attempt.get("move"),
                                     "category": attempt.get("category"), "fen": attempt.get("fen"), **sp})
        row["standard_pattern_phantom"] = sp_counts["phantom_standard"]
        row["standard_pattern_phantom_chance"] = sp_counts["phantom_chance"]
        row["standard_castling_notation"] = sp_counts["standard_castling_notation"]
        # Dragon Elo anchors only hold for the time-limited engine, not for a depth-limited one
        row["dragon_level"] = (
            settings.get("dragon_level")
            if row["opponent"].startswith("C2-dragon") and settings.get("engine_depth") is None
            else None
        )
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and v != ""]
    return sum(vals) / len(vals) if vals else None


def _weighted_rate(rows: List[Dict[str, Any]], key: str, n_key: str) -> Optional[float]:
    """Move-weighted average of a per-game rate (e.g. blunder_rate weighted by moves)."""
    num = den = 0.0
    for r in rows:
        if r.get(key) is not None and r.get(n_key):
            num += r[key] * r[n_key]
            den += r[n_key]
    return num / den if den else None


def performance_elo(rows: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Maximum-likelihood Elo vs Dragon anchors (125*(level+1)); engine plays white (+35)."""
    blocks = [(125.0 * (r["dragon_level"] + 1), r["score"]) for r in rows if r.get("dragon_level") is not None]
    if not blocks:
        return None, None
    scores = [s for _, s in blocks]
    if all(s == 1.0 for s in scores) or all(s == 0.0 for s in scores):
        return None, None  # undefined with all wins or all losses

    def expected(r: float, opp: float) -> float:
        return 1.0 / (1.0 + 10 ** ((opp + DRAGON_WHITE_ADVANTAGE - r) / 400.0))

    lo, hi = -1000.0, 4000.0
    for _ in range(100):  # bisection on sum(score - expected) = 0
        mid = (lo + hi) / 2
        if sum(s - expected(mid, opp) for opp, s in blocks) > 0:
            lo = mid
        else:
            hi = mid
    elo = (lo + hi) / 2
    info = sum(
        (math.log(10) / 400) ** 2 * expected(elo, opp) * (1 - expected(elo, opp)) for opp, _ in blocks
    )
    moe = 1.96 / math.sqrt(info) if info > 0 else None
    return elo, moe


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    out: Dict[str, Any] = {"games": n}
    for res in ("win", "draw", "loss"):
        out[f"{res}_rate"] = sum(r["result"] == res for r in rows) / n if n else None
    causes = defaultdict(int)
    for r in rows:
        if r["result"] == "loss":
            causes[r["loss_cause"] or "unknown"] += 1
    for cause in ("checkmate", "illegal_move_forfeit", "format_forfeit", "mixed_forfeit", "max_turns_forfeit", "error"):
        out[f"losses_{cause}"] = causes.get(cause, 0)

    turns = sum(r["llm_turns"] or 0 for r in rows)
    out["llm_turns"] = turns
    out["illegal_per_turn"] = sum(r["illegal_attempts"] for r in rows) / turns if turns else None
    for cat in ILLEGAL_CATEGORIES:
        out[f"illegal_{cat}_per_turn"] = sum(r[f"illegal_{cat}"] for r in rows) / turns if turns else None
    out["first_attempt_illegal_rate"] = _mean([r["first_attempt_illegal_rate"] for r in rows])
    out["phantom_standard_attempts"] = sum(r.get("standard_pattern_phantom", 0) for r in rows)
    out["phantom_chance_expected"] = sum(r.get("standard_pattern_phantom_chance", 0.0) for r in rows)
    out["standard_castling_notation_attempts"] = sum(r.get("standard_castling_notation", 0) for r in rows)
    out["format_normalizations_per_turn"] = sum(r["format_normalizations"] for r in rows) / turns if turns else None

    checked = [r for r in rows if r["comprehension_checked"]]
    out["comprehension_checked"] = len(checked)
    out["comprehension_pass_rate"] = (
        sum(bool(r["comprehension_correct"]) for r in checked) / len(checked) if checked else None
    )

    for phase in ("opening", "later", "all"):
        out[f"{phase}_moves"] = sum(r.get(f"{phase}_moves") or 0 for r in rows)
        for k in ("acpl", "blunder_rate", "mistake_rate", "inaccuracy_rate", "best_move_rate"):
            out[f"{phase}_{k}"] = _weighted_rate(rows, f"{phase}_{k}", f"{phase}_moves")

    out["elo_vs_dragon"], out["elo_moe_95"] = performance_elo(rows)
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if v is None else (round(v, 4) if isinstance(v, float) else v)) for k, v in r.items()})


def find_stockfish(explicit: Optional[str]) -> Optional[str]:
    for candidate in (explicit, shutil.which("stockfish"), "/opt/homebrew/bin/stockfish", "/usr/games/stockfish"):
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Analyse Chess960 experiment logs.")
    parser.add_argument("paths", nargs="*", default=["_logs/chess960"], help="log folders or game JSON files")
    parser.add_argument("--out", default="_analysis", help="output folder (default _analysis)")
    parser.add_argument("--stockfish", help="Stockfish binary (default: auto-detect)")
    parser.add_argument("--depth", type=int, default=12, help="analysis depth per position (default 12)")
    parser.add_argument("--time", type=float, help="analysis seconds per position (overrides --depth)")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-engine", action="store_true", help="skip move-quality analysis")
    parser.add_argument("--exclude-failed-comprehension", action="store_true",
                        help="drop games where the comprehension check was answered wrongly")
    args = parser.parse_args(argv)

    logs = list(iter_game_logs(args.paths))
    if not logs:
        print(f"No experiment game logs found in {args.paths}")
        return 1

    engine = None
    if not args.no_engine:
        sf = find_stockfish(args.stockfish)
        if sf is None:
            print("Stockfish not found; use --stockfish PATH or --no-engine. Skipping move quality.")
        else:
            engine = chess.engine.SimpleEngine.popen_uci(sf)
            engine.configure({"Threads": args.threads})
    limit = chess.engine.Limit(time=args.time) if args.time else chess.engine.Limit(depth=args.depth)

    rows: List[Dict[str, Any]] = []
    move_rows: List[Dict[str, Any]] = []
    attempt_rows: List[Dict[str, Any]] = []
    skipped_errors: List[str] = []
    try:
        for i, (path, game) in enumerate(logs, start=1):
            print(f"[{i}/{len(logs)}] {path}", file=sys.stderr)
            if game.get("outcome", {}).get("loss_cause") == "error" or game.get("reason") == "ERROR OCCURED":
                skipped_errors.append(path)  # interrupted by an API/engine error: not a chess result
                continue
            rows.extend(game_rows(path, game, engine, limit, move_rows, attempt_rows))
    finally:
        if engine is not None:
            engine.quit()

    if args.exclude_failed_comprehension:
        rows = [r for r in rows if not (r["comprehension_checked"] and not r["comprehension_correct"])]
        kept = {(r["log_file"], r["side"]) for r in rows}
        move_rows = [m for m in move_rows if (m["log_file"], m["side"]) in kept]
        attempt_rows = [a for a in attempt_rows if (a["log_file"], a["side"]) in kept]

    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["start"], r["legal_moves"], r["opponent"])].append(r)

    summaries = []
    for (model, start, legal, opponent), group in sorted(groups.items()):
        base = {"model": model, "start": start, "legal_moves": legal, "opponent": opponent}
        summaries.append({**base, "subset": "all_games", **summarize(group)})
        passed = [r for r in group if r["comprehension_checked"] and r["comprehension_correct"]]
        if any(r["comprehension_checked"] for r in group):
            summaries.append({**base, "subset": "comprehension_passed", **summarize(passed)} if passed
                             else {**base, "subset": "comprehension_passed", "games": 0})

    os.makedirs(args.out, exist_ok=True)
    write_csv(os.path.join(args.out, "games.csv"), rows)
    write_csv(os.path.join(args.out, "conditions.csv"), summaries)
    write_csv(os.path.join(args.out, "moves.csv"), move_rows)
    write_csv(os.path.join(args.out, "illegal_attempts.csv"), attempt_rows)

    try:
        from tabulate import tabulate

        cols = ["model", "start", "legal_moves", "opponent", "subset", "games", "win_rate", "loss_rate",
                "illegal_per_turn", "all_acpl", "elo_vs_dragon"]
        table = [[s.get(c) if not isinstance(s.get(c), float) else round(s[c], 3) for c in cols] for s in summaries]
        print(tabulate(table, headers=cols))
    except ImportError:
        pass
    if skipped_errors:
        print(f"\nSkipped {len(skipped_errors)} game(s) that ended in an error (rerun to replace them):")
        for p in skipped_errors:
            print(f"  {p}")
    print(f"\nWrote {os.path.join(args.out, 'games.csv')} and {os.path.join(args.out, 'conditions.csv')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
