"""Run a batch of Chess960 (Fischer Random) games and aggregate the results.

Usage:
    uv run python run_multiple_games_960.py                      # settings below
    uv run python run_multiple_games_960.py --help               # all flags

Experiment conditions (command-line flags override the settings below):
    A1 standard start, 960 pipeline:  --positions 518
    A2 Chess960 fixed sample:         --positions a2   (the 30 positions in A2_POSITIONS)
    B1 with legal-move list:          (default)
    B2 without legal-move list:       --no-legal-moves
    C1 random opponent:               --opponent random
    C2 engine opponent:               --opponent stockfish --level 0 --engine-depth 1   (or dragon)
    Comprehension check:              --comprehension-check
    Shuffled, not told it is 960:     --no-variant-hint   (combine with --positions / --sample)
    Engine-vs-engine baseline:        --opponent stockfish --player stockfish --level 5 --player-level 5

Models/keys come from .env (MODEL_KIND_W / MODEL_KIND_B etc.), exactly as for run_multiple_games.py.
Analyse finished runs with: uv run python analyze_960.py

Logs go to _logs/chess960/<category>/<model>/<timestamp>/:
  - one <time>.json per game (includes chess960_position and starting_fen)
  - _aggregate_results.json with totals plus the list of start positions used
  - _run.json with run metadata, output.txt with the console output

Set CHESS960 = False to run the *same* setup on standard chess (logged under
_logs/chess960_baseline/...), which gives a like-for-like baseline for comparison.
"""

import argparse
import json
import os
import random
import statistics
from typing import Any, Dict, List, Optional

import llm_chess
from get_run_metadata import collect_run_metadata, write_run_metadata
from run_multiple_games import build_log_folder
from utils import get_llms, setup_console_logging


# ---------------------------------------------------------------------------
# Launch settings - edit these for a new experiment
# ---------------------------------------------------------------------------
NUM_REPETITIONS = 10

CHESS960 = True  # False = identical run on standard chess (baseline)

# Fixed sample for condition A2: 30 positions drawn once (seed 960, without replacement, #518 excluded).
# Keep this list identical for every model.
A2_POSITIONS: List[int] = [
    45, 67, 171, 177, 235, 257, 266, 298, 316, 319, 357, 388, 397, 490, 492,
    539, 565, 573, 624, 720, 733, 780, 793, 842, 865, 884, 904, 911, 926, 939,
]

# Start positions (Scharnagl numbers 0-959):
# - None: random positions drawn without replacement (see SEED / EXCLUDE_STANDARD_POSITION)
# - list of ints: used in order, cycling if NUM_REPETITIONS is larger
CHESS960_POSITIONS: Optional[List[int]] = None
SEED: Optional[int] = 960  # makes the random positions reproducible; None = different every run
EXCLUDE_STANDARD_POSITION = True  # never draw #518 (the normal chess setup) at random

# Experiment switches
VARIANT_HINT = True  # False = shuffled position, but the prompt does not say it is Chess960
LEGAL_MOVES = True  # False = condition B2 (LLM cannot call get_legal_moves)
COMPREHENSION_CHECK = False  # ask the LLM to state its back rank before move 1
MAX_FAILED_ATTEMPTS = 3  # wrong replies within one move before the game is forfeited
MAX_GAME_MOVES = 200  # plies before the game is stopped as a draw

WHITE_PLAYER_TYPE = llm_chess.PlayerType.RANDOM_PLAYER
BLACK_PLAYER_TYPE = llm_chess.PlayerType.LLM_BLACK
ENGINE_LEVEL = 1  # level of the white engine (Stockfish 0-20, Dragon 1-25)
BLACK_ENGINE_LEVEL: Optional[int] = None  # level of the black engine (engine-vs-engine); None = ENGINE_LEVEL
ENGINE_TIME_PER_MOVE = 0.1  # seconds (cap per move)
ENGINE_DEPTH: Optional[int] = None  # search-depth limit, e.g. 1; None = time limit only. Reproducible across computers.
STOCKFISH_PATH: Optional[str] = None  # None = keep llm_chess.stockfish_path
DRAGON_PATH: Optional[str] = None  # None = keep llm_chess.dragon_path

# Model settings. Leave REASONING_EFFORT = None for Anthropic (it is not supported there;
# Claude 5.x models think adaptively by default). Use e.g. "high" for OpenAI-style models.
REASONING_EFFORT: Optional[str] = None
HYPERPARAMS: Dict[str, Any] = {
    # Sampling params are dropped automatically for Claude 5.x models.
    # "temperature": 0.3,
    # "top_p": 1.0,
}

# Pacing / retries
THROTTLE_DELAY = 1  # seconds between moves
DIALOG_TURN_DELAY = 1  # seconds between turns inside one move
MAX_API_RETRIES = 6
API_RETRY_DELAY = 60  # base seconds, exponential backoff

STORE_INDIVIDUAL_LOGS = True
LOG_FOLDER: Optional[str] = None  # None = computed automatically
RUN_LABEL: Optional[str] = None  # optional extra tag in the log folder name


def pick_positions(n: int) -> List[int]:
    """Return the start position for each of the n games."""
    if CHESS960_POSITIONS:
        for p in CHESS960_POSITIONS:
            if not 0 <= p <= 959:
                raise ValueError(f"Invalid Chess960 position {p}; must be 0-959")
        return [CHESS960_POSITIONS[i % len(CHESS960_POSITIONS)] for i in range(n)]

    rng = random.Random(SEED)
    pool = [p for p in range(960) if not (EXCLUDE_STANDARD_POSITION and p == 518)]
    sample = rng.sample(pool, min(n, len(pool)))  # no repeats until the pool is exhausted
    return [sample[i % len(sample)] for i in range(n)]


def condition_label() -> str:
    """Folder tag describing the start-position and legal-move conditions, e.g. 'A1-pos518_B1-legal'."""
    if not CHESS960:
        start = "standard-chess"
    elif CHESS960_POSITIONS and set(CHESS960_POSITIONS) == {518}:
        start = "A1-pos518"
    elif CHESS960_POSITIONS:
        start = f"A2-{len(set(CHESS960_POSITIONS))}pos"
    else:
        start = "A2-random"
    if CHESS960 and not VARIANT_HINT:
        start += "-nohint"
    parts = [start, "B1-legal" if LEGAL_MOVES else "B2-nolegal"]
    if COMPREHENSION_CHECK:
        parts.append("cc")
    engines = (llm_chess.PlayerType.CHESS_ENGINE_STOCKFISH, llm_chess.PlayerType.CHESS_ENGINE_DRAGON)
    if ENGINE_DEPTH is not None and (WHITE_PLAYER_TYPE in engines or BLACK_PLAYER_TYPE in engines):
        parts.append(f"depth{ENGINE_DEPTH}")
    if RUN_LABEL:
        parts.append(RUN_LABEL)
    return "_".join(parts)


def _stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _new_side() -> Dict[str, Any]:
    return {
        "name": "",
        "model": "",
        "total_material": 0,
        "wrong_moves": 0,
        "wrong_actions": 0,
        "material_list": [],
        "reflections_used": 0,
        "reflections_used_before_board": 0,
    }


def _update_side(agg: Dict, side_key: str, player, game_stats: Dict) -> None:
    agg[side_key]["name"] = player.name
    llm_config = getattr(player, "llm_config", None)
    agg[side_key]["model"] = (
        llm_config["config_list"][0]["model"] if isinstance(llm_config, dict) else ""
    )
    material = game_stats["material_count"]["white" if side_key == "player_white" else "black"]
    agg[side_key]["total_material"] += material
    agg[side_key]["material_list"].append(material)
    agg[side_key]["wrong_moves"] += game_stats[side_key]["wrong_moves"]
    agg[side_key]["wrong_actions"] += game_stats[side_key]["wrong_actions"]
    agg[side_key]["reflections_used"] += game_stats[side_key].get("reflections_used", 0)
    agg[side_key]["reflections_used_before_board"] += game_stats[side_key].get(
        "reflections_used_before_board", 0
    )


def run_games() -> Dict[str, Any]:
    global LOG_FOLDER

    side_params: Dict[str, Any] = {"hyperparams": dict(HYPERPARAMS)}
    white_params = dict(side_params)
    black_params = dict(side_params)
    if REASONING_EFFORT:
        if WHITE_PLAYER_TYPE in (llm_chess.PlayerType.LLM_WHITE, llm_chess.PlayerType.LLM_NON):
            white_params["reasoning_effort"] = REASONING_EFFORT
        if BLACK_PLAYER_TYPE in (llm_chess.PlayerType.LLM_BLACK, llm_chess.PlayerType.LLM_NON):
            black_params["reasoning_effort"] = REASONING_EFFORT

    llm_config_white, llm_config_black = get_llms(
        white_hyperparams=white_params,
        black_hyperparams=black_params,
    )

    # Game settings
    black_level = BLACK_ENGINE_LEVEL if BLACK_ENGINE_LEVEL is not None else ENGINE_LEVEL
    llm_chess.white_player_type = WHITE_PLAYER_TYPE
    llm_chess.black_player_type = BLACK_PLAYER_TYPE
    llm_chess.stockfish_level = ENGINE_LEVEL
    llm_chess.dragon_level = ENGINE_LEVEL
    llm_chess.stockfish_time_per_move = ENGINE_TIME_PER_MOVE
    llm_chess.dragon_time_per_move = ENGINE_TIME_PER_MOVE
    llm_chess.engine_depth = ENGINE_DEPTH
    if STOCKFISH_PATH:
        llm_chess.stockfish_path = STOCKFISH_PATH
    if DRAGON_PATH:
        llm_chess.dragon_path = DRAGON_PATH
    llm_chess.enable_legal_moves = LEGAL_MOVES
    llm_chess.chess960_prompt_hint = VARIANT_HINT
    llm_chess.comprehension_check = COMPREHENSION_CHECK
    llm_chess.max_failed_attempts = MAX_FAILED_ATTEMPTS
    llm_chess.max_game_moves = MAX_GAME_MOVES
    llm_chess.record_experiment_stats = True
    llm_chess.throttle_delay = THROTTLE_DELAY
    llm_chess.dialog_turn_delay = DIALOG_TURN_DELAY
    llm_chess.max_api_retries = MAX_API_RETRIES
    llm_chess.api_retry_delay = API_RETRY_DELAY
    llm_chess.remove_text = llm_chess.DEFAULT_REMOVE_TEXT_REGEX
    llm_chess.chess960 = CHESS960

    engines = (llm_chess.PlayerType.CHESS_ENGINE_STOCKFISH, llm_chess.PlayerType.CHESS_ENGINE_DRAGON)
    if (
        WHITE_PLAYER_TYPE in engines
        and WHITE_PLAYER_TYPE == BLACK_PLAYER_TYPE
        and black_level != ENGINE_LEVEL
    ):
        raise ValueError(
            "Both sides use the same engine type, which shares one level setting in llm_chess.py. "
            "Use the same level for both, or mix stockfish and dragon."
        )
    if BLACK_PLAYER_TYPE == llm_chess.PlayerType.CHESS_ENGINE_STOCKFISH and WHITE_PLAYER_TYPE != BLACK_PLAYER_TYPE:
        llm_chess.stockfish_level = black_level
    if BLACK_PLAYER_TYPE == llm_chess.PlayerType.CHESS_ENGINE_DRAGON and WHITE_PLAYER_TYPE != BLACK_PLAYER_TYPE:
        llm_chess.dragon_level = black_level

    if LOG_FOLDER is None:
        base = build_log_folder(
            white_player_type=WHITE_PLAYER_TYPE,
            black_player_type=BLACK_PLAYER_TYPE,
            llm_config_white=llm_config_white,
            llm_config_black=llm_config_black,
            stockfish_level=llm_chess.stockfish_level,
            dragon_level=llm_chess.dragon_level,
        )
        LOG_FOLDER = base.replace("_logs/", f"_logs/chess960/{condition_label()}/", 1)

    non_llm_configs_white = [
        {**llm_config_white, "temperature": 0.0},
        {**llm_config_white, "temperature": 1.0},
    ]
    non_llm_configs_black = [
        {**llm_config_black, "temperature": 0.0},
        {**llm_config_black, "temperature": 1.0},
    ]

    setup_console_logging(LOG_FOLDER)

    run_metadata = None
    try:
        run_metadata = collect_run_metadata(
            log_folder_relative=LOG_FOLDER,
            num_repetitions=NUM_REPETITIONS,
            store_individual_logs=STORE_INDIVIDUAL_LOGS,
            llm_config_white=llm_config_white,
            llm_config_black=llm_config_black,
            non_llm_configs_white=non_llm_configs_white,
            non_llm_configs_black=non_llm_configs_black,
        )
        write_run_metadata(run_metadata, os.path.join(LOG_FOLDER, "_run.json"))
    except Exception as meta_err:
        print(f"[run_metadata] Error: {meta_err}")

    positions = pick_positions(NUM_REPETITIONS) if CHESS960 else [None] * NUM_REPETITIONS

    aggregate: Dict[str, Any] = {
        "run_metadata": {},
        "variant": "chess960" if CHESS960 else "standard",
        "condition": condition_label(),
        "experiment_settings": {
            "chess960": CHESS960,
            "positions": CHESS960_POSITIONS,
            "seed": SEED,
            "legal_moves": LEGAL_MOVES,
            "variant_hint": VARIANT_HINT,
            "comprehension_check": COMPREHENSION_CHECK,
            "max_failed_attempts": MAX_FAILED_ATTEMPTS,
            "max_game_moves": MAX_GAME_MOVES,
            "white_player_type": WHITE_PLAYER_TYPE.name,
            "black_player_type": BLACK_PLAYER_TYPE.name,
            "white_engine_level": ENGINE_LEVEL,
            "black_engine_level": black_level,
            "engine_time_per_move": ENGINE_TIME_PER_MOVE,
            "engine_depth": ENGINE_DEPTH,
        },
        "loss_causes": {},
        "illegal_attempts_by_type": {"white": {}, "black": {}},
        "comprehension_correct": {"white": 0, "black": 0},
        "total_games": 0,
        "white_wins": 0,
        "black_wins": 0,
        "draws": 0,
        "total_moves": 0,
        "reasons": {},
        "player_white": _new_side(),
        "player_black": _new_side(),
        "games": [],
    }
    moves_list: List[int] = []

    previous_position = llm_chess.chess960_position
    try:
        for i, position in enumerate(positions, start=1):
            llm_chess.chess960_position = position
            label = f"position #{position}" if CHESS960 else "standard chess"
            print(f"\n\033[95m=== Game {i}/{NUM_REPETITIONS} ({label}) ===\033[0m")

            game_stats, player_white, player_black = llm_chess.run(
                log_dir=LOG_FOLDER if STORE_INDIVIDUAL_LOGS else None,
                llm_config_white=llm_config_white,
                llm_config_black=llm_config_black,
                non_llm_configs_white=non_llm_configs_white,
                non_llm_configs_black=non_llm_configs_black,
            )

            moves_list.append(game_stats["number_of_moves"])
            aggregate["total_games"] += 1
            aggregate["total_moves"] += game_stats["number_of_moves"]

            if game_stats["winner"] == player_white.name:
                aggregate["white_wins"] += 1
                result = "1-0"
            elif game_stats["winner"] == player_black.name:
                aggregate["black_wins"] += 1
                result = "0-1"
            else:
                aggregate["draws"] += 1
                result = "1/2-1/2"

            reason = game_stats["reason"]
            aggregate["reasons"][reason] = aggregate["reasons"].get(reason, 0) + 1
            cause = game_stats.get("outcome", {}).get("loss_cause", "unknown")
            aggregate["loss_causes"][cause] = aggregate["loss_causes"].get(cause, 0) + 1
            for side in ("white", "black"):
                for cat, cnt in game_stats[f"player_{side}"].get("illegal_attempts_by_type", {}).items():
                    bucket = aggregate["illegal_attempts_by_type"][side]
                    bucket[cat] = bucket.get(cat, 0) + cnt
                if game_stats.get("comprehension", {}).get(side, {}).get("correct"):
                    aggregate["comprehension_correct"][side] += 1

            _update_side(aggregate, "player_white", player_white, game_stats)
            _update_side(aggregate, "player_black", player_black, game_stats)

            aggregate["games"].append(
                {
                    "game": i,
                    "chess960_position": game_stats.get("chess960_position"),
                    "starting_fen": game_stats.get("starting_fen", llm_chess.starting_fen),
                    "result": result,
                    "reason": reason,
                    "loss_cause": cause,
                    "losing_side": game_stats.get("outcome", {}).get("losing_side"),
                    "comprehension": {
                        side: game_stats.get("comprehension", {}).get(side, {}).get("correct")
                        for side in ("white", "black")
                        if side in game_stats.get("comprehension", {})
                    },
                    "number_of_moves": game_stats["number_of_moves"],
                    "black_wrong_moves": game_stats["player_black"]["wrong_moves"],
                    "black_wrong_actions": game_stats["player_black"]["wrong_actions"],
                    "white_wrong_moves": game_stats["player_white"]["wrong_moves"],
                    "white_wrong_actions": game_stats["player_white"]["wrong_actions"],
                }
            )
    finally:
        llm_chess.chess960_position = previous_position

    total = aggregate["total_games"]
    aggregate["average_moves"] = aggregate["total_moves"] / total if total else 0
    aggregate["std_dev_moves"] = _stdev(moves_list)
    for side in ("player_white", "player_black"):
        materials = aggregate[side].pop("material_list")
        aggregate[side]["avg_material"] = aggregate[side]["total_material"] / total if total else 0
        aggregate[side]["std_dev_material"] = _stdev(materials)

    if run_metadata:
        aggregate["run_metadata"] = run_metadata

    with open(os.path.join(LOG_FOLDER, "_aggregate_results.json"), "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)

    print("\n\n\033[92m" + json.dumps({k: v for k, v in aggregate.items() if k != "run_metadata"}, indent=4) + "\033[0m")
    print(f"\nLogs written to {LOG_FOLDER}")
    return aggregate


_PLAYER_CHOICES = {
    "random": llm_chess.PlayerType.RANDOM_PLAYER,
    "stockfish": llm_chess.PlayerType.CHESS_ENGINE_STOCKFISH,
    "dragon": llm_chess.PlayerType.CHESS_ENGINE_DRAGON,
}


def _parse_positions(text: str) -> List[int]:
    if text.strip().lower() == "a2":
        return list(A2_POSITIONS)
    try:
        positions = [int(p) for p in text.replace(" ", "").split(",") if p]
    except ValueError:
        raise argparse.ArgumentTypeError("positions must be 'a2' or comma-separated numbers 0-959")
    if not positions or any(not 0 <= p <= 959 for p in positions):
        raise argparse.ArgumentTypeError("positions must be numbers between 0 and 959")
    return positions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_multiple_games_960.py",
        description="Run a batch of Chess960 games (LLM plays black by default). Flags override the settings in the file.",
    )
    parser.add_argument("--games", type=int, help=f"number of games (default {NUM_REPETITIONS})")
    pos = parser.add_mutually_exclusive_group()
    pos.add_argument("--positions", type=_parse_positions,
                     help="'a2' (fixed 30-position sample), '518' (A1), or a comma-separated list, e.g. '45,67,171'")
    pos.add_argument("--sample", type=int, metavar="N",
                     help="draw N distinct random positions (excl. 518) with --seed, cycled over the games")
    pos.add_argument("--standard", action="store_true",
                     help="play standard chess instead (no Chess960 prompt/notation)")
    parser.add_argument("--seed", type=int, help=f"seed for random positions (default {SEED})")
    parser.add_argument("--no-legal-moves", action="store_true", help="condition B2: LLM cannot use get_legal_moves")
    parser.add_argument("--no-variant-hint", action="store_true",
                        help="shuffled position, but the prompt does not say it is Chess960 (no rules/castling explanation)")
    parser.add_argument("--comprehension-check", action="store_true",
                        help="ask the LLM to state its back rank before move 1 (logged)")
    parser.add_argument("--opponent", choices=["random", "stockfish", "dragon", "llm"],
                        help="white player (default: random)")
    parser.add_argument("--level", type=int, help=f"white engine level (default {ENGINE_LEVEL})")
    parser.add_argument("--player", choices=["llm", "random", "stockfish", "dragon"],
                        help="black player (default: llm)")
    parser.add_argument("--player-level", type=int, help="black engine level (engine-vs-engine baseline)")
    parser.add_argument("--engine-time", type=float, help=f"engine seconds per move (default {ENGINE_TIME_PER_MOVE})")
    parser.add_argument("--engine-depth", type=int,
                        help="limit engine search depth (e.g. 1); reproducible strength independent of computer speed")
    parser.add_argument("--stockfish-path", help="path to the Stockfish binary")
    parser.add_argument("--dragon-path", help="path to the Komodo Dragon binary")
    parser.add_argument("--max-failed-attempts", type=int,
                        help=f"wrong replies per move before forfeit (default {MAX_FAILED_ATTEMPTS})")
    parser.add_argument("--max-moves", type=int, help=f"max plies per game (default {MAX_GAME_MOVES})")
    parser.add_argument("--reasoning-effort", help="reasoning effort for OpenAI-style models (not Anthropic)")
    parser.add_argument("--label", help="extra tag added to the log folder name")
    parser.add_argument("--log-dir", help="explicit log folder (default: computed under _logs/chess960/)")
    return parser


def apply_args(args: argparse.Namespace) -> None:
    """Copy parsed command-line flags onto the module settings."""
    global NUM_REPETITIONS, CHESS960, CHESS960_POSITIONS, SEED, LEGAL_MOVES, COMPREHENSION_CHECK
    global WHITE_PLAYER_TYPE, BLACK_PLAYER_TYPE, ENGINE_LEVEL, BLACK_ENGINE_LEVEL, ENGINE_TIME_PER_MOVE
    global STOCKFISH_PATH, DRAGON_PATH, MAX_FAILED_ATTEMPTS, MAX_GAME_MOVES, REASONING_EFFORT
    global RUN_LABEL, LOG_FOLDER, VARIANT_HINT, ENGINE_DEPTH

    if args.games is not None:
        NUM_REPETITIONS = args.games
    if args.seed is not None:
        SEED = args.seed
    if args.standard:
        CHESS960 = False
    if args.positions is not None:
        CHESS960_POSITIONS = args.positions
    if args.sample is not None:
        rng = random.Random(SEED)
        pool = [p for p in range(960) if p != 518]
        CHESS960_POSITIONS = sorted(rng.sample(pool, args.sample))
    if args.no_legal_moves:
        LEGAL_MOVES = False
    if args.no_variant_hint:
        VARIANT_HINT = False
    if args.comprehension_check:
        COMPREHENSION_CHECK = True
    if args.opponent is not None:
        WHITE_PLAYER_TYPE = (
            llm_chess.PlayerType.LLM_WHITE if args.opponent == "llm" else _PLAYER_CHOICES[args.opponent]
        )
    if args.player is not None:
        BLACK_PLAYER_TYPE = (
            llm_chess.PlayerType.LLM_BLACK if args.player == "llm" else _PLAYER_CHOICES[args.player]
        )
    if args.level is not None:
        ENGINE_LEVEL = args.level
    if args.player_level is not None:
        BLACK_ENGINE_LEVEL = args.player_level
    if args.engine_time is not None:
        ENGINE_TIME_PER_MOVE = args.engine_time
    if args.engine_depth is not None:
        if args.engine_depth < 1:
            raise SystemExit("--engine-depth must be at least 1")
        ENGINE_DEPTH = args.engine_depth
    if args.stockfish_path:
        STOCKFISH_PATH = args.stockfish_path
    if args.dragon_path:
        DRAGON_PATH = args.dragon_path
    if args.max_failed_attempts is not None:
        MAX_FAILED_ATTEMPTS = args.max_failed_attempts
    if args.max_moves is not None:
        MAX_GAME_MOVES = args.max_moves
    if args.reasoning_effort:
        REASONING_EFFORT = args.reasoning_effort
    if args.label:
        RUN_LABEL = args.label
    if args.log_dir:
        LOG_FOLDER = args.log_dir


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    apply_args(args)
    return run_games()


if __name__ == "__main__":
    main()
