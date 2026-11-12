"""Run a batch of Chess960 (Fischer Random) games and aggregate the results.

Usage:
    uv run python run_multiple_games_960.py

Edit the launch settings below for a new experiment. Models/keys come from .env
(MODEL_KIND_W / MODEL_KIND_B etc.), exactly as for run_multiple_games.py.

Logs go to _logs/chess960/<category>/<model>/<timestamp>/:
  - one <time>.json per game (includes chess960_position and starting_fen)
  - _aggregate_results.json with totals plus the list of start positions used
  - _run.json with run metadata, output.txt with the console output

Set CHESS960 = False to run the *same* setup on standard chess (logged under
_logs/chess960_baseline/...), which gives a like-for-like baseline for comparison.
"""

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

# Start positions (Scharnagl numbers 0-959):
# - None: a random position per game (see SEED / EXCLUDE_STANDARD_POSITION)
# - list of ints: used in order, cycling if NUM_REPETITIONS is larger
CHESS960_POSITIONS: Optional[List[int]] = None
SEED: Optional[int] = 960  # makes the random positions reproducible; None = different every run
EXCLUDE_STANDARD_POSITION = True  # never draw #518 (the normal chess setup) at random

WHITE_PLAYER_TYPE = llm_chess.PlayerType.RANDOM_PLAYER
BLACK_PLAYER_TYPE = llm_chess.PlayerType.LLM_BLACK
ENGINE_LEVEL = 1  # used only if a side is Stockfish/Dragon

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


def pick_positions(n: int) -> List[int]:
    """Return the start position for each of the n games."""
    if CHESS960_POSITIONS:
        for p in CHESS960_POSITIONS:
            if not 0 <= p <= 959:
                raise ValueError(f"Invalid Chess960 position {p}; must be 0-959")
        return [CHESS960_POSITIONS[i % len(CHESS960_POSITIONS)] for i in range(n)]

    rng = random.Random(SEED)
    pool = [p for p in range(960) if not (EXCLUDE_STANDARD_POSITION and p == 518)]
    return [rng.choice(pool) for _ in range(n)]


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
    llm_chess.white_player_type = WHITE_PLAYER_TYPE
    llm_chess.black_player_type = BLACK_PLAYER_TYPE
    llm_chess.stockfish_level = ENGINE_LEVEL
    llm_chess.dragon_level = ENGINE_LEVEL
    llm_chess.throttle_delay = THROTTLE_DELAY
    llm_chess.dialog_turn_delay = DIALOG_TURN_DELAY
    llm_chess.max_api_retries = MAX_API_RETRIES
    llm_chess.api_retry_delay = API_RETRY_DELAY
    llm_chess.remove_text = llm_chess.DEFAULT_REMOVE_TEXT_REGEX
    llm_chess.chess960 = CHESS960

    if LOG_FOLDER is None:
        base = build_log_folder(
            white_player_type=WHITE_PLAYER_TYPE,
            black_player_type=BLACK_PLAYER_TYPE,
            llm_config_white=llm_config_white,
            llm_config_black=llm_config_black,
            stockfish_level=ENGINE_LEVEL,
            dragon_level=ENGINE_LEVEL,
        )
        prefix = "_logs/chess960/" if CHESS960 else "_logs/chess960_baseline/"
        LOG_FOLDER = base.replace("_logs/", prefix, 1)

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

            _update_side(aggregate, "player_white", player_white, game_stats)
            _update_side(aggregate, "player_black", player_black, game_stats)

            aggregate["games"].append(
                {
                    "game": i,
                    "chess960_position": game_stats.get("chess960_position"),
                    "starting_fen": game_stats.get("starting_fen", llm_chess.starting_fen),
                    "result": result,
                    "reason": reason,
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


if __name__ == "__main__":
    run_games()
