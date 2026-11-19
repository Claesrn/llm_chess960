"""Chess960 (Fischer Random Chess) entry point for the LLM Chess benchmark.

Runs a single game exactly like `llm_chess.run`, but from a Chess960 starting
position. All other settings (player types, models, limits, engine paths, ...)
are taken from the module-level configuration in `llm_chess.py`.

Usage:
    uv run llm-chess960                 # random start position (0-959)
    uv run llm-chess960 --position 42   # specific Scharnagl position number
"""

import argparse
from typing import Any, Dict, Optional, Tuple

import llm_chess


def run(
    log_dir: Optional[str] = "_logs",
    position: Optional[int] = None,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Any, Any]:
    """Run one Chess960 game.

    Args:
        log_dir: Directory for the JSON game log (None disables logging).
        position: Chess960 start position number 0-959 (518 is the standard
            setup). None keeps `llm_chess.chess960_position`, which defaults to
            a random position per game.
        **kwargs: Forwarded to `llm_chess.run` (e.g. llm_config_white).
    """
    if position is not None and not 0 <= position <= 959:
        raise ValueError(f"position must be in 0..959, got {position}")

    previous = (llm_chess.chess960, llm_chess.chess960_position, llm_chess.record_experiment_stats)
    llm_chess.chess960 = True
    llm_chess.record_experiment_stats = True
    if position is not None:
        llm_chess.chess960_position = position
    try:
        return llm_chess.run(log_dir=log_dir, **kwargs)
    finally:
        llm_chess.chess960, llm_chess.chess960_position, llm_chess.record_experiment_stats = previous


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm-chess960",
        description="Play one Chess960 (Fischer Random) game using the settings in llm_chess.py.",
    )
    parser.add_argument(
        "--position",
        type=int,
        default=None,
        help="Chess960 start position number 0-959 (default: random).",
    )
    parser.add_argument(
        "--log-dir",
        default="_logs",
        help="Directory for the JSON game log (default: _logs).",
    )
    args = parser.parse_args(argv)
    if args.position is not None and not 0 <= args.position <= 959:
        parser.error("--position must be between 0 and 959")
    run(log_dir=args.log_dir, position=args.position)


if __name__ == "__main__":
    main()
