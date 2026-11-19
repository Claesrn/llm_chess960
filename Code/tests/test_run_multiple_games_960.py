import json
import os
import tempfile
import unittest

from . import helper  # noqa: F401  (sets mock LLM env vars)

import llm_chess
import run_multiple_games_960 as rmg960
from .test_experiment_960 import snapshot_llm_chess_globals


class TestRunMultipleGames960(unittest.TestCase):
    def setUp(self):
        self._saved_module = {k: getattr(rmg960, k) for k in (
            "NUM_REPETITIONS", "WHITE_PLAYER_TYPE", "BLACK_PLAYER_TYPE", "LOG_FOLDER",
            "THROTTLE_DELAY", "DIALOG_TURN_DELAY", "CHESS960", "CHESS960_POSITIONS", "SEED",
        )}
        self._saved_llm_chess = snapshot_llm_chess_globals()
        self.tmp = tempfile.mkdtemp()
        rmg960.NUM_REPETITIONS = 3
        rmg960.WHITE_PLAYER_TYPE = llm_chess.PlayerType.RANDOM_PLAYER
        rmg960.BLACK_PLAYER_TYPE = llm_chess.PlayerType.RANDOM_PLAYER
        rmg960.THROTTLE_DELAY = 0
        rmg960.DIALOG_TURN_DELAY = 0
        rmg960.LOG_FOLDER = self.tmp
        llm_chess.max_game_moves = 10

    def tearDown(self):
        for k, v in self._saved_module.items():
            setattr(rmg960, k, v)
        for k, v in self._saved_llm_chess.items():
            setattr(llm_chess, k, v)
        llm_chess._setup_board()

    def test_fixed_positions(self):
        rmg960.CHESS960_POSITIONS = [0, 42]
        agg = rmg960.run_games()
        self.assertEqual(agg["variant"], "chess960")
        self.assertEqual(agg["total_games"], 3)
        self.assertEqual([g["chess960_position"] for g in agg["games"]], [0, 42, 0])
        with open(os.path.join(self.tmp, "_aggregate_results.json")) as f:
            self.assertEqual(json.load(f)["total_games"], 3)
        game_logs = [n for n in os.listdir(self.tmp) if n.endswith(".json") and not n.startswith("_")]
        self.assertEqual(len(game_logs), 3)

    def test_random_positions_reproducible_and_skip_518(self):
        rmg960.CHESS960_POSITIONS = None
        rmg960.SEED = 123
        a = rmg960.pick_positions(200)
        b = rmg960.pick_positions(200)
        self.assertEqual(a, b)
        self.assertNotIn(518, a)
        self.assertTrue(all(0 <= p <= 959 for p in a))

    def test_standard_baseline(self):
        rmg960.CHESS960 = False
        agg = rmg960.run_games()
        self.assertEqual(agg["variant"], "standard")
        self.assertTrue(all(g["chess960_position"] is None for g in agg["games"]))


if __name__ == "__main__":
    unittest.main()
