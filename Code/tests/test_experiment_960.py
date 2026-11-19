import json
import os
import shutil
import tempfile
import types
import unittest

import chess
import requests

from .helper import _MockServerTestCaseBase, get_mock_base_url

import analyze_960
import llm_chess
import run_multiple_games_960 as rmg960
from llm_chess import PlayerType, TerminationReason

def snapshot_llm_chess_globals():
    """All plain module-level settings of llm_chess (run_games() changes many of them)."""
    return {
        k: v for k, v in vars(llm_chess).items()
        if not k.startswith("_") and not callable(v) and not isinstance(v, types.ModuleType)
    }


class _GlobalsMixin:
    def save_globals(self):
        self._saved = snapshot_llm_chess_globals()
        llm_chess.record_experiment_stats = True
        llm_chess.throttle_delay = 0
        llm_chess.dialog_turn_delay = 0
        llm_chess.random_print_board = False

    def restore_globals(self):
        for k, v in self._saved.items():
            setattr(llm_chess, k, v)
        llm_chess._setup_board()


class TestDescribeIllegalMove(unittest.TestCase):
    def setUp(self):
        self._saved = (llm_chess.chess960, llm_chess.chess960_position)
        llm_chess.chess960 = True
        llm_chess.chess960_position = 518
        llm_chess._setup_board()

    def tearDown(self):
        llm_chess.chess960, llm_chess.chess960_position = self._saved
        llm_chess._setup_board()

    def category(self, move):
        return llm_chess.describe_illegal_move(move)["category"]

    def test_categories(self):
        self.assertEqual(self.category("e2e5"), "piece_movement")
        self.assertEqual(self.category("e1g1"), "castling")
        self.assertEqual(self.category("e1h1"), "castling")
        self.assertEqual(self.category("e3e4"), "wrong_piece_on_square")
        self.assertEqual(self.category("e7e5"), "wrong_piece_on_square")
        self.assertEqual(self.category("zz99"), "unparseable")

    def test_leaves_king_in_check_and_promotion(self):
        llm_chess.board.set_fen("4k3/P7/8/8/8/8/3r4/4K2R w K - 0 1")
        llm_chess.board.chess960 = True
        self.assertEqual(self.category("e1d1"), "leaves_king_in_check")
        self.assertEqual(self.category("a7a8"), "missing_promotion")

    def test_record_contains_position(self):
        info = llm_chess.describe_illegal_move("e2e5")
        self.assertEqual(info["ply"], 1)
        self.assertEqual(info["side"], "white")
        self.assertEqual(info["fen"], llm_chess.board.fen())


class TestRandomPlayerKeepsLegalMovesInB2(_GlobalsMixin, unittest.TestCase):
    def setUp(self):
        self.save_globals()

    def tearDown(self):
        self.restore_globals()

    def test_random_vs_random_without_legal_moves(self):
        llm_chess.white_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.black_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.enable_legal_moves = False
        llm_chess.chess960 = True
        llm_chess.chess960_position = 0
        llm_chess.max_game_moves = 10
        stats, _, _ = llm_chess.run(log_dir=None)
        self.assertEqual(stats["reason"], TerminationReason.MAX_MOVES.value)
        self.assertEqual(stats["player_white"]["wrong_actions"], 0)
        self.assertFalse(stats["settings"]["legal_moves_enabled"])
        self.assertEqual(stats["outcome"]["loss_cause"], "max_moves_draw")


class TestLLMConditions(_GlobalsMixin, _MockServerTestCaseBase):
    def setUp(self):
        requests.post(f"{get_mock_base_url()}/reset", json={"scenarioType": "default", "useThinking": False}, timeout=10)
        self.save_globals()
        llm_chess.white_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.black_player_type = PlayerType.LLM_BLACK
        llm_chess.chess960 = True
        llm_chess.chess960_position = 42
        llm_chess.max_game_moves = 6

    def tearDown(self):
        self.restore_globals()

    def test_b1_with_comprehension_check(self):
        llm_chess.comprehension_check = True
        stats, _, _ = llm_chess.run(log_dir=None)
        comp = stats["comprehension"]["black"]
        self.assertEqual(comp["expected"], chess.Board.from_chess960_pos(42).fen().split("/")[0].upper())
        self.assertFalse(comp["correct"])  # the mock does not answer the question
        self.assertNotIn("white", stats["comprehension"])  # random player is not asked
        black = stats["player_black"]
        self.assertGreater(black["turns"], 0)
        self.assertEqual(black["illegal_attempts"], [])
        self.assertEqual(stats["reason"], TerminationReason.MAX_MOVES.value)

    def test_b2_rejects_get_legal_moves(self):
        llm_chess.enable_legal_moves = False
        stats, _, _ = llm_chess.run(log_dir=None)
        black = stats["player_black"]
        self.assertGreater(black["disallowed_action_requests"], 0)
        self.assertEqual(stats["reason"], TerminationReason.TOO_MANY_WRONG_ACTIONS.value)
        self.assertEqual(stats["outcome"], {"loss_cause": "format_forfeit", "losing_side": "black"})

    def _prompts_sent_to(self, player):
        return [m.get("content", "") for msgs in player.chat_messages.values() for m in msgs]

    def test_no_variant_hint_hides_chess960(self):
        llm_chess.chess960_prompt_hint = False
        stats, _, player_black = llm_chess.run(log_dir=None)
        prompts = self._prompts_sent_to(player_black)
        self.assertTrue(prompts)
        self.assertFalse(any("Chess960" in p for p in prompts))
        self.assertFalse(stats["settings"]["chess960_prompt_hint"])
        self.assertTrue(stats["settings"]["chess960"])  # rules/board are still Chess960

    def test_variant_hint_shown_by_default(self):
        stats, _, player_black = llm_chess.run(log_dir=None)
        self.assertTrue(any("Chess960" in p for p in self._prompts_sent_to(player_black)))
        self.assertTrue(stats["settings"]["chess960_prompt_hint"])

    def test_illegal_move_logged(self):
        requests.post(f"{get_mock_base_url()}/reset", json={"scenarioType": "invalid_action", "useThinking": False}, timeout=10)
        llm_chess.max_game_moves = 4
        stats, _, _ = llm_chess.run(log_dir=None)
        attempts = stats["player_black"]["illegal_attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["move"], "d4d5")
        self.assertEqual(attempts[0]["side"], "black")
        self.assertEqual(stats["player_black"]["first_attempt_illegal"], 1)


class TestRunnerCliAndAnalysis(_GlobalsMixin, unittest.TestCase):
    def setUp(self):
        self.save_globals()
        self._saved_module = {k: getattr(rmg960, k) for k in dir(rmg960) if k.isupper()}
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        for k, v in self._saved_module.items():
            setattr(rmg960, k, v)
        self.restore_globals()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cli_flags(self):
        args = rmg960.build_arg_parser().parse_args(
            ["--games", "60", "--positions", "a2", "--no-legal-moves", "--opponent", "dragon", "--level", "3",
             "--comprehension-check", "--max-failed-attempts", "5"]
        )
        rmg960.apply_args(args)
        self.assertEqual(rmg960.NUM_REPETITIONS, 60)
        self.assertEqual(rmg960.CHESS960_POSITIONS, rmg960.A2_POSITIONS)
        self.assertFalse(rmg960.LEGAL_MOVES)
        self.assertEqual(rmg960.WHITE_PLAYER_TYPE, PlayerType.CHESS_ENGINE_DRAGON)
        self.assertEqual(rmg960.ENGINE_LEVEL, 3)
        self.assertEqual(rmg960.MAX_FAILED_ATTEMPTS, 5)
        self.assertEqual(rmg960.condition_label(), "A2-30pos_B2-nolegal_cc")

    def test_no_variant_hint_flag_and_label(self):
        rmg960.apply_args(rmg960.build_arg_parser().parse_args(["--sample", "10", "--no-variant-hint"]))
        self.assertFalse(rmg960.VARIANT_HINT)
        self.assertEqual(rmg960.condition_label(), "A2-10pos-nohint_B1-legal")
        game = {"settings": {"chess960": True, "chess960_position": 45, "chess960_prompt_hint": False,
                             "legal_moves_enabled": True, "white_player_type": "RANDOM_PLAYER",
                             "black_player_type": "LLM_BLACK"}}
        self.assertEqual(analyze_960.condition_of(game, "black")["start"], "A2-chess960-uninformed")

    def test_a1_label_and_sample(self):
        rmg960.apply_args(rmg960.build_arg_parser().parse_args(["--positions", "518"]))
        self.assertEqual(rmg960.condition_label(), "A1-pos518_B1-legal")
        rmg960.CHESS960_POSITIONS = None
        rmg960.apply_args(rmg960.build_arg_parser().parse_args(["--sample", "20", "--seed", "1"]))
        self.assertEqual(len(set(rmg960.CHESS960_POSITIONS)), 20)
        self.assertNotIn(518, rmg960.CHESS960_POSITIONS)

    def test_a2_list_is_fixed(self):
        self.assertEqual(len(rmg960.A2_POSITIONS), 30)
        self.assertEqual(len(set(rmg960.A2_POSITIONS)), 30)
        self.assertNotIn(518, rmg960.A2_POSITIONS)

    def test_analysis_on_runner_logs(self):
        # Random vs random game logs, then mark black as an "LLM" so the analysis has a side to score.
        rmg960.apply_args(rmg960.build_arg_parser().parse_args(
            ["--games", "2", "--positions", "0,42", "--player", "random", "--max-moves", "12",
             "--log-dir", self.tmp]
        ))
        rmg960.THROTTLE_DELAY = 0
        rmg960.DIALOG_TURN_DELAY = 0
        agg = rmg960.run_games()
        self.assertEqual(agg["total_games"], 2)
        self.assertIn("loss_causes", agg)

        game_files = [f for f in os.listdir(self.tmp) if f.endswith(".json") and not f.startswith("_")]
        self.assertEqual(len(game_files), 2)
        for name in game_files:
            path = os.path.join(self.tmp, name)
            with open(path) as f:
                data = json.load(f)
            data["settings"]["black_player_type"] = "LLM_BLACK"
            data["player_black"]["model"] = "fake-model"
            with open(path, "w") as f:
                json.dump(data, f)

        out = os.path.join(self.tmp, "analysis")
        sf = analyze_960.find_stockfish(os.environ.get("STOCKFISH_PATH"))
        argv = [self.tmp, "--out", out, "--depth", "6"] + ([] if sf else ["--no-engine"])
        if sf:
            argv += ["--stockfish", sf]
        self.assertEqual(analyze_960.main(argv), 0)
        with open(os.path.join(out, "conditions.csv")) as f:
            text = f.read()
        self.assertIn("A2-chess960", text)
        self.assertIn("C1-random", text)
        with open(os.path.join(out, "games.csv")) as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(len(lines), 3)  # header + 2 games
        if sf:
            with open(os.path.join(out, "moves.csv")) as f:
                header = f.readline()
            self.assertIn("cp_loss", header)
            self.assertIn("phase", header)

    def test_engine_depth_flag(self):
        rmg960.apply_args(rmg960.build_arg_parser().parse_args(
            ["--positions", "518", "--opponent", "stockfish", "--level", "0", "--engine-depth", "1"]))
        self.assertEqual(rmg960.ENGINE_DEPTH, 1)
        self.assertEqual(rmg960.condition_label(), "A1-pos518_B1-legal_depth1")
        game = {"settings": {"chess960": True, "chess960_position": 518, "legal_moves_enabled": True,
                             "white_player_type": "CHESS_ENGINE_STOCKFISH", "black_player_type": "LLM_BLACK",
                             "stockfish_level": 0, "engine_depth": 1}}
        self.assertEqual(analyze_960.condition_of(game, "black")["opponent"], "C2-stockfish-lvl0-depth1")

    def test_engine_depth_game(self):
        sf = analyze_960.find_stockfish(os.environ.get("STOCKFISH_PATH"))
        if sf is None:
            self.skipTest("Stockfish not installed")
        llm_chess.white_player_type = PlayerType.CHESS_ENGINE_STOCKFISH
        llm_chess.black_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.stockfish_path = sf
        llm_chess.stockfish_level = 0
        llm_chess.engine_depth = 1
        llm_chess.chess960 = True
        llm_chess.chess960_position = 518
        llm_chess.max_game_moves = 10
        stats, white, _ = llm_chess.run(log_dir=None)
        self.assertEqual(white.depth, 1)
        self.assertEqual(stats["settings"]["engine_depth"], 1)
        self.assertEqual(stats["player_white"]["wrong_actions"], 0)
        self.assertEqual(stats["reason"], TerminationReason.MAX_MOVES.value)

    def test_read_timeout_is_retryable(self):
        from custom_agents import is_retryable_error
        self.assertTrue(is_retryable_error(Exception("The read operation timed out")))
        self.assertTrue(is_retryable_error(Exception("Error code: 529 - overloaded_error")))
        self.assertFalse(is_retryable_error(Exception("invalid_request_error: bad parameter")))

    def test_standard_pattern_classification(self):
        # Position #3 has a rook on g8; g8f6 is a knight move from the standard knight square -> phantom piece
        start = chess.Board.from_chess960_pos(3)
        self.assertEqual(analyze_960.back_rank_letters(start, chess.BLACK)[6], "R")
        board = start.copy(); board.push_uci("e2e4")
        sp = analyze_960.classify_standard_pattern({"move": "g8f6", "fen": board.fen()}, start.fen(), True, 3)
        self.assertTrue(sp["phantom_standard"])
        self.assertTrue(0 < sp["phantom_chance"] < 1)
        # A move that only fits the real piece is not a phantom move
        sp = analyze_960.classify_standard_pattern({"move": "g8g1", "fen": board.fen()}, start.fen(), True, 3)
        self.assertFalse(sp["phantom_standard"])
        # Standard castling notation is flagged in any condition; phantom test is undefined at #518
        sp = analyze_960.classify_standard_pattern({"move": "e8g8", "fen": chess.Board().fen()}, chess.STARTING_FEN, True, 518)
        self.assertTrue(sp["standard_castling_notation"])
        self.assertIsNone(sp["phantom_standard"])

    def test_performance_elo(self):
        rows = [{"dragon_level": 3, "score": s} for s in (1.0, 0.0, 0.5, 0.0)]
        elo, moe = analyze_960.performance_elo(rows)
        self.assertIsNotNone(elo)
        self.assertLess(elo, 125 * 4 + 35)  # scored below 50% => below the anchor
        self.assertEqual(analyze_960.performance_elo([{"dragon_level": 3, "score": 1.0}]), (None, None))


if __name__ == "__main__":
    unittest.main()
