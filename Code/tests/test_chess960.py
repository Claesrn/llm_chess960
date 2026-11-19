import unittest

import chess

from . import helper  # noqa: F401  (sets mock LLM env vars before importing llm_chess)

import llm_chess
import llm_chess960
from llm_chess import PlayerType, TerminationReason


class TestChess960(unittest.TestCase):
    def setUp(self):
        self._saved = (
            llm_chess.white_player_type,
            llm_chess.black_player_type,
            llm_chess.max_game_moves,
            llm_chess.throttle_delay,
            llm_chess.dialog_turn_delay,
            llm_chess.random_print_board,
            llm_chess.chess960,
            llm_chess.chess960_position,
        )
        llm_chess.white_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.black_player_type = PlayerType.RANDOM_PLAYER
        llm_chess.max_game_moves = 20
        llm_chess.throttle_delay = 0
        llm_chess.dialog_turn_delay = 0
        llm_chess.random_print_board = False

    def tearDown(self):
        (
            llm_chess.white_player_type,
            llm_chess.black_player_type,
            llm_chess.max_game_moves,
            llm_chess.throttle_delay,
            llm_chess.dialog_turn_delay,
            llm_chess.random_print_board,
            llm_chess.chess960,
            llm_chess.chess960_position,
        ) = self._saved
        llm_chess.chess960 = False
        llm_chess._setup_board()

    def test_fixed_position_game(self):
        game_stats, _, _ = llm_chess960.run(log_dir=None, position=0)

        expected_fen = chess.Board.from_chess960_pos(0).fen()
        self.assertEqual(game_stats["variant"], "chess960")
        self.assertEqual(game_stats["chess960_position"], 0)
        self.assertEqual(game_stats["starting_fen"], expected_fen)
        self.assertIn('[Variant "Chess960"]', game_stats["pgn"])
        self.assertIn('[SetUp "1"]', game_stats["pgn"])
        self.assertIn(f'[FEN "{expected_fen}"]', game_stats["pgn"])
        self.assertEqual(game_stats["player_white"]["wrong_moves"], 0)
        self.assertEqual(game_stats["player_black"]["wrong_moves"], 0)
        self.assertEqual(game_stats["reason"], TerminationReason.MAX_MOVES.value)

        # Wrapper restores the global flag afterwards
        self.assertFalse(llm_chess.chess960)

    def test_random_position_is_in_range(self):
        game_stats, _, _ = llm_chess960.run(log_dir=None)
        self.assertTrue(0 <= game_stats["chess960_position"] <= 959)

    def test_invalid_position_rejected(self):
        with self.assertRaises(ValueError):
            llm_chess960.run(log_dir=None, position=960)

    def test_castling_uses_king_takes_rook_uci(self):
        llm_chess.chess960 = True
        llm_chess.chess960_position = 518  # standard setup, but in 960 mode
        llm_chess._setup_board()
        llm_chess.san_moves.clear()
        for mv in ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]:
            llm_chess.make_move(mv)
        self.assertIn("e1h1", llm_chess.get_legal_moves().split(","))
        llm_chess.make_move("e1h1")
        self.assertEqual(llm_chess.san_moves[-1], "O-O")
        self.assertEqual(llm_chess.board.piece_at(chess.G1), chess.Piece(chess.KING, chess.WHITE))
        self.assertEqual(llm_chess.board.piece_at(chess.F1), chess.Piece(chess.ROOK, chess.WHITE))

    def test_pgn_replays_from_fen(self):
        import io
        import chess.pgn

        game_stats, _, _ = llm_chess960.run(log_dir=None, position=123)
        game = chess.pgn.read_game(io.StringIO(game_stats["pgn"]))
        self.assertTrue(game.board().chess960)
        self.assertEqual(game.board().fen(), game_stats["starting_fen"])
        self.assertEqual(len(game.errors), 0)

    def test_standard_mode_unchanged(self):
        llm_chess.chess960 = False
        game_stats, _, _ = llm_chess.run(log_dir=None)
        self.assertNotIn("variant", game_stats)
        self.assertNotIn("[Variant", game_stats["pgn"])
        self.assertIn('[Event "Chess Game"]', game_stats["pgn"])


if __name__ == "__main__":
    unittest.main()
