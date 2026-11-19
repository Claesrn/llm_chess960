import unittest

from custom_agents import normalize_tool_call_syntax


class TestNormalizeToolCallSyntax(unittest.TestCase):
    def test_plain_text_unchanged(self):
        self.assertEqual(normalize_tool_call_syntax("make_move e2e4"), "make_move e2e4")

    def test_make_move_invoke(self):
        text = '<invoke name="make_move">\n<parameter name="move">d7d5</parameter>\n</invoke>'
        self.assertEqual(normalize_tool_call_syntax(text), "make_move d7d5")

    def test_no_param_invoke(self):
        text = '<invoke name="get_legal_moves">\n</invoke>'
        self.assertEqual(normalize_tool_call_syntax(text), "get_legal_moves")

    def test_only_first_invoke_used(self):
        text = (
            '<invoke name="make_move"><parameter name="move">d7d5</parameter></invoke>\n'
            '<invoke name="get_current_board"></invoke>'
        )
        self.assertEqual(normalize_tool_call_syntax(text), "make_move d7d5")

    def test_surrounding_text_ignored(self):
        text = 'I will push the pawn.\n<invoke name="make_move"><parameter name="move">e7e5</parameter></invoke>'
        self.assertEqual(normalize_tool_call_syntax(text), "make_move e7e5")


if __name__ == "__main__":
    unittest.main()
