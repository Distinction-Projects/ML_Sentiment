import unittest

import src.app  # noqa: F401
from src.pages.news_stats import _tag_figure


class NewsStatsTagFigureTests(unittest.TestCase):
    def test_tag_figure_excludes_general_variants(self):
        counts = [
            {"tag": "General", "count": 9},
            {"tag": " general ", "count": 8},
            {"tag": "GEN\u200bERAL", "count": 7},
            {"tag": "OpenAI", "count": 6},
            {"tag": "Policy", "count": 5},
        ]

        figure = _tag_figure(counts)
        self.assertEqual(len(figure.data), 1)
        self.assertEqual(list(figure.data[0].y), ["OpenAI", "Policy"])
        self.assertEqual(list(figure.data[0].x), [6, 5])

    def test_tag_figure_returns_empty_when_only_general(self):
        counts = [{"tag": "General", "count": 3}]
        figure = _tag_figure(counts)
        self.assertEqual(len(figure.data), 0)
        self.assertEqual(figure.layout.title.text, "Top Tags")


if __name__ == "__main__":
    unittest.main()
