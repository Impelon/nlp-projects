import unittest

from dataset_loader import merge_empty_pairs


class TestMergeEmptyPairs(unittest.TestCase):

    def assertMergedEqual(self, pairs, reference):
        merged_pairs = merge_empty_pairs(pairs)
        self.assertSequenceEqual(list(merged_pairs), reference)

    def test_no_pairs(self):
        pairs = merge_empty_pairs([])
        with self.assertRaises(StopIteration):
            next(pairs)

    def test_no_empties(self):
        pairs = [("A", "1"), ("B", "2"), ("C", "3"), ("D", "4")]
        self.assertMergedEqual(pairs, pairs)

    def test_no_merge_possible(self):
        pairs = [("A", "")]
        self.assertMergedEqual(pairs, pairs)

    def test_single_empty(self):
        pairs = [("A", "1"), ("", "2"), ("C", "3"), ("D", "4")]
        reference = [("AC", "123"), ("D", "4")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("B", "2"), ("C", ""), ("D", "4")]
        reference = [("A", "1"), ("BCD", "24")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", ""), ("B", "2"), ("C", "3"), ("D", "4")]
        reference = [("AB", "2"), ("C", "3"), ("D", "4")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("B", "2"), ("C", "3"), ("", "4")]
        reference = [("A", "1"), ("B", "2"), ("C", "34")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", ""), ("B", "2")]
        reference = [("B", "2")]
        self.assertMergedEqual(pairs, reference)

    def test_multiple_empty(self):
        pairs = [("", "1"), ("B", "2"), ("C", "3"), ("D", "")]
        reference = [("B", "12"), ("CD", "3")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("B", ""), ("", "3"), ("D", "4")]
        reference = [("ABD", "134")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", "1"), ("", "2"), ("C", "3"), ("D", "4")]
        reference = [("C", "123"), ("D", "4")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", ""), ("", "2"), ("C", ""), ("", "4")]
        reference = [("AC", "24")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", ""), ("B", ""), ("C", ""), ("D", "")]
        reference = [("ABCD", "")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("B", ""), ("", "3"), ("D", "4")]
        reference = [("ABD", "134")]
        self.assertMergedEqual(pairs, reference)

    def test_delete_completely_empty(self):
        pairs = [("", ""), ("", ""), ("", ""), ("", "")]
        reference = []
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("", ""), ("", ""), ("D", "4")]
        reference = [("A", "1"), ("D", "4")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("B", ""), ("", ""), ("D", "4")]
        reference = [("ABD", "14")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("A", "1"), ("", ""), ("C", ""), ("D", "4")]
        reference = [("ACD", "14")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", ""), ("B", "2"), ("C", "3"), ("", "")]
        reference = [("B", "2"), ("C", "3")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", ""), ("", ""), ("C", "3"), ("", "")]
        reference = [("C", "3")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", ""), ("", "2"), ("C", "3"), ("", "")]
        reference = [("C", "23")]
        self.assertMergedEqual(pairs, reference)
        pairs = [("", ""), ("B", "2"), ("C", ""), ("", "")]
        reference = [("BC", "2")]
        self.assertMergedEqual(pairs, reference)


if __name__ == "__main__":
    unittest.main()
