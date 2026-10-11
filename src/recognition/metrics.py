"""Metrics for plate reading: exact-match accuracy and character accuracy."""


def edit_distance(a, b):
    """Levenshtein distance between two sequences.

    The minimum number of insertions, deletions and substitutions
    (each costs 1) needed to turn a into b.
    """
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(prev[j] + 1,            # delete x
                           cur[j - 1] + 1,         # insert y
                           prev[j - 1] + (x != y)))  # substitute (free if equal)
        prev = cur
    return prev[-1]


def score(predictions, references):
    """Compare two lists of token-id lists.

    Returns:
        (exact_matches, total_edit_distance, total_reference_tokens)
    """
    exact = edits = total = 0
    for pred, ref in zip(predictions, references):
        exact += pred == ref
        edits += edit_distance(pred, ref)
        total += len(ref)
    return exact, edits, total


if __name__ == "__main__":
    # Quick self-test: python -m src.recognition.metrics
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 3]) == 1        # one deletion
    assert edit_distance([1, 2, 3], [1, 9, 3]) == 1     # one substitution
    assert edit_distance([], [1, 2]) == 2               # two insertions
    assert score([[1, 2], [3, 4]], [[1, 2], [3, 5]]) == (1, 1, 4)
    print("metrics OK")