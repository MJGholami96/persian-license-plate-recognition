"""Character set for Iranian license plates.

A plate label is a list of tokens, not a string, because some letters
(for example "الف") are a single class made of several Unicode characters.
Index 0 is reserved for the CTC blank, so real tokens start at 1.
"""

DIGITS = list("0123456789")
LETTERS = [
    "الف", "ب", "پ", "ت", "ث", "ج", "د", "ز", "ژ", "س", "ش",
    "ص", "ط", "ع", "ق", "ل", "م", "ن", "و", "ه", "ی",
]

# Raw label names in the XML files that must be mapped to a standard token.
NORMALIZE = {
    "ه\u200d": "ه",                        # "ه" followed by a zero-width joiner
    "ژ (معلولین و جانبازان)": "ژ",           # name with an extra description
}

BLANK = 0
TOKENS = DIGITS + LETTERS
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS, start=1)}
ID_TO_TOKEN = {i: token for token, i in TOKEN_TO_ID.items()}
NUM_CLASSES = len(TOKENS) + 1               # real tokens + blank


def normalize(token):
    """Strip whitespace and map known raw names to their standard token."""
    token = token.strip()
    return NORMALIZE.get(token, token)


def encode(tokens):
    """Turn a list of tokens into a list of integer ids."""
    ids = []
    for token in tokens:
        token = normalize(token)
        if token not in TOKEN_TO_ID:
            raise KeyError(f"Unknown token: {token!r}")
        ids.append(TOKEN_TO_ID[token])
    return ids


def decode(ids):
    """Turn a list of integer ids back into a list of tokens (blank is skipped)."""
    return [ID_TO_TOKEN[i] for i in ids if i != BLANK]


if __name__ == "__main__":
    # Quick self-test: python -m src.recognition.charset
    raw = ["9", "5", "ت", "3", "8", "6", "3", "3"]
    ids = encode(raw)
    print("ids:", ids)
    assert decode(ids) == raw

    assert encode(["ه\u200d"]) == encode(["ه"])
    assert encode(["ژ (معلولین و جانبازان)"]) == encode(["ژ"])

    print("number of classes (with blank):", NUM_CLASSES)
    assert NUM_CLASSES == 32
    print("charset OK")