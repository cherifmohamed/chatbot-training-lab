import re

def tokenize(text, word2idx):
    words = re.findall(r"\\w+|[^\\w\\s]", text.lower(), re.UNICODE)
    return [word2idx.get(w, word2idx["<pad>"]) for w in words]
