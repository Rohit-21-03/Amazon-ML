from collections import Counter


def top_tokens(series, top_n=100):

    counter = Counter()

    for text in series.fillna("").astype(str):

        tokens = text.lower().split()

        counter.update(tokens)

    return counter.most_common(top_n)