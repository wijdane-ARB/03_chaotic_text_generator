"""Lab 03, Part 2 - the same HMM and Viterbi decoding on REAL text.   Run:  python -m src.evaluate_real

Data: Universal Dependencies English EWT (web text, 17 universal part-of-speech tags, licence CC BY-SA 4.0).
The three files are downloaded once into data/ewt/ from the treebank's GitHub repository.
The model code is the one used in the main lab (src/pipeline.py); only the data changes.
"""
import json
import os
import urllib.request
from collections import Counter, defaultdict

from src.pipeline import HMM

BASE = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/"
FILES = {"train": "en_ewt-ud-train.conllu", "test": "en_ewt-ud-test.conllu"}
DIR = "data/ewt"


def download():
    os.makedirs(DIR, exist_ok=True)
    for name in FILES.values():
        path = os.path.join(DIR, name)
        if not os.path.exists(path):
            print("downloading", name)
            urllib.request.urlretrieve(BASE + name, path)


def load_conllu(path):
    """Return [(words, upos_tags)], words lower-cased. Multi-word tokens and empty nodes are skipped."""
    sents, words, tags = [], [], []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            if words:
                sents.append((words, tags)); words, tags = [], []
        elif not line.startswith("#"):
            c = line.split("\t")
            if "-" not in c[0] and "." not in c[0]:
                words.append(c[1].lower()); tags.append(c[3])
    if words:
        sents.append((words, tags))
    return sents


def score(test, predict, vocab):
    n = ok = known = known_ok = unk = unk_ok = sent_ok = 0
    conf = Counter()
    for words, gold in test:
        pred = predict(words)
        all_ok = True
        for w, g, p in zip(words, gold, pred):
            n += 1; ok += g == p; all_ok &= g == p
            if w in vocab:
                known += 1; known_ok += g == p
            else:
                unk += 1; unk_ok += g == p
            if g != p:
                conf[(g, p)] += 1
        sent_ok += all_ok
    return {"accuracy": ok / n, "known_words": known_ok / known, "unknown_words": unk_ok / unk,
            "unknown_share_of_tokens": unk / n, "whole_sentence": sent_ok / len(test)}, conf


def main():
    download()
    train, test = (load_conllu(os.path.join(DIR, FILES[k])) for k in ("train", "test"))
    print(f"train: {len(train)} sentences, test: {len(test)} sentences")
    patterns = {" ".join(t) for _, t in train}
    print(f"distinct tag sequences in the training set: {len(patterns)} (synthetic corpus: 16)")

    hmm = HMM(train)
    print(f"tags: {len(hmm.tags)}, vocabulary: {len(hmm.vocab)} words (synthetic corpus: 6 tags, 61 words)\n")

    most_freq = defaultdict(Counter)                   # baseline: most frequent tag of each word
    for words, tags in train:
        for w, t in zip(words, tags):
            most_freq[w][t] += 1
    overall = Counter(t for _, tags in train for t in tags).most_common(1)[0][0]
    baseline = lambda ws: [most_freq[w].most_common(1)[0][0] if w in most_freq else overall for w in ws]

    results = {}
    print(f"{'model':<20}{'accuracy':>9}{'known':>9}{'unknown':>9}{'sentence':>10}")
    for name, fn in (("most-frequent tag", baseline), ("HMM + Viterbi", hmm.viterbi)):
        r, conf = score(test, fn, hmm.vocab)
        results[name] = r
        print(f"{name:<20}{r['accuracy']:>9.1%}{r['known_words']:>9.1%}{r['unknown_words']:>9.1%}{r['whole_sentence']:>10.1%}")
    unk = results["HMM + Viterbi"]["unknown_share_of_tokens"]
    print(f"\nunknown words (not in the training vocabulary) are {unk:.1%} of the test tokens")
    print("\nmost frequent HMM errors (gold -> predicted):")
    for (g, p), c in conf.most_common(8):
        print(f"  {g:>6} -> {p:<6} {c}")
    os.makedirs("results", exist_ok=True)
    json.dump({"models": results, "confusions": [[g, p, c] for (g, p), c in conf.most_common(20)]},
              open("results/ewt_results.json", "w"), indent=2)
    print("\nSaved results/ewt_results.json")


if __name__ == "__main__":
    main()
