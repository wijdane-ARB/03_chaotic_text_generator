"""Lab 03 - evaluation.   Run:  python -m src.evaluate"""
import json
import os
import random

from src.pipeline import (HMM, NGramGenerator, load_tagged, split_by_document, generation_metrics, oov_rate)

N = 500
TEMPS = [0.5, 1.0, 1.5, 2.5]


def main():
    docs = load_tagged()
    train, test = split_by_document(docs)
    print(f"train: {len(train)} sentences, test: {len(test)} sentences (split by document)")
    hmm = HMM(train)                              # plain POS states
    hmm2 = HMM(train, split_states=True)          # states split by role (S / O / P)
    ng = NGramGenerator([w for w, _ in train])

    # ---- 1. Viterbi tagging accuracy on the held-out documents
    acc = {}
    for name, model in (("plain states", hmm), ("role-split states", hmm2)):
        correct = total = w_ok = w_tot = 0
        for words, gold in test:
            pred = model.viterbi(words)
            for w, g, p in zip(words, gold, pred):
                total += 1; correct += g == p
                if w == "watch":
                    w_tot += 1; w_ok += g == p
        acc[name] = (correct / total, w_ok, w_tot)
        print(f"Viterbi POS-tagging accuracy, {name:<18}: {correct / total:.1%} "
              f"| ambiguous word 'watch' (NOUN or VERB): {w_ok}/{w_tot}")
    correct, total = None, None
    w_ok, w_tot = acc["plain states"][1:]

    # ---- 2. generation quality, HMM vs trigram, several temperatures
    results = {"hmm": {}, "hmm_split": {}, "ngram": {}}
    print(f"\n{N} sentences per setting.  columns: valid-tag-pattern | plausible subject-verb | novel | distinct-2")
    for T in TEMPS:
        rng = random.Random(42)
        hs = [hmm.sample_sentence(rng, T) for _ in range(N)]
        rng = random.Random(42)
        h2 = [hmm2.sample_sentence(rng, T) for _ in range(N)]
        rng = random.Random(42)
        ns = [(ng.sample_sentence(rng, T), None) for _ in range(N)]
        results["hmm"][T] = generation_metrics(hs, train, hmm)
        results["hmm_split"][T] = generation_metrics(h2, train, hmm)
        results["ngram"][T] = generation_metrics(ns, train, hmm2)
        oov = sum(oov_rate(w, hmm.vocab) for w, _ in hs + h2 + ns) / (3 * N)
        for name in ("hmm", "hmm_split", "ngram"):
            m = results[name][T]
            print(f"  T={T:<3} {name:<10} {m['valid_tag_pattern']:>6.1%} {m['plausible_subject_verb']:>9.1%} "
                  f"{m['novel_sentences']:>9.1%} {m['distinct_2']:>8.3f}")
    print(f"  out-of-vocabulary rate of generated words: {oov:.1%} (by construction: both models only emit training words)")

    # ---- 3. failure examples to discuss
    rng = random.Random(3)
    odd = []
    pairs = {(n, v) for w, t in train for n, v in [(next((x for x, y in zip(w, t) if y == "NOUN"), None),
                                                  next((x for x, y in zip(w, t) if y == "VERB"), None))]}
    while len(odd) < 4:
        words, tags = hmm2.sample_sentence(rng, 1.0)
        n = next((x for x, y in zip(words, tags) if y == "NOUN"), None)
        v = next((x for x, y in zip(words, tags) if y == "VERB"), None)
        if (n, v) not in pairs:
            odd.append(" ".join(words))
    rng = random.Random(3)
    copies = [" ".join(s) for s in (ng.sample_sentence(rng, 0.5) for _ in range(300))
              if " ".join(s) in {" ".join(w) for w, _ in train}][:2]
    os.makedirs("results", exist_ok=True)
    lines = ["Role-split HMM sentences: grammatical pattern, odd subject-verb pair (states are not meaning):"]
    lines += [f"  - {s}" for s in odd]
    lines += ["n-gram (T=0.5) sentences copied verbatim from the training set (memorisation):"]
    lines += [f"  - {s}" for s in copies]
    open("results/failures.txt", "w", encoding="utf-8").write("\n".join(lines))
    print("\n" + "\n".join(lines))

    json.dump({"viterbi_acc": {k: v[0] for k, v in acc.items()}, "watch": [w_ok, w_tot],
               "generation": {k: {str(t): v for t, v in d.items()} for k, d in results.items()}},
              open("results/metrics.json", "w"), indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    tags, mat = hmm.transition_matrix()           # plain POS states (the readable one)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    im = axes[0].imshow(np.array(mat), cmap="Greys", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(tags))); axes[0].set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(tags))); axes[0].set_yticklabels(tags, fontsize=8)
    for i in range(len(tags)):
        for j in range(len(tags)):
            if mat[i][j] > 0.005:
                axes[0].text(j, i, f"{mat[i][j]:.2f}", ha="center", va="center", fontsize=7,
                             color="white" if mat[i][j] > 0.6 else "black")
    axes[0].set_title("Learned transitions P(next tag | tag)"); axes[0].set_xlabel("next tag"); axes[0].set_ylabel("current tag")
    xs = np.arange(len(TEMPS))
    axes[1].bar(xs - 0.27, [results["hmm"][t]["valid_tag_pattern"] for t in TEMPS], 0.27, color="#8a8a8a", label="POS-HMM (plain states)")
    axes[1].bar(xs, [results["hmm_split"][t]["valid_tag_pattern"] for t in TEMPS], 0.27, color="#7a7a7a", label="POS-HMM (role-split states)")
    axes[1].bar(xs + 0.27, [results["ngram"][t]["valid_tag_pattern"] for t in TEMPS], 0.27, color="#333333", label="trigram")
    axes[1].set_xticks(xs); axes[1].set_xticklabels([f"T={t}" for t in TEMPS]); axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("share of sentences with a valid\ngrammatical pattern"); axes[1].legend(fontsize=7, loc="upper right")
    axes[1].set_title("Richer hidden states restore sentence structure")
    fig.tight_layout(); fig.savefig("results/hmm_results.png", dpi=160)
    print("\nSaved results/hmm_results.png, results/metrics.json, results/failures.txt")


if __name__ == "__main__":
    main()
