"""Lab 03 - Chaotic Text Generator: n-gram sampling vs a supervised POS-HMM (no UI code here).

HMM = hidden states (part-of-speech tags) + transition probabilities P(tag_t | tag_{t-1})
      + emission probabilities P(word_t | tag_t).
Generation: sample a hidden tag sequence first, then sample one word per tag.
Tagging  : Viterbi finds the most probable tag sequence for a given sentence.
"""
import csv
import math
import random
from collections import Counter, defaultdict

BOS, EOS = "<s>", "</s>"


# ------------------------------------------------------------------ data
def load_tagged(path="data/corpus.csv"):
    """Return {doc_id: [(words, tags), ...]}."""
    docs = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            words, tags = row["sentence"].split(), row["tags"].split()
            assert len(words) == len(tags), "word/tag length mismatch"
            docs[int(row["doc_id"])].append((words, tags))
    return dict(docs)


def split_by_document(docs, test_fraction=0.2):
    ids = sorted(docs)
    cut = int(len(ids) * (1 - test_fraction))
    return ([x for i in ids[:cut] for x in docs[i]], [x for i in ids[cut:] for x in docs[i]])


# ------------------------------------------------------------------ sampling helper
def sample_from(counter, temperature, rng):
    """Sample a key of `counter`; weights c ** (1/T). T < 1 -> safer, T > 1 -> more chaotic."""
    items = sorted(counter.items())                       # sorted => reproducible for a given seed
    weights = [c ** (1.0 / temperature) for _, c in items]
    return rng.choices([k for k, _ in items], weights=weights, k=1)[0]


# ------------------------------------------------------------------ state splitting
def refine(tags):
    """Split every state by its ROLE in the sentence: S = before the verb, O = after the verb,
    P = inside the prepositional phrase.  DET NOUN VERB DET NOUN PREP DET NOUN PUNCT becomes
    DET_S NOUN_S VERB DET_O NOUN_O PREP DET_P NOUN_P PUNCT.  A first-order HMM cannot know where it
    is in the sentence from the coarse tag alone; richer states give it that memory."""
    out, role = [], "S"
    for t in tags:
        if t == "VERB":
            out.append(t); role = "O"
        elif t == "PREP":
            out.append(t); role = "P"
        elif t == "PUNCT":
            out.append(t)
        else:
            out.append(f"{t}_{role}")
    return out


def coarse(states):
    """DET_S -> DET (back to the plain part-of-speech tag)."""
    return [s.split("_")[0] for s in states]


# ------------------------------------------------------------------ HMM
class HMM:
    def __init__(self, tagged_sentences, k=0.01, split_states=False):
        self.k, self.split_states = k, split_states
        self.start, self.trans, self.emit = Counter(), defaultdict(Counter), defaultdict(Counter)
        for words, tags in tagged_sentences:
            tags = refine(tags) if split_states else tags
            self.start[tags[0]] += 1
            for i, (w, t) in enumerate(zip(words, tags)):
                self.emit[t][w] += 1
                if i > 0:
                    self.trans[tags[i - 1]][t] += 1
        self.tags = sorted(self.emit)
        self.vocab = {w for c in self.emit.values() for w in c}
        self.V = len(self.vocab) + 1                      # +1 for the unknown-word slot
        self.tag_totals = {t: sum(c.values()) for t, c in self.emit.items()}

    # ---- generation
    def sample_sentence(self, rng, temperature=1.0, max_len=14):
        tag = sample_from(self.start, temperature, rng)
        tags = [tag]
        while tag != "PUNCT" and len(tags) < max_len and self.trans.get(tag):
            tag = sample_from(self.trans[tag], temperature, rng)
            tags.append(tag)
        words = [sample_from(self.emit[t], temperature, rng) for t in tags]
        return words, coarse(tags)

    def sample_with_states(self, rng, temperature=1.0, max_len=14):
        """Same as sample_sentence but also returns the hidden states (refined if split_states)."""
        tag = sample_from(self.start, temperature, rng)
        states = [tag]
        while tag != "PUNCT" and len(states) < max_len and self.trans.get(tag):
            tag = sample_from(self.trans[tag], temperature, rng)
            states.append(tag)
        return [sample_from(self.emit[t], temperature, rng) for t in states], states

    # ---- probabilities (smoothed) used by Viterbi
    def p_trans(self, a, b):
        c = self.trans.get(a, Counter())
        return (c[b] + self.k) / (sum(c.values()) + self.k * len(self.tags))

    def p_start(self, t):
        return (self.start[t] + self.k) / (sum(self.start.values()) + self.k * len(self.tags))

    def p_emit(self, t, w):
        return (self.emit[t][w] + self.k) / (self.tag_totals[t] + self.k * self.V)

    def viterbi(self, words):
        """Most probable state sequence (log-space dynamic programming); returns COARSE tags."""
        if not words:
            return []
        T = self.tags
        score = [{t: math.log(self.p_start(t)) + math.log(self.p_emit(t, words[0])) for t in T}]
        back = []
        for w in words[1:]:
            col, bp = {}, {}
            for t in T:
                best_prev = max(T, key=lambda p: score[-1][p] + math.log(self.p_trans(p, t)))
                col[t] = score[-1][best_prev] + math.log(self.p_trans(best_prev, t)) + math.log(self.p_emit(t, w))
                bp[t] = best_prev
            score.append(col); back.append(bp)
        last = max(T, key=lambda t: score[-1][t])
        path = [last]
        for bp in reversed(back):
            path.append(bp[path[-1]])
        return coarse(path[::-1])

    def transition_matrix(self):
        """Row-normalised transition counts (for the heat-map)."""
        rows = []
        for a in self.tags:
            tot = sum(self.trans[a].values()) or 1
            rows.append([self.trans[a][b] / tot for b in self.tags])
        return self.tags, rows


# ------------------------------------------------------------------ n-gram generator (re-uses Lab 02 ideas)
class NGramGenerator:
    """Trigram counts with fall-back to bigram / unigram when a context is unseen."""

    def __init__(self, sentences):
        self.tri, self.bi, self.uni = defaultdict(Counter), defaultdict(Counter), Counter()
        for s in sentences:
            p = [BOS, BOS] + s + [EOS]
            for i in range(2, len(p)):
                self.tri[(p[i - 2], p[i - 1])][p[i]] += 1
                self.bi[(p[i - 1],)][p[i]] += 1
                self.uni[p[i]] += 1
        self.vocab = {w for s in sentences for w in s}

    def next_counts(self, history):
        h = [BOS, BOS] + history
        for table, ctx in ((self.tri, tuple(h[-2:])), (self.bi, tuple(h[-1:]))):
            if ctx in table:
                return table[ctx]
        return self.uni

    def sample_sentence(self, rng, temperature=1.0, max_len=14, prefix=None):
        words = list(prefix or [])
        while len(words) < max_len:
            w = sample_from(self.next_counts(words), temperature, rng)
            if w == EOS:
                break
            words.append(w)
        return words


# ------------------------------------------------------------------ evaluation helpers
def sentence_features(words, tags):
    """(tag pattern, (first noun, first verb) pair) of one tagged sentence."""
    noun = next((w for w, t in zip(words, tags) if t == "NOUN"), None)
    verb = next((w for w, t in zip(words, tags) if t == "VERB"), None)
    return " ".join(tags), (noun, verb)


def generation_metrics(samples, train_sentences, tagger):
    """samples: list of (words, tags_or_None). Tags are computed with Viterbi when missing."""
    train_text = {" ".join(w) for w, _ in train_sentences}
    train_patterns = {" ".join(t) for _, t in train_sentences}
    train_pairs = {sentence_features(w, t)[1] for w, t in train_sentences}
    valid = novel = plausible = 0
    bigrams = []
    for words, tags in samples:
        tags = tags or tagger.viterbi(words)
        pattern, pair = sentence_features(words, tags)
        valid += pattern in train_patterns
        novel += " ".join(words) not in train_text
        plausible += pair in train_pairs
        bigrams += list(zip(words, words[1:]))
    n = len(samples)
    return {"valid_tag_pattern": valid / n, "novel_sentences": novel / n,
            "plausible_subject_verb": plausible / n,
            "distinct_2": len(set(bigrams)) / max(len(bigrams), 1)}


def oov_rate(words, vocab):
    return sum(w not in vocab for w in words) / max(len(words), 1)


if __name__ == "__main__":
    docs = load_tagged()
    train, _ = split_by_document(docs)
    hmm, ng = HMM(train), NGramGenerator([w for w, _ in train])
    rng = random.Random(7)
    print("HMM    :", " ".join(hmm.sample_sentence(rng)[0]))
    print("n-gram :", " ".join(ng.sample_sentence(rng)))
