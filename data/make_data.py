"""Build the small instructor corpus used by Labs 02, 03, 07 and 09 (identical file in each lab).

Run from the lab folder:  python data/make_data.py
Output: data/corpus.csv  with columns doc_id, sentence, tags

The corpus is SYNTHETIC: sentences are sampled from a small hand-written grammar
(fixed seed 42), so there are no licensing or privacy issues and every lab runs offline.
Consequences you must mention in your README:
  * the language is far more regular than real English -> perplexities are LOW
  * the vocabulary has semantic groups (animals, foods, people, places, objects) used in Lab 07
  * every word has a gold part-of-speech tag (used in Lab 03); "watch" is ambiguous (NOUN/VERB)
Sentences are grouped into "documents" of 20 sentences: train/test splits are made BY DOCUMENT.
"""
import csv
import os
import random

SEED = 42
N_SENTENCES = 1200
DOC_SIZE = 20

ADJ = ["small", "hungry", "happy", "old", "quiet", "brave", "young", "lazy"]
ANIMALS = ["cats", "dogs", "foxes", "horses", "birds", "mice"]
FOODS = ["bread", "cheese", "apples", "soup", "rice", "cakes"]
PEOPLE = ["girls", "boys", "farmers", "teachers", "doctors", "children"]
PLACES = ["garden", "market", "school", "river", "kitchen", "park"]
OBJECTS = ["watch", "ball", "book", "lamp", "key", "coin"]
PREPS = ["in", "near", "behind", "at", "by"]
V_ANIMAL_INTR = ["sleep", "run", "play"]
V_ANIMAL_TR = ["chase", "follow", "fear"]
V_FOOD = ["eat", "cook", "buy", "sell", "bring"]
V_PLACE = ["visit", "clean", "leave"]
V_FIND = ["find", "buy", "bring", "lose"]

CATEGORIES = {"animal": ANIMALS, "food": FOODS, "person": PEOPLE, "place": PLACES, "object": OBJECTS}


def np_(rng, nouns, adj_prob=0.35):
    """DET (ADJ) NOUN  -> (words, tags)"""
    words, tags = ["the"], ["DET"]
    if rng.random() < adj_prob:
        words.append(rng.choice(ADJ)); tags.append("ADJ")
    words.append(rng.choice(nouns)); tags.append("NOUN")
    return words, tags


def pp(rng):
    p = rng.choice(PREPS)
    w, t = np_(rng, PLACES, 0.15)
    return [p] + w, ["PREP"] + t


def sentence(rng):
    kind = rng.choice(["a_intr", "a_tr", "p_food", "p_place", "p_watch", "a_eat", "p_find"])
    words, tags = [], []

    def add(w, t):
        words.extend(w); tags.extend(t)

    if kind == "a_intr":      # the happy cats sleep in the garden .
        add(*np_(rng, ANIMALS)); add([rng.choice(V_ANIMAL_INTR)], ["VERB"]); add(*pp(rng))
    elif kind == "a_tr":      # the dogs chase the cats .
        add(*np_(rng, ANIMALS)); add([rng.choice(V_ANIMAL_TR)], ["VERB"]); add(*np_(rng, ANIMALS, 0.2))
    elif kind == "p_food":    # the farmers sell the cheese at the market .
        add(*np_(rng, PEOPLE)); add([rng.choice(V_FOOD)], ["VERB"]); add(*np_(rng, FOODS, 0.2)); add(*pp(rng))
    elif kind == "p_place":   # the teachers visit the school .
        add(*np_(rng, PEOPLE)); add([rng.choice(V_PLACE)], ["VERB"]); add(*np_(rng, PLACES, 0.2))
    elif kind == "p_watch":   # the girls watch the birds in the park .   (watch = VERB)
        add(*np_(rng, PEOPLE)); add(["watch"], ["VERB"]); add(*np_(rng, ANIMALS, 0.2)); add(*pp(rng))
    elif kind == "a_eat":     # the mice eat the cheese in the kitchen .
        add(*np_(rng, ANIMALS)); add(["eat"], ["VERB"]); add(*np_(rng, FOODS, 0.2)); add(*pp(rng))
    else:                     # the children find the watch in the garden .  (watch = NOUN)
        add(*np_(rng, PEOPLE)); add([rng.choice(V_FIND)], ["VERB"]); add(*np_(rng, OBJECTS, 0.2)); add(*pp(rng))
    words.append("."); tags.append("PUNCT")
    return words, tags


def main():
    rng = random.Random(SEED)
    os.makedirs("data", exist_ok=True)
    with open("data/corpus.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["doc_id", "sentence", "tags"])
        for i in range(N_SENTENCES):
            words, tags = sentence(rng)
            w.writerow([i // DOC_SIZE, " ".join(words), " ".join(tags)])
    print(f"wrote data/corpus.csv: {N_SENTENCES} sentences in {N_SENTENCES // DOC_SIZE} documents")


if __name__ == "__main__":
    main()
