"""Lab 03 - Chaotic Text Generator, complete Streamlit test bench.
Run from project root: streamlit run app.py
"""
import csv
import io
import json
import random
from pathlib import Path

import streamlit as st

from src.pipeline import (
    HMM, NGramGenerator, generation_metrics, load_tagged,
    sentence_features, split_by_document,
)

st.set_page_config(page_title="Chaotic Text Generator | HMM Lab", page_icon="🧪", layout="wide")
st.title("🧪 Chaotic Text Generator")
st.write("Explore a first-order POS-HMM, a role-split POS-HMM, and a trigram generator on the same synthetic corpus.")
st.info("The corpus is synthetic and the grammar is tiny. Metrics such as valid tag pattern are defined relative to this corpus and do not measure the quality of real English.")

MODEL_NAMES = {"Plain HMM": "hmm", "Role-split HMM": "hmm_split", "Trigram": "ngram"}

@st.cache_resource(show_spinner="Training count-based generators...")
def get_models():
    docs = load_tagged(str(Path(__file__).parent / "data" / "corpus.csv"))
    train, test = split_by_document(docs)
    return train, test, HMM(train), HMM(train, split_states=True), NGramGenerator([w for w, _ in train])

train, test, hmm_plain, hmm_split, trigram = get_models()
models = {"Plain HMM": hmm_plain, "Role-split HMM": hmm_split, "Trigram": trigram}
train_patterns = {" ".join(tags) for _, tags in train}
train_text = {" ".join(words) for words, _ in train}
train_pairs = {sentence_features(words, tags)[1] for words, tags in train}


def generate_batch(name, n, temp, seed, prefix=None):
    rng = random.Random(int(seed))
    samples = []
    for _ in range(n):
        if name == "Trigram":
            words = trigram.sample_sentence(rng, temp, prefix=prefix)
            tags = hmm_split.viterbi(words)  # Trigram has no hidden states; infer POS with Viterbi.
        else:
            words, tags = models[name].sample_sentence(rng, temp)
        samples.append((words, tags))
    return samples


def rows_and_metrics(samples, name):
    rows = []
    for i, (words, tags) in enumerate(samples, 1):
        pattern, pair = sentence_features(words, tags)
        valid = pattern in train_patterns
        copied = " ".join(words) in train_text
        rows.append({"#": i, "Sentence": " ".join(words),
                     "POS tags (trigram: Viterbi inferred)": " ".join(tags),
                     "Valid pattern": "✓" if valid else "✗",
                     "Subject–verb plausible": "✓" if pair in train_pairs else "✗",
                     "Copied from training": "Yes" if copied else "No"})
    return rows, generation_metrics(samples, train, hmm_split)


with st.sidebar:
    st.header("Generation controls")
    selected = st.selectbox("Model", list(MODEL_NAMES))
    temperature = st.slider("Temperature", min_value=0.3, max_value=3.0, value=1.0, step=0.1,
                            help="Below 1.0: more predictable. Above 1.0: more varied, possibly less valid.")
    seed = st.number_input("Random seed", min_value=0, max_value=1000000, value=42, step=1)
    count = st.slider("Number of sentences", 10, 500, 100, step=10)
    prompt = st.text_input("Optional trigram prefix", value="", help="Only applies to Trigram; words must be in the training vocabulary.") if selected == "Trigram" else ""
    st.caption(f"Training: {len(train)} sentences | Test: {len(test)} sentences | Six POS tags")

prefix = prompt.lower().strip().split()
unknown_prefix = [w for w in prefix if w not in trigram.vocab]
if unknown_prefix:
    st.warning("Unknown trigram prompt word(s): " + ", ".join(sorted(set(unknown_prefix))) + ". Please use training-vocabulary words.")
if len(prefix) >= 14:
    st.warning("The trigram prompt must contain fewer than 14 words.")

st.subheader("1. Generate and evaluate")
if st.button("Generate sentences", type="primary", disabled=bool(unknown_prefix or len(prefix) >= 14)):
    batch = generate_batch(selected, count, temperature, seed, prefix if selected == "Trigram" else None)
    rows, metrics = rows_and_metrics(batch, selected)
    st.session_state["generation"] = {"rows": rows, "metrics": metrics, "model": selected,
                                      "temperature": temperature, "seed": int(seed)}

result = st.session_state.get("generation")
if result:
    st.caption(f"Batch: {result['model']} • T={result['temperature']:.1f} • seed={result['seed']} • {len(result['rows'])} sentences")
    m = result["metrics"]
    boxes = st.columns(4)
    for col, label, key in zip(boxes, ["Valid tag pattern", "Plausible subject–verb", "Novel sentences", "Distinct-2"],
                               ["valid_tag_pattern", "plausible_subject_verb", "novel_sentences", "distinct_2"]):
        col.metric(label, f"{m[key]:.1%}" if key != "distinct_2" else f"{m[key]:.3f}")
    st.dataframe(result["rows"], hide_index=True, use_container_width=True, height=420)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(result["rows"][0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(result["rows"])
    st.download_button("Download generated sentences (CSV)", buffer.getvalue().encode("utf-8-sig"),
                       file_name="hmm_generated_sentences.csv", mime="text/csv")
else:
    st.caption("Choose settings and generate 10–500 sentences to see per-sentence tags and aggregate metrics.")

st.divider()
st.subheader("2. Viterbi POS tagger")
st.caption("Enter a sentence; the HMM predicts the most likely part-of-speech sequence in log-probability space.")
custom_text = st.text_input("Sentence to tag", value="the cats watch the birds .")
use_role_split = st.checkbox("Use role-split HMM to tag", value=False)
if st.button("Tag sentence"):
    tokens = custom_text.lower().strip().split()
    if not tokens:
        st.warning("Enter at least one word.")
    else:
        vocabulary = hmm_plain.vocab
        unknown = sorted({w for w in tokens if w not in vocabulary})
        if unknown:
            st.warning("Out-of-vocabulary word(s): " + ", ".join(unknown) + ". Add-k smoothing prevents a crash, but these tags may be unreliable.")
        tagger = hmm_split if use_role_split else hmm_plain
        tags = tagger.viterbi(tokens)
        st.dataframe([{"Position": i + 1, "Word": word, "Viterbi POS tag": tag,
                       "Vocabulary": "Known" if word in vocabulary else "Unknown"}
                      for i, (word, tag) in enumerate(zip(tokens, tags))],
                     use_container_width=True, hide_index=True)

st.divider()
st.subheader("3. Model comparison")
with st.expander("Compare all three models using the same settings"):
    st.caption("Each model uses its own RNG with the same seed; this is a reproducible comparison, not identical samples.")
    if st.button("Run comparison", key="compare"):
        comparison = []
        for name in MODEL_NAMES:
            samples = generate_batch(name, count, temperature, seed)
            metrics = generation_metrics(samples, train, hmm_split)
            comparison.append({"Model": name, **{k: round(v, 4) for k, v in metrics.items()}})
        st.session_state["comparison"] = comparison
    if "comparison" in st.session_state:
        st.dataframe(st.session_state["comparison"], use_container_width=True, hide_index=True)

with st.expander("Evaluation at four temperatures (500 samples per model)"):
    metrics_file = Path(__file__).parent / "results" / "metrics.json"
    if metrics_file.exists():
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))["generation"]
        plot_rows = []
        for model, temps in saved.items():
            for temp, values in temps.items():
                plot_rows.append({"Temperature": float(temp), "Model": model,
                                  "Valid tag pattern": values["valid_tag_pattern"]})
        import pandas as pd
        frame = pd.DataFrame(plot_rows)
        st.line_chart(frame.pivot(index="Temperature", columns="Model", values="Valid tag pattern"))
        st.caption("Precomputed with python -m src.evaluate, 500 samples per setting, seed=42.")
    else:
        st.caption("Run `python -m src.evaluate` to generate the temperature-comparison plot.")

st.divider()
st.caption("Coursework demo • Generated text may be nonsensical or reproduce training examples. Not an English-language quality benchmark.")
