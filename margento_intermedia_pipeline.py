


# POETRY SONIC-TEMPORAL & AFFECT ANALYSIS PIPELINE
import re, unicodedata, math
from collections import Counter


# Download Unicode Scripts.txt (nned to run only once; you can also cache this locally)
SCRIPTS_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Scripts.txt"


# ALL SCRIPTS 'UNDER THE SUN' [IN UNICODE, THAT IS]

import regex
import urllib.request


def get_all_scripts() -> set[str]:
    """
    Fetch the official Unicode script names from Scripts.txt.
    """
    with urllib.request.urlopen(SCRIPTS_URL) as f:
        lines = f.read().decode("utf-8").splitlines()

    scripts = set()
    for line in lines:
        if line.strip() and not line.startswith("#"):
            # Example line: "0041..005A; Latin # L&  [26] LATIN CAPITAL LETTER A..Z"
            parts = line.split(";")
            if len(parts) >= 2:
                script = parts[1].strip().split()[0]
                scripts.add(script)
    return scripts

UNICODE_SCRIPTS = sorted(get_all_scripts())

def char_script(ch):
    import regex
    
    if not ch or len(ch) != 1:
        return "INVALID"

    for script in UNICODE_SCRIPTS:
        try:
            # Use the script name exactly as Unicode defines it
            if regex.match(rf"\p{{Script={script}}}", ch):
                return script  # return it as-is
        except regex.error:
            continue  # skip invalid/unrecognized scripts

    return "UNKNOWN"


from collections import Counter

def word_script(word: str) -> str:
    """
    Return the dominant script of a word (based on majority of alphabetic chars).
    """
    scripts = Counter(char_script(ch) for ch in word if ch.isalpha())
    return scripts.most_common(1)[0][0] if scripts else "OTHER"


def get_unicode_name(ch):
    try:
        return unicodedata.name(ch)
    except ValueError:
        return None


import unicodedata

# Latin/Cyrillic/Greek/Devanagari vowels (extendable)
_vowel_re_latin = re.compile(r"[aeiouy\u00E0-\u00FF]+", re.IGNORECASE)
_vowel_re_cyrillic = re.compile(r"[аеёиоуыэюя]+", re.IGNORECASE)  # basic Russian vowels
_vowel_re_greek = re.compile(r"[αεηιουωάέήίόύώ]", re.IGNORECASE)   # modern Greek vowels
_vowel_re_devanagari = re.compile(r"[अआइईउऊएऐओऔऋॠॡॢॣ]", re.IGNORECASE)
#!pip install nltk

def approx_syllables_word(word: str) -> int:
    if not word:
        return 0
    w = unicodedata.normalize("NFC", word)
    script = word_script(w)

    if script == "LATIN":
        groups = _vowel_re_latin.findall(w)
        count = len(groups)
        if w.lower().endswith("e") and count > 1:  # silent 'e'
            count -= 1
        return max(1, count)

    if script == "CYRILLIC":
        groups = _vowel_re_cyrillic.findall(w)
        return max(1, len(groups))

    if script == "GREEK":
        groups = _vowel_re_greek.findall(w)
        return max(1, len(groups))

    if script == "DEVANAGARI":
        groups = _vowel_re_devanagari.findall(w)
        return max(1, len(groups))

    if script in ("HIRAGANA", "KATAKANA"):
        kana_chars = [ch for ch in w if '\u3040' <= ch <= '\u30FF']
        return max(1, len(kana_chars))

    if script == "HANGUL":
        return len([ch for ch in w if '\uAC00' <= ch <= '\uD7A3'])

    if script == "CJK":
        chars = [ch for ch in w if '\u4E00' <= ch <= '\u9FFF']
        return max(1, len(chars))

    if script == "THAI":
        return max(1, len([ch for ch in w if ch.strip()]))

    # Fallback
    groups = _vowel_re_latin.findall(w)
    return max(1, len(groups) if groups else len(w))


def extract_phonological_clusters(word: str):
    clusters = set()
    w = unicodedata.normalize("NFC", word.lower())
    script = word_script(w)

    if script in ("LATIN", "GREEK", "CYRILLIC"):
        consonant_matches = re.findall(r'[^aeiouy]+', w)
        for c in consonant_matches:
            for i in range(len(c)):
                for j in range(i+1, len(c)+1):
                    clusters.add(c[i:j])
        vowel_matches = re.findall(r'[aeiouy]+', w)
        for v in vowel_matches:
            for i in range(len(v)):
                for j in range(i+1, len(v)+1):
                    clusters.add(v[i:j])
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script in ("ARABIC", "HEBREW"):
        consonant_runs = re.findall(r'[^aeiou]+', w)
        for c in consonant_runs:
            for i in range(len(c)):
                for j in range(i+1, len(c)+1):
                    clusters.add(c[i:j])
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script == "DEVANAGARI":
        groups = _vowel_re_devanagari.findall(w)
        for g in groups:
            clusters.add(g)
        for k in range(2, 5):
            if len(w) >= k:
                clusters.add(w[-k:])

    elif script in ("HIRAGANA", "KATAKANA", "HANGUL", "CJK"):
        chars = list(w)
        clusters.update(chars)
        for i in range(len(chars)-1):
            clusters.add(chars[i] + chars[i+1])

    else:
        for i in range(len(w)):
            for j in range(i+1, min(i+4, len(w))+1):
                clusters.add(w[i:j])

    return clusters


_word_re = re.compile(r"\w+", re.UNICODE)

def tokenize_text(text):
    tokens = []
    for m in _word_re.finditer(text):
        tok = m.group(0)
        tokens.append(tok)
    return tokens

_fricatives = set(list("fvsz") + ["sh","zh","th"])
_plosives = set(list("pbtdkg"))

def phonetic_density(tokens):
    latin_tokens = [t for t in tokens if char_script(t[0]) == "LATIN"]
    joined = " ".join(latin_tokens).lower()
    letters = re.sub(r'[^a-z]', '', joined)
    if not letters:
        return 0.0, 0.0, 0.0
    fric_count = sum(joined.count(f) for f in ["f","v","s","z","sh","zh","th"])
    plos_count = sum(joined.count(p) for p in ["p","b","t","d","k","g"])
    vowel_count = sum(1 for c in letters if c in "aeiouy")
    total = len(letters)
    return fric_count/total, plos_count/total, vowel_count/(total+1e-9)

import re
import nltk
from nltk.corpus import words, stopwords, wordnet as wn
from nltk.corpus.reader import WordListCorpusReader
import numpy as np

# Download necessary NLTK resources
nltk.download('words')
nltk.download('stopwords')
nltk.download('omw')
nltk.download('omw-1.4')

# Load English words and stopwords
english_words = set(words.words())
stop_words = set(stopwords.words('english'))

# Define common prefixes and suffixes
PL_PREFIXES = {"re", "un", "in", "dis", "pre", "sub"}
PL_SUFFIXES = {"ing", "ed", "er", "ly", "es", "ful"}

def is_plausible_fragment(fragment):
    """Check if fragment is a plausible English word, prefix/suffix, or foreign fragment."""
    fragment = fragment.lower()
    if not fragment:
        return False
    if fragment in english_words:
        return True
    if fragment in PL_PREFIXES or fragment in PL_SUFFIXES:
        return True
    # Check if fragment exists in WordNet for any language
    for lang in wn.langs():
        if wn.synsets(fragment, lang=lang):
            return True
    # Fallback: accept fragments that are at least 2 characters long
    if len(fragment) > 1:
        return True
    return False

def extract_audio_features_from_stanza(stanza, expected_feet_per_line=(5,6), foot_syllables=(2,3)):
    lines = [ln.strip() for ln in stanza.strip().split("\n") if ln.strip()]
    n_lines = max(1, len(lines))
    tokens = tokenize_text(stanza)
    syll_counts_tokens = [approx_syllables_word(t) for t in tokens]
    total_syllables = sum(syll_counts_tokens)
    n_words = len(tokens) if tokens else 1
    syllable_density = total_syllables / n_words if n_words else 0.0
    target_feet = np.mean(expected_feet_per_line)
    avg_syll_per_line = total_syllables / max(1, len(lines))
    avg_foot_syll = np.mean(foot_syllables)
    tempo = avg_syll_per_line / avg_foot_syll
    sylls_per_line = [sum(approx_syllables_word(t) for t in tokenize_text(ln)) for ln in lines]
    pacing_variance = float(np.var(sylls_per_line)) if sylls_per_line else 0.0
    fric_density, plos_density, vowel_ratio = phonetic_density(tokens)
    vocal_smoothness = float(vowel_ratio)

    # --- [word-splitting] enjambment detection (needed in this specific case; if you need to process enjambments in general see https://github.com/Margento/Computationally_Assembled_Belgian_Poetry_Anthology ---
    enjambments = 0
    enjambed_positions = set()

    for ln_idx, ln in enumerate(lines):
        # 1. End-of-line split (including ellipses)
        end_match = re.search(r'(\w+(?:\.\.\.)?)-/?(\w*)$', ln)
        if end_match:
            left, right = end_match.groups()
            if is_plausible_fragment(left) and (not right or is_plausible_fragment(right)):
                enjambments += 1
                enjambed_positions.add(end_match.start())

        # 2. Start-of-line split
        if ln_idx > 0:
            start_match = re.match(r'^(\w*)-/(\w+)', ln)
            if start_match:
                left, right = start_match.groups()
                if (not left or is_plausible_fragment(left)) and is_plausible_fragment(right):
                    enjambments += 1
                    enjambed_positions.add(start_match.start())

        # 3. Multi-word or foreign-word consideration (fallback)
        for match in re.finditer(r'(\S+)/(\S+)', ln):
            left, right = match.groups()
            if is_plausible_fragment(left) and is_plausible_fragment(right):
                enjambments += 1
                enjambed_positions.add(match.start())

    # Count pause marks excluding those part of valid enjambments
    pause_marks = 0
    for m in re.finditer(r'[,;:\-\—\(\)]', stanza):
        if m.start() not in enjambed_positions:
            pause_marks += 1

    silence_ratio = pause_marks / (total_syllables + 1e-9)
    caesura = sum(1 for ln in lines if "," in ln or ";" in ln or "—" in ln)

    enjambments_norm = enjambments / n_lines
    caesura_norm = caesura / n_lines

    audio = {
        "syllable_density": float(syllable_density),
        "tempo": float(tempo),
        "pacing_variance": float(pacing_variance),
        "fricative_density": float(fric_density),
        "plosive_density": float(plos_density),
        "vocal_smoothness": float(vocal_smoothness),
        "silence_ratio": float(silence_ratio),
        "total_syllables": int(total_syllables),
        "sylls_per_line": sylls_per_line,
        "enjambments": float(enjambments_norm),
        "caesura": float(caesura_norm),
        "n_words": n_words
    }
    return audio

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ Using Apple Silicon GPU via MPS")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not available, falling back to CPU")



from transformers import (
    XLMRobertaTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)
import torch

model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    use_safetensors=True
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)


def stanza_affect_vector(stanza):
    """
    Extract affective features (valence, arousal, energy) from a stanza of text,
    combining multilingual sentiment analysis with audio-like features.
    """

    # --- 1. Multilingual Sentiment Analysis (Hugging Face) ---
    try:
        sentiment_result = sentiment_pipeline(stanza[:512])[0]  # truncate to model limit
        label = sentiment_result["label"].lower()
        score = sentiment_result["score"]

        # Map labels to a polarity value in [-1, 1]
        if "negative" in label:
            polarity = -score
        elif "positive" in label:
            polarity = score
        else:  # neutral
            polarity = 0.0
    except Exception as e:
        print(f"Sentiment analysis failed: {e}")
        polarity = 0.0

    # Calculate valence from polarity
    valence = float(math.tanh(polarity * 5.0))

    # --- 2. Extract "audio" features ---
    audio_feats = extract_audio_features_from_stanza(stanza)

    # --- 3. Calculate arousal & energy ---
    arousal = (audio_feats["pacing_variance"] ** 0.5
               + audio_feats["fricative_density"] * 0.5
               + min(1.0, audio_feats["silence_ratio"] * 2.0))
    arousal = float(math.tanh(arousal))

    energy = float(math.tanh(
        (audio_feats["tempo"] * 0.6) +
        (audio_feats["syllable_density"] * 0.2)
    ))

    return {
        "valence": valence,
        "arousal": arousal,
        "energy": energy,
        # "audio_feats": audio_feats
    }
# TEMPORAL FEATURES
def stanza_temporal_structures(stanza):
    lines = [ln.strip() for ln in stanza.strip().split("\n") if ln.strip()]
    n_segments = len(lines)
    segment_annotations = []
    motifs_counter = Counter()
    
    # process clusters instead of whole words
    for i, ln in enumerate(lines):
        words = tokenize_text(ln)
        segment_annotations.append(
            f"line_{i+1}: {len(words)} words, {sum(approx_syllables_word(w) for w in words)} sylls"
        )
        for w in words:
            clusters = extract_phonological_clusters(w)
            for c in clusters:
                motifs_counter[c] += 1
    
    motifs = [cl for cl,cnt in motifs_counter.items() if cnt > 1]
    n_motifs = len(motifs)
    n_uniques = len(motifs_counter)
    
    sylls_per_line = [
        sum(approx_syllables_word(w) for w in tokenize_text(ln)) for ln in lines
    ] if lines else []
    
    ruptures = []
    if sylls_per_line:
        mean = np.mean(sylls_per_line); sd = np.std(sylls_per_line)
        for i, s in enumerate(sylls_per_line):
            if sd > 0 and abs(s-mean) > 1.5*sd:
                ruptures.append({
                    "line": i+1, 
                    "syllables": int(s), 
                    "deviation": float((s-mean)/sd)
                })
    
    score_linear = 0.0
    if len(sylls_per_line) > 1:
        x = np.arange(len(sylls_per_line))
        y = np.array(sylls_per_line)
        cov = np.cov(x, y)[0,1]
        if np.std(x) > 0 and np.std(y) > 0:
            score_linear = float(cov / (np.std(x) * np.std(y)))
    
    score_cyclical = 0.0
    if len(sylls_per_line) > 2:
        y = np.array(sylls_per_line) - np.mean(sylls_per_line)
        score_cyclical = float(np.correlate(y, np.roll(y,1))[0] / (np.sum(y*y)+1e-9))
    
    ngrams = Counter()
    for ln in lines:
        toks = [t.lower() for t in tokenize_text(ln)]
        for i in range(len(toks)-1):
            ngrams[" ".join(toks[i:i+2])] += 1
    repeated_ngrams = sum(1 for c in ngrams.values() if c>1)
    score_recursive = float(repeated_ngrams / (len(ngrams)+1e-9))
    score_hybrid = float((abs(score_linear) + abs(score_cyclical) + score_recursive)/3.0)
    
    recursive_events = motifs[:5]
    
    return {
        "segments": n_segments,
        "segment_annotations": segment_annotations,
        "number_of_motifs": n_motifs,
        "motifs": motifs,
        "uniques": n_uniques,
        "ruptures": ruptures,
        "score_linear": round(score_linear, 3),
        "score_cyclical": round(score_cyclical, 3),
        "score_recursive": round(score_recursive, 3),
        "score_hybrid": round(score_hybrid, 3),
        "recursive_events": recursive_events
    }

def extract_full_stanza_representation(stanza):
    audio_feats =  extract_audio_features_from_stanza(stanza)
    affect = stanza_affect_vector(stanza)
    temporal = stanza_temporal_structures(stanza)

    return {
        "audio_features": audio_feats,
        "affect_vector": affect,
        "temporal_features": temporal,
    }



def build_prompt(shards, profile, poem_context):
    system = """
You are not a co-author.
You are urban noise and linguistic interference.

You do not extend the poem.
You interrupt it.

Your output must feel misaligned, partial, and parasitic.
"""

    rules = f"""
CONTEXT
-------
You are disrupting a poem currently unfolding.
You do not summarize it.
You do not continue it.
You cut into it sideways.

CURRENT POEM CONTEXT
--------------------
{poem_context}

FRAGMENTS (external, unreliable)
---------------
You have been given fragments sampled from a multilingual poetic corpus.
These fragments were selected because they are *near* the poem in rhythm, affect, or motif — not meaning.

INSTRUCTIONS
------------
• Primary output language: Chinese (hybrid, unstable).
• Always include:
  - English fragments
  - Original-language fragments (Japanese, Romanian, etc.)
• Do NOT fully translate anything.
• Allow mistranslations to contradict their originals.
• If a phrase becomes fluent, break it mid-line.

CORPUS ATTUNEMENT
-----------------
• Reuse short motifs (1–3 characters or syllables) across lines.
• Let recursive sounds echo incorrectly.
• Where the original is erotic, make the translation eother clinical or sexually impudent.
• Where the original is delicate, make the translation blunt.
• Where syllable density feels high, remove grammar or make it sound diffuse.
• Where silence or caesura appears, insert more gaps or single characters/short words.

REGISTER MIX (bias, not balance)
--------------------------------
Classical → archaic particles, parallelism, aphoristic compression
Street    → slang, abrupt tone shifts, vulgar intrusions
Broken    → truncation, repetition, unfinished syntax
Technical → sterile nouns, procedural phrasing, dead metaphors

Register weights:
{profile["register_mix"]}

FAILURE MODES (DESIRED)
----------------------
• Awkward line breaks
• Semantic drift
• Inconsistent tense
• Unresolved references
"""

    body = "\n\n".join(
        f"[{s['language']}]\n{s['original']}\n\nEN (unreliable):\n{s['translation']}"
        for s in shards
    )

    return system + rules + "\n\nFRAGMENTS\n---------\n" + body




import numpy as np
import random

# ------------------------------
# CONFIG & UTILITIES
# ------------------------------
WINDOW_SIZE = 4
window_buffer = []
feature_history = []
noise_log = []
prompts = []

theta_start = 0.55
theta_stop  = 1.10
delta_max   = 0.5
# eta_cap     = 0.0
# eta_cap     = 0.1
eta_cap     = 0.2



models = ["gpt-4.1", "qwen2.5-14b"]

def call_writer_model(prompt, model):
    if model == "gpt-4.1":
        return call_gpt41(prompt)
    elif model == "qwen2.5-14b":
        return call_qwen(prompt)
    else:
        raise ValueError(f"Unknown model: {model}")

import os
import requests

API_KEY = os.environ["OPENROUTER_API_KEY"]

def call_qwen(prompt):
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "HTTP-Referer": "your-project",
                "X-Title": "poetic-noise"
            },
            json={
                "model": "qwen/qwen-2.5-14b-instruct",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.9,
                "max_tokens": 1000
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"<<QWEN FAILED: {str(e)[:120]}>>"

def call_gpt41(prompt):
    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"<<GPT-4.1 FAILED: {str(e)[:120]}>>"




def poem_input_stream(path, delay=0.4, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\n")
            time.sleep(delay)


# ------------------------------
# WINDOW MANAGEMENT
# ------------------------------
def update_window(new_line):
    window_buffer.append(new_line)
    if len(window_buffer) > WINDOW_SIZE:
        window_buffer.pop(0)
    return "\n".join(window_buffer)

# ------------------------------
# SAFE FEATURE FLATTENING
# ------------------------------
def flatten_feature_dict(F):
    if isinstance(F, list):
        F = F[0]

    # Handle malformed input (float, None, etc.)
    if not isinstance(F, dict):
        return np.zeros(10, dtype=float)

    audio = F.get("audio_features", {})
    affect = F.get("affect_vector", {})
    temporal = F.get("temporal_features", {})

    vec = [
        audio.get("syllable_density", 0.0),
        audio.get("tempo", 0.0),
        audio.get("pacing_variance", 0.0),
        audio.get("fricative_density", 0.0),
        affect.get("valence", 0.0),
        affect.get("arousal", 0.0),
        affect.get("energy", 0.0),
        temporal.get("score_recursive", 0.0),
        temporal.get("score_linear", 0.0),
        temporal.get("score_cyclical", 0.0)
    ]
    return np.array(vec, dtype=float)

# ------------------------------
# DRIFT & NOISE FUNCTIONS
# ------------------------------
def poetic_saturation(F):
    audio = F["audio_features"]
    affect = F["affect_vector"]
    temporal = F["temporal_features"]

    phon_entropy = audio.get("syllable_density", 0.0)
    repetition   = temporal.get("score_recursive", 0.0)
    affect_var   = affect.get("arousal", 0.0)

    return (
        0.4 * (1 - phon_entropy) +
        0.4 * repetition +
        0.2 * (1 - affect_var)
    )

def external_noise_load(noise_log, horizon=20, decay=0.85):
    total = 0.0
    w = 1.0
    for e in reversed(noise_log[-horizon:]):
        total += w * e["intensity"]
        w *= decay
    return total

theta_start = min(theta_start, 0.85)

def drift_instability(F_hist):
    if len(F_hist) < 3:
        return 0.0
    delta1 = F_hist[-1] - F_hist[-2]
    delta2 = F_hist[-2] - F_hist[-3]
    return np.linalg.norm(delta1 - delta2)


def drift_thresholds():
    global theta_start, theta_stop
    epsilon = random.choice([-1, 1]) * random.paretovariate(1.5) * 0.01
    theta_start += epsilon
    theta_stop  -= epsilon / 2

def should_interrupt(F, F_hist, noise_log):
    S = poetic_saturation(F)
    D = drift_instability(F_hist)
    N = external_noise_load(noise_log)

    drift_thresholds()
    random_spike = random.random() < 0.1  # occasional stochastic trigger

    return (S > theta_start and D < delta_max and N > eta_cap) or random_spike

def generate_noise_profile():
    return {
        "language_bias": {
            "zh": random.uniform(0.45, 0.65),
            "en": random.uniform(0.10, 0.20),
            "rare": random.uniform(0.20, 0.35)
        },
        "register_mix": {
            "classical": random.uniform(0.15, 0.30),
            "street": random.uniform(0.25, 0.40),
            "broken": random.uniform(0.20, 0.30),
            "technical": random.uniform(0.10, 0.20)
        }
    }

# ------------------------------
# NOISY TARGET FEATURES
# ------------------------------
def noisy_target_features(F_vec, noise_profile, noise_load):
    dim = len(F_vec)
    scale = 0.05 + 0.15 * min(noise_load, 1.0)
    reg = noise_profile["register_mix"]

    directional = np.array([
        reg["broken"],
        reg["street"],
        reg["classical"],
        reg["technical"]
    ])
    directional = np.resize(directional, dim)
    xi = np.random.laplace(0.0, scale, size=dim)
    return F_vec + xi + directional * scale

# ------------------------------
# CORPUS SAMPLING
# ------------------------------
def sample_corpus(corpus, F_vec, noise_profile, noise_log, k=10):
    noise_load = external_noise_load(noise_log)
    Ft = noisy_target_features(F_vec, noise_profile, noise_load)

    scored = []
    for entry in corpus:
        # F_trans = entry.get("features_trans", {})
        F_trans = entry.get("features_or", {})
        entry_vec = flatten_feature_dict(F_trans)
        entry["features_vec"] = entry_vec  # cache

        d = np.linalg.norm(entry_vec - Ft)
        reuse_penalty = entry.get("recent_hits", 0) * 0.5
        lang_bonus = random.uniform(0, 0.3)
        score = d + reuse_penalty - lang_bonus
        scored.append((score, entry))

    scored.sort(key=lambda x: x[0])
    band_start = random.randint(len(scored)//6, max(len(scored)//3, 1))
    band = scored[band_start:band_start + k]

    for _, e in band:
        e["recent_hits"] = e.get("recent_hits", 0) + 1

    return [e for _, e in band]



for line in poem_input_stream('margento_hk_suite_live.txt'):

    # Update rolling window
    window_text = update_window(line)

    # Extract stanza features
    F = extract_full_stanza_representation(window_text)
    F_vec = flatten_feature_dict(F)

    # Track feature history
    feature_history.append(F_vec)

    # Decide whether to inject noise
    if should_interrupt(F, feature_history, noise_log):

        profile = generate_noise_profile()
        shards = sample_corpus(corpus, F_vec, profile, noise_log)
        prompt = build_prompt(shards, profile, window_text)
        prompts.append(prompt)

        model = random.choice(models)
        noise = call_writer_model(prompt, model)

        is_error = isinstance(noise, str) and noise.startswith("<<")

        # Log first attempt
        noise_log.append({
            "intensity": sum(profile["register_mix"].values()),
            "model": model,
            "status": "error" if is_error else "ok"
        })

        # Optional fallback: Qwen → GPT
        if is_error and model == "qwen2.5-14b":
            fallback_model = "gpt-4.1"
            noise = call_writer_model(prompt, fallback_model)

            fallback_error = noise.startswith("<<")

            noise_log.append({
                "intensity": sum(profile["register_mix"].values()),
                "model": fallback_model,
                "status": "error" if fallback_error else "ok"
            })

        # Random silent line
        if random.random() < 0.15:
            print("雨")
            
        else:
            print(noise)

        # --- FEEDBACK LOOP: noise becomes part of the poem ---
        tagged_noise = f"[NOISE]\n{noise}"

        # Update rolling window with noise
        window_text = update_window(tagged_noise)

        # Re-extract features INCLUDING the noise
        F_noise = extract_full_stanza_representation(window_text)
        F_noise_vec = flatten_feature_dict(F_noise)

        # Track new state
        feature_history.append(F_noise_vec)

        theta_start = min(theta_start, 0.85)
        
    else:
        print(line)





def extract_audio_from_mp4(mp4_path, wav_out_path):
    clip = VideoFileClip(mp4_path)
    # clip.audio.write_audiofile(wav_out_path, fps=48000, verbose=False, logger=None)
    clip.audio.write_audiofile(wav_out_path, fps=48000, logger=None)

def extract_audio_affect_vector(audio_chunk, sr, embedding_size):
    """
    Extracts an affect vector using OpenL3. Returns the average embedding across time.
    """
    # OpenL3 expects stereo, so ensure shape is (samples, channels)
    if audio_chunk.ndim == 1:
        audio_chunk = np.stack([audio_chunk, audio_chunk], axis=-1)

    emb, timestamps = openl3.get_audio_embedding(
        audio=audio_chunk,
        sr=sr,
        input_repr="mel256",         # REQUIRED! Also valid: "mel128", "linear"
        content_type="music",          # "env" or "music"
        embedding_size=512           # or 6144
    )
    
    return np.mean(emb, axis=0)  # average over time

def get_openl3_vector_from_mp4(mp4_path, tmp_wav="temp.wav"):
    extract_audio_from_mp4(mp4_path, tmp_wav)
    audio, sr = sf.read(tmp_wav)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=-1)

    emb, _ = openl3.get_audio_embedding(audio, sr, content_type="music", embedding_size=512)
    return np.mean(emb, axis=0)
    

AUDIO_EXT = ".wav"
VIDEO_EXT = ".mp4"

# --- AUDIO CHUNKER ---

def chunk_audio(audio_path, chunk_duration):
    audio, sr = sf.read(audio_path)
    chunk_length = int(chunk_duration * sr)
    audio_chunks = []

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i:i + chunk_length]
        if len(chunk) >= chunk_length * 0.8:  # only keep fairly full chunks
            audio_chunks.append(chunk)

    return audio_chunks, sr

# --- VIDEO CHUNKER ---

def chunk_video(video_path, chunk_duration):
    clip = VideoFileClip(video_path)
    video_chunks = []

    duration = clip.duration
    t = 0.0
    while t + chunk_duration <= duration:
        subclip = clip.subclip(t, t + chunk_duration)
        video_chunks.append(subclip)
        t += chunk_duration

    return video_chunks

# --- WRAPPER FOR SYNCHRONIZED AUDIO/VIDEO CHUNKS ---

def chunk_av_for_affect(video_path, audio_path, chunk_duration):
    audio_chunks, sr = chunk_audio(audio_path, chunk_duration)
    video_chunks = chunk_video(video_path, chunk_duration)

    min_len = min(len(audio_chunks), len(video_chunks))
    return audio_chunks[:min_len], video_chunks[:min_len], sr
    
def process_audio_affect_from_videos(folder_path, chunk_duration, content_type="music"): # OR CONTENT TYPE, "env"!!!!!!!!!!
    results = {}
    for file in os.listdir(folder_path):
        if file.lower().endswith(".mp4"):
            basename = os.path.splitext(file)[0]
            mp4_path = os.path.join(folder_path, file)
            wav_path = os.path.join(folder_path, f"{basename}.wav")

            print(f"🔍 Processing {file}...")

            # Extract WAV from MP4
            extract_audio_from_mp4(mp4_path, wav_path)

            # Chunk audio
            audio_chunks, sr = chunk_audio(wav_path, chunk_duration=chunk_duration)

            # Extract affect vectors per chunk
            vectors = [
                extract_audio_affect_vector(chunk, sr, 512)
                for chunk in audio_chunks
            ]

            results[file] = {
                "name": basename,
                "n_chunks": len(audio_chunks),
                "vectors": vectors
            }

    return results

def continue_audio_affect_from_videos(folder_path, chunk_duration, content_type="music"): # OR CONTENT "env"!!!!!!!!!!!!
    for file in os.listdir(folder_path):
        # if file.lower().endswith(".mp4"):
            if file not in list(audio_affect_vectors.keys()):
                basename = os.path.splitext(file)[0]
                wav_path = os.path.join(folder_path, file)
                # wav_path = os.path.join(folder_path, f"{basename}.wav")

                print(f"🔍 Processing {file}...")

                # Extract WAV from MP4
                # extract_audio_from_mp4(mp4_path, wav_path)

                # Chunk audio
                audio_chunks, sr = chunk_audio(wav_path, chunk_duration=chunk_duration)

                # Extract affect vectors per chunk
                vectors = [
                    extract_audio_affect_vector(chunk, sr, 512)
                    for chunk in audio_chunks
                ]

                audio_affect_vectors[file] = {
                    "name": basename,
                    "n_chunks": len(audio_chunks),
                    "vectors": vectors
                }



if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")  # fallback if MPS isn't available

print("Using device:", device)


model = CLIPModel.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", torch_dtype=torch.float16).to(device)
processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-B-32-laion2B-s34B-b79K")


# WE NEED A WARMUP CALL FOR THE MPS
# Warmup data: dummy text and image
dummy_text = ["warmup text"]
dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

# Prepare inputs and send to MPS
inputs = processor(text=dummy_text, images=dummy_image, return_tensors="pt", padding=True)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Run warmup
with torch.no_grad():
    _ = model(**inputs)


def chunk_video_frames(video_path, chunk_duration):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise ValueError(f"Could not determine FPS for video: {video_path}")
    
    frames_per_chunk = int(chunk_duration * fps)
    chunks = []
    current_chunk = []

    success, frame = cap.read()
    while success:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_chunk.append(frame_rgb)

        if len(current_chunk) == frames_per_chunk:
            chunks.append(current_chunk)
            current_chunk = []

        success, frame = cap.read()

    if len(current_chunk) >= frames_per_chunk * 0.5:
        chunks.append(current_chunk)

    cap.release()
    return chunks

def get_clip_vector(frame_rgb, model=model, processor=processor, device="mps"):
    # Convert frame to PIL image
    image = Image.fromarray(frame_rgb)

    # Preprocess image using the processor
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)

    # Return as a NumPy array
    return outputs.squeeze().cpu().numpy()

def extract_video_affect_vector(chunk_frames):
    vectors = [get_clip_vector(frame) for frame in chunk_frames]
    return np.mean(vectors, axis=0)



def process_video_affect_from_videos(folder_path, chunk_duration):
    results = {}

    for file in os.listdir(folder_path):
        if file.lower().endswith(".mp4"):
            video_path = os.path.join(folder_path, file)
            print(f"🎥 Processing {file}...")

            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames_per_chunk = int(chunk_duration * fps)

                current_chunk = []
                vectors = []
                success, frame = cap.read()

                while success:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    current_chunk.append(frame_rgb)

                    if len(current_chunk) == frames_per_chunk:
                        vectors.append(extract_video_affect_vector(current_chunk))
                        current_chunk = []
                        torch.mps.empty_cache()  # frees MPS memory
                        gc.collect()

                    success, frame = cap.read()

                if len(current_chunk) > 0:
                    vectors.append(extract_video_affect_vector(current_chunk))

                cap.release()

                results[file] = {
                    "name": file,
                    "n_chunks": len(vectors),
                    "vectors": vectors
                }

                # Save incrementally (prevents losing all progress if it crashes)
                # with open('nonotak_narcisse_video_affect_vectors_2s.pkl', 'wb') as fp:
                    # pickle.dump(results, fp)

            except Exception as e:
                print(f"❌ Failed to process {file}: {e}")
                continue

    return results

window_duration = 1.0  # seconds FOR AUDIO
hop_duration = 0.5  # seconds  FOR AUDIO


def compute_recursive_drift(segments, labels, hop_duration):
    motif_instances = {}
    event_types = []

    # Dynamically computed time threshold
    avg_segment_duration = np.mean([(end - start) * hop_duration for start, end, _ in segments])
    # time_threshold = max(2.0, avg_segment_duration * 1.5)  # or some multiple of avg segment size
    time_threshold = max(2.0, avg_segment_duration * 2)  # using multiples of avg_segment_duration for clearer distinction between loop and return
    
    for idx, (start, end, features) in enumerate(segments):
        label = labels[idx]
        if label == -1:   # the last one
            continue
        if label not in motif_instances:
            motif_instances[label] = []
        motif_instances[label].append((start, end, features))

    drift_scores = []
    for label, instances in motif_instances.items():
        instances.sort(key=lambda x: x[0])
        for i in range(1, len(instances)):
            prev = instances[i-1]
            curr = instances[i]
            feature_drift = euclidean(curr[2], prev[2])   # distance between features
            time_drift = (curr[0] - prev[0]) * hop_duration   # window_duration = 1.0 [seconds]  hop_duration = 0.5 [seconds]
            drift = feature_drift * time_drift
            drift_scores.append(drift)

            # Event typing based on thresholds (adjustable)
            if feature_drift < 0.3 and time_drift < time_threshold:  # small feature + small time drift → loop
                event_type = 'loop'
            elif feature_drift < 0.3 and time_drift >= time_threshold:
                event_type = 'return'  # or 'resonance', 'memory motif'
            elif feature_drift < 0.7 and time_drift >= time_threshold:  # large time drift + medium feature drift → recursive echo
                event_type = 'recursive echo'
            elif feature_drift >= 0.7 and time_drift < time_threshold:   # small time, large feature drift → mutation (e.g., ironic chorus?)
                event_type = 'mutation'
            else:                            # large time + large feature drift → ghost return or palimpsest
                event_type = 'ghost return'  # significant transformation + long delay

            event_types.append({
                'label': label,
                'from': prev[0] * hop_duration,
                'to': curr[0] * hop_duration,
                'feature_drift': round(feature_drift, 3),
                'drift': round(drift, 3),
                'time_drift': round(time_drift, 2),
                'duration_prev': round((prev[1] - prev[0]) * hop_duration, 2),
                'duration_curr': round((curr[1] - curr[0]) * hop_duration, 2),
                'type': event_type
            })

    recursive_score = np.mean(drift_scores) if drift_scores else 0.0
    return recursive_score, event_types


def analyze_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    window_size = int(sr * window_duration)    # window_duration = 1.0 seconds [but that can be changed]
    hop_size = int(sr * hop_duration)      # hop_duration = 0.5 seconds

    # Extract features
    feature_vectors = []
    for start in range(0, len(y) - window_size, hop_size):
        window = y[start:start + window_size]
        mfcc = np.mean(librosa.feature.mfcc(y=window, sr=sr), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(y=window, sr=sr), axis=1)
        rms = np.mean(librosa.feature.rms(y=window))
        feature_vector = np.concatenate([mfcc, chroma, [rms]])
        feature_vectors.append(feature_vector)

    feature_vectors = np.array(feature_vectors)

    # Compute FVI
    fv_array = np.array(feature_vectors)
    fvi = np.mean(np.abs(np.diff(fv_array, axis=0)), axis=0) / (np.std(fv_array, axis=0) + 1e-6)
    fvi_normalized = fvi / np.sum(fvi)
    weighted_features = fv_array * fvi_normalized

    # Change scores and ruptures
    change_scores = [euclidean(weighted_features[i+1], weighted_features[i]) for i in range(len(weighted_features)-1)]
    threshold = np.mean(change_scores) + np.std(change_scores) * 0.5
    ruptures = [i for i, score in enumerate(change_scores) if score > threshold]

    # Segment features
    segments = []
    start_idx = 0
    for rupture_idx in ruptures + [len(feature_vectors)-1]:
        segment = weighted_features[start_idx:rupture_idx+1]
        avg_feature = np.mean(segment, axis=0)
        segments.append((start_idx, rupture_idx+1, avg_feature))
        start_idx = rupture_idx+1

    segment_features = [seg[2] for seg in segments]
    similarity = cosine_similarity(segment_features)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(segment_features)
    labels = clustering.labels_

    avg_segment_duration = np.mean([(end - start) * hop_duration for start, end, _ in segments])
    rupture_threshold = max(1.5, avg_segment_duration * 0.75)
    
    segment_annotations = []
    for idx, (start, end, _) in enumerate(segments):
        duration = (end - start) * hop_duration
        label = labels[idx]
        if duration < rupture_threshold and (idx == 0 or idx in ruptures):
            seg_type = 'rupture'
        elif label == -1:
            seg_type = 'unique'
        else:
            seg_type = 'motif'
        segment_annotations.append({
            'start': start * hop_duration,
            'end': end * hop_duration,
            'type': seg_type,
            'label': label
        })

    # Calculate scores
    n_segments = len(segment_annotations)
    n_motifs = sum(1 for s in segment_annotations if s['type'] == 'motif')
    n_uniques = sum(1 for s in segment_annotations if s['type'] == 'unique')
    n_ruptures = sum(1 for s in segment_annotations if s['type'] == 'rupture')

    score_linear = 1 - (n_ruptures + n_uniques) / n_segments
    score_cyclical = n_motifs / n_segments
    score_recursive = len(set(l['label'] for l in segment_annotations if l['label'] != -1)) / n_segments  # number of segments that are not unique [i.e., not part of a cluster]
    score_hybrid = 1.0 - min(score_linear, score_cyclical)

    # Compute drift-based recursion and event typing
    recursive_drift_score, event_types = compute_recursive_drift(segments, labels, hop_duration)

    return {
        'file': os.path.basename(file_path),
        'segments': n_segments,
        'segment_annotations': segment_annotations,
        'motifs': n_motifs,
        'uniques': n_uniques,
        'ruptures': n_ruptures,
        'score_linear': round(score_linear, 3),
        'score_cyclical': round(score_cyclical, 3),
        'score_recursive': round(score_recursive, 3),
        'score_recursive_drift': round(recursive_drift_score, 3),
        'score_hybrid': round(score_hybrid, 3),
        'recursive_events': event_types
    }



from sklearn.metrics.pairwise import cosine_distances
import numpy as np
# from sklearn.cluster import DBSCAN
import os

frame_rate = 2

import cv2

def extract_frames(video_path, frame_rate):
    """
    Extract frames from the video at `frame_rate` frames per second.
    Returns list of PIL Images for compatibility with CLIPProcessor.
    """
    from PIL import Image
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps / frame_rate)
    
    frames = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            frames.append(pil_img)
        frame_idx += 1
    
    cap.release()
    return frames

def detect_video_recursion(
    frame_embeddings,
    hop_duration,
    time_threshold,
    feature_drift_threshold=0.3,
    merge_gap=1.0,
    from_merge_gap=1.0
):
    n_segments = len(frame_embeddings)
    dists = cosine_distances(frame_embeddings)

    all_pairs = []
    for i in range(n_segments):
        for j in range(i + 1, n_segments):
            feature_drift = dists[i, j]
            time_drift = (j - i) * hop_duration

            if feature_drift < feature_drift_threshold:
                event_type = 'loop' if time_drift < time_threshold else 'return'
                all_pairs.append({
                    'from_idx': i,
                    'to_idx': j,
                    'from': i * hop_duration,
                    'to': j * hop_duration,
                    'feature_drift': feature_drift,
                    'drift': feature_drift * time_drift,
                    'time_drift': time_drift,
                    'duration_prev': hop_duration,
                    'duration_curr': hop_duration,
                    'type': event_type
                })

    merged_events = []
    if all_pairs:
        all_pairs.sort(key=lambda x: (x['from_idx'], x['to_idx']))
        current_event = dict(all_pairs[0])

        for ev in all_pairs[1:]:
            same_type = ev['type'] == current_event['type']
            from_gap = ev['from'] - current_event['from']
            to_gap = ev['to'] - current_event['to']

            if same_type and from_gap <= from_merge_gap and to_gap <= merge_gap:
                current_event['to'] = max(current_event['to'], ev['to'])
                current_event['to_idx'] = max(current_event['to_idx'], ev['to_idx'])
                current_event['from'] = min(current_event['from'], ev['from'])
                current_event['from_idx'] = min(current_event['from_idx'], ev['from_idx'])
                current_event['feature_drift'] = min(current_event['feature_drift'], ev['feature_drift'])
                current_event['drift'] = min(current_event['drift'], ev['drift'])
            else:
                total_duration = current_event['to'] - current_event['from']
                avg_duration = (current_event['duration_prev'] + current_event['duration_curr']) / 2
                current_event['total_duration'] = total_duration
                current_event['avg_duration'] = avg_duration
                merged_events.append(current_event)
                current_event = dict(ev)

        total_duration = current_event['to'] - current_event['from']
        avg_duration = (current_event['duration_prev'] + current_event['duration_curr']) / 2
        current_event['total_duration'] = total_duration
        current_event['avg_duration'] = avg_duration
        merged_events.append(current_event)

    return merged_events, all_pairs

def get_frame_embeddings(frames, model, processor, device="mps", batch_size=8):
    """
    Given a list of PIL images, return their normalized embeddings from CLIP.
    """
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for img in frames:
            inputs = processor(images=img, return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs)
            embeddings = outputs / outputs.norm(dim=-1, keepdim=True)  # normalize
            all_embeddings.append(embeddings.cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)  # shape (N_frames, embedding_dim)
    return all_embeddings

from typing import List, Dict

def analyze_video_file(file_path, window_size=1, hop_size=1, eps=0.3, min_samples=2):
    
    # Extract features
    feature_vectors = get_frame_embeddings(
        extract_frames(file_path, frame_rate=1),
        model=model,
        processor=processor,
        device=device
    )
    feature_vectors = np.array(feature_vectors)
    
    frame_rate = 1  # Assuming frame_rate=1 fps here; adjust as needed
    hop_duration = 1.0 / frame_rate
    
    # Compute FVI (optional, as per your original code)
    fv_array = np.array(feature_vectors)
    fvi = np.mean(np.abs(np.diff(fv_array, axis=0)), axis=0) / (np.std(fv_array, axis=0) + 1e-6)
    fvi_normalized = fvi / np.sum(fvi)
    weighted_features = fv_array * fvi_normalized
    
    # Change scores and ruptures
    change_scores = [np.linalg.norm(weighted_features[i+1] - weighted_features[i]) for i in range(len(weighted_features)-1)]
    threshold = np.mean(change_scores) + np.std(change_scores) * 0.5
    ruptures = [i for i, score in enumerate(change_scores) if score > threshold]
    
    # Segment features
    segments = []
    start_idx = 0
    for rupture_idx in ruptures + [len(feature_vectors)-1]:
        segment = weighted_features[start_idx:rupture_idx+1]
        avg_feature = np.mean(segment, axis=0)
        segments.append((start_idx, rupture_idx+1, avg_feature))
        start_idx = rupture_idx+1
    
    segment_features = [seg[2] for seg in segments]
    
    # Clustering segments
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine').fit(segment_features)
    labels = clustering.labels_
    
    avg_segment_duration = np.mean([(end - start) for start, end, _ in segments])
    rupture_threshold = max(1.5, avg_segment_duration * 0.75)
    
    segment_annotations = []
    for idx, (start, end, _) in enumerate(segments):
        duration = (end - start) * hop_duration
        label = labels[idx]
        if duration < rupture_threshold and (idx == 0 or idx in ruptures):
            seg_type = 'rupture'
        elif label == -1:
            seg_type = 'unique'
        else:
            seg_type = 'motif'
        segment_annotations.append({
            'start': start * hop_duration,
            'end': end * hop_duration,
            'type': seg_type,
            'label': label
        })

    # Scores calculation
    n_segments = len(segment_annotations)
    n_motifs = sum(1 for s in segment_annotations if s['type'] == 'motif')
    n_uniques = sum(1 for s in segment_annotations if s['type'] == 'unique')
    n_ruptures = sum(1 for s in segment_annotations if s['type'] == 'rupture')

    score_linear = 1 - (n_ruptures + n_uniques) / n_segments
    score_cyclical = n_motifs / n_segments
    score_recursive = len(set(l['label'] for l in segment_annotations if l['label'] != -1)) / n_segments
    score_hybrid = 1.0 - min(score_linear, score_cyclical)

    # Calculate time threshold for recursion detection
    time_threshold = max(2.0, avg_segment_duration * 2)

    # Use new detect_video_recursion to get merged recursive events and all pairs
    recursive_events, all_pairs = detect_video_recursion(
        frame_embeddings=feature_vectors,
        hop_duration=hop_duration,
        time_threshold=time_threshold,
        feature_drift_threshold=0.3,
        merge_gap=1.0,
        from_merge_gap=1.0
    )

    # Compute recursive_drift_score as average drift of merged events
    recursive_drift_score = np.mean([ev['drift'] for ev in recursive_events]) if recursive_events else 0.0

    return {
        'file': os.path.basename(file_path),
        'segments': n_segments,
        'segment_annotations': segment_annotations,
        'motifs': n_motifs,
        'uniques': n_uniques,
        'ruptures': n_ruptures,
        'score_linear': round(score_linear, 3),
        'score_cyclical': round(score_cyclical, 3),
        'score_recursive': round(score_recursive, 3),
        'score_recursive_drift': round(recursive_drift_score, 3),
        'score_hybrid': round(score_hybrid, 3),
        'recursive_events': recursive_events,
        'all_recursive_pairs': all_pairs  # Optional, for debugging if needed
    }


