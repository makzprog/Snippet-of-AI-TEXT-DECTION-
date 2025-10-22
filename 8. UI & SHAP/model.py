# model.py — RoBERTa (6 labels) WITH CHUNKING, length-weighted aggregation + fast path
import os
from typing import Dict, Any, Tuple, List, Optional, Union

import numpy as np
import torch
import shap
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification

# Cleaner (used only for Intro/Conclusion)
from citations_cleaner import strip_citations  # strict=True default

# ---------------------- Sections & checkpoints ----------------------
SECTION_CONFIG = {
    "Abstract": {
        "BEST_CKPT": os.environ.get(
            "BEST_CKPT_ABS",
            "/Users/sm/Developer/Master_V2/8. UI & SHAP/Checkpoints/Abstract/checkpoint-1750"
        ),
        "MODEL_NAME": os.environ.get("MODEL_NAME_ABS", "roberta-base"),
    },
    "Introduction": {
        "BEST_CKPT": os.environ.get(
            "BEST_CKPT_INTRO",
            "/Users/sm/Developer/Master_V2/8. UI & SHAP/Checkpoints/Introduction/checkpoint-2000"
        ),
        "MODEL_NAME": os.environ.get("MODEL_NAME_INTRO", "roberta-base"),
    },
    "Conclusion": {
        "BEST_CKPT": os.environ.get(
            "BEST_CKPT_CONC",
            "/Users/sm/Developer/Master_V2/8. UI & SHAP/Checkpoints/Conclusion/checkpoint-2250"  # placeholder
        ),
        "MODEL_NAME": os.environ.get("MODEL_NAME_CONC", "roberta-base"),
    },
}

# ---- Fixed to 6 labels ----
LABELS = ["Human (0)", "AI (1)", "AI 80% (2)", "AI 60% (3)", "AI 40% (4)", "AI 20% (5)"]
NUM_LABELS = 6
MAX_LEN = 512
LABEL_TO_BINARY = {0: "Human", 1: "AI", 2: "AI", 3: "AI", 4: "AI", 5: "AI"}

# ---- Chunking + aggregation knobs (default/global) ----
CHUNK_WINDOW_WORDS = int(os.environ.get("CHUNK_WINDOW_WORDS", "300"))
CHUNK_STRIDE_WORDS = int(os.environ.get("CHUNK_STRIDE_WORDS", "170"))

# If the processed text token length <= MAX_LEN, skip chunking entirely
FASTPATH_IF_LEQ_MAXLEN = os.environ.get("FASTPATH_IF_LEQ_MAXLEN", "1").lower() in {"1", "true", "yes"}

# Drop very tiny chunks (by token length) to avoid diluting with short tails (Intro/Conclusion path)
MIN_CHUNK_TOKENS = int(os.environ.get("MIN_CHUNK_TOKENS", "96"))  # ~1/5 of 512
LENGTH_WEIGHTED = os.environ.get("LENGTH_WEIGHTED", "1").lower() in {"1", "true", "yes"}

# CPU only (safe defaults)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
device = torch.device("cpu")
torch.set_num_threads(max(1, torch.get_num_threads()))

# Cache per (section + checkpoint)
_ASSET_CACHE: Dict[str, Dict[str, Any]] = {}

# ============================== Helpers ==============================
def get_split(text: str, window: int = CHUNK_WINDOW_WORDS, stride: int = CHUNK_STRIDE_WORDS) -> List[str]:
    words = str(text).split()
    if not words:
        return [""]
    chunks = [" ".join(words[:window])]
    start = stride
    while start < len(words):
        end = start + window
        part = words[start:end]
        if not part:
            break
        chunks.append(" ".join(part))
        start += stride
    return chunks

def _safe_first_length(length_field: Union[int, List[int], np.ndarray]) -> int:
    if isinstance(length_field, (list, tuple, np.ndarray)):
        return int(length_field[0])
    return int(length_field)

def _token_len(tokenizer: RobertaTokenizerFast, s: str) -> int:
    # Count tokens the same way we will encode (includes special tokens)
    enc = tokenizer(s, truncation=False, padding=False, add_special_tokens=True, return_length=True)
    return _safe_first_length(enc["length"])

def _split_and_measure_intro_conc(
    tokenizer: RobertaTokenizerFast,
    text: str,
    window_words: Optional[int] = None,
    stride_words: Optional[int] = None
):
    """Split + drop tiny chunks (Intro/Conclusion behavior)."""
    w = window_words or CHUNK_WINDOW_WORDS
    s = stride_words or CHUNK_STRIDE_WORDS
    chunks = get_split(text, window=w, stride=s)
    lens = [_token_len(tokenizer, c) for c in chunks]
    keep = [(l >= MIN_CHUNK_TOKENS) for l in lens]
    if any(keep):
        chunks = [c for c, k in zip(chunks, keep) if k]
        lens = [l for l, k in zip(lens, keep) if k]
    return chunks, lens

def _split_no_filter(
    tokenizer: RobertaTokenizerFast,
    text: str,
    window_words: Optional[int] = None,
    stride_words: Optional[int] = None
):
    """Split without filtering (ABSTRACT_ALL_MIXET behavior)."""
    w = window_words or CHUNK_WINDOW_WORDS
    s = stride_words or CHUNK_STRIDE_WORDS
    chunks = get_split(text, window=w, stride=s)
    lens = [_token_len(tokenizer, c) for c in chunks]
    return chunks, lens

def _ensure_eval(model: torch.nn.Module) -> None:
    model.eval()

def _predict_logits_for_texts(tokenizer, model, text_list: List[str]) -> np.ndarray:
    if len(text_list) == 0:
        return np.zeros((0, NUM_LABELS), dtype=np.float32)
    # Match notebook encoding for Abstracts; safe for others, too
    enc = tokenizer(
        list(text_list),
        return_tensors="pt",
        truncation=True,
        padding=True,        # NOTE: notebook uses padding=True (longest)
        max_length=MAX_LEN,
        add_special_tokens=True
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    _ensure_eval(model)  # keep dropout off, always
    with torch.inference_mode():
        logits = model(**enc).logits  # [N, C]
    return logits.cpu().numpy().astype(np.float32)

# ============================== Builders ==============================
def _build_assets_for_section(section: str) -> Dict[str, Any]:
    cfg = SECTION_CONFIG[section]

    # Resolve checkpoint & tokenizer — tokenizer from ckpt first (matches notebook)
    best_ckpt = cfg["BEST_CKPT"]
    if not best_ckpt or not os.path.isdir(best_ckpt):
        raise ValueError(
            f"Checkpoint path for section '{section}' not found at '{best_ckpt}'. "
            f"Please set the correct path or the env var BEST_CKPT_{section.upper()}."
        )
    try:
        tokenizer = RobertaTokenizerFast.from_pretrained(best_ckpt)
    except Exception:
        tokenizer = RobertaTokenizerFast.from_pretrained(cfg["MODEL_NAME"])

    model = RobertaForSequenceClassification.from_pretrained(best_ckpt, num_labels=NUM_LABELS)
    model.to(device)
    _ensure_eval(model)

    # Section-specific preprocessing/aggregation:
    IS_ABSTRACT = (section == "Abstract")
    CLEAN_THIS_SECTION = not IS_ABSTRACT  # clean only for Intro/Conclusion

    # SHAP over CHUNKS (uses the same tokenizer)
    masker = shap.maskers.Text(tokenizer)

    def predict_proba_chunk(text_list: List[str]) -> np.ndarray:
        logits = _predict_logits_for_texts(tokenizer, model, text_list)
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        return probs

    def predict_proba(
        text_list: List[str],
        window_words: Optional[int] = None,
        stride_words: Optional[int] = None
    ) -> np.ndarray:
        """
        Document-level predictor.

        Abstract  :  no cleaning, no tiny-chunk filtering, uniform mean over chunks (ABSTRACT_ALL_MIXET).
        Intro/Conc:  strict cleaning, drop tiny chunks, length-weighted mean (original behavior).
        Fast path (single pass) is used when processed token length <= 512 in both cases.
        """
        out = np.zeros((len(text_list), NUM_LABELS), dtype=np.float32)

        for i, t in enumerate(text_list):
            t_proc = strip_citations(t, strict=True) if CLEAN_THIS_SECTION else t

            # Lock fast-path ON for Abstracts to match predictor; respect env for other sections
            FASTPATH_LOCAL = True if IS_ABSTRACT else FASTPATH_IF_LEQ_MAXLEN
            if FASTPATH_LOCAL and _token_len(tokenizer, t_proc) <= MAX_LEN:
                logits = _predict_logits_for_texts(tokenizer, model, [t_proc])
                out[i] = torch.softmax(torch.tensor(logits[0]), dim=-1).numpy()
                continue

            # Long text → chunk
            if IS_ABSTRACT:
                # ABSTRACT_ALL_MIXET behavior: keep all chunks, no length-weighting
                chunks, lens = _split_no_filter(tokenizer, t_proc, window_words, stride_words)
                if len(chunks) == 0:
                    out[i] = np.full(NUM_LABELS, 1.0 / NUM_LABELS, dtype=np.float32)
                    continue

                probs_k = predict_proba_chunk(chunks)  # [K, C]
                out[i] = probs_k.mean(axis=0)          # UNIFORM MEAN (no 1e-12, no weights)

            else:
                # Original behavior for Intro/Conclusion
                chunks, lens = _split_and_measure_intro_conc(tokenizer, t_proc, window_words, stride_words)
                if len(chunks) == 0:
                    out[i] = np.full(NUM_LABELS, 1.0 / NUM_LABELS, dtype=np.float32)
                    continue

                probs_k = predict_proba_chunk(chunks)  # [K, C]
                if LENGTH_WEIGHTED and len(chunks) > 1:
                    w = np.array(lens, dtype=np.float32)
                    w = w / (w.sum() + 1e-12)         # epsilon only for weighted path
                    out[i] = (probs_k * w[:, None]).sum(axis=0)
                else:
                    out[i] = probs_k.mean(axis=0)

        return out

    explainer = shap.Explainer(predict_proba_chunk, masker, output_names=LABELS)

    return {
        "section": section,
        "ckpt_path": best_ckpt,          # <— record the exact checkpoint used
        "tokenizer": tokenizer,
        "model": model,
        "explainer": explainer,          # chunk-level explainer
        "predict_proba": predict_proba,  # document-level predictor
        "predict_proba_chunk": predict_proba_chunk,
        "labels_human": LABELS,
        "is_abstract": IS_ABSTRACT,
        "clean_this_section": CLEAN_THIS_SECTION,
    }

def get_assets(section: str) -> Dict[str, Any]:
    if section not in SECTION_CONFIG:
        raise ValueError(f"Unknown section '{section}'. Valid: {list(SECTION_CONFIG.keys())}")

    cfg = SECTION_CONFIG[section]
    ckpt = cfg["BEST_CKPT"]
    cache_key = f"{section}::{ckpt}"

    # If the same section was loaded from a different ckpt earlier, don't reuse it
    if cache_key not in _ASSET_CACHE:
        _ASSET_CACHE[cache_key] = _build_assets_for_section(section)
    return _ASSET_CACHE[cache_key]

# ============================== Inference APIs ==============================
def predict(
    text: str,
    assets: Dict[str, Any],
    window_words: Optional[int] = None,
    stride_words: Optional[int] = None
) -> Tuple[np.ndarray, int, str]:
    probs = assets["predict_proba"]([text], window_words=window_words, stride_words=stride_words)[0]
    pred_idx = int(np.argmax(probs))
    pred_label_str = assets["labels_human"][pred_idx]
    return probs, pred_idx, pred_label_str

def shap_for_pred_class(
    text: str,
    assets: Dict[str, Any],
    window_words: Optional[int] = None,
    stride_words: Optional[int] = None
):
    """
    Compute SHAP for the predicted class on the most-supportive chunk.
    Uses the same conditional preprocessing and selection policy as prediction.
    """
    HIDE_NEGATIVE = True

    # --- doc-level prediction (already applies section-specific preprocessing) ---
    probs_doc = assets["predict_proba"]([text], window_words=window_words, stride_words=stride_words)[0]
    pred_idx = int(np.argmax(probs_doc))
    pred_label_str = assets["labels_human"][pred_idx]

    # --- choose chunk using the same preprocessing policy ---
    tokenizer = assets["tokenizer"]
    is_abstract = assets.get("is_abstract", False)
    clean_this_section = assets.get("clean_this_section", not is_abstract)

    text_proc = strip_citations(text, strict=True) if clean_this_section else text

    if is_abstract:
        chunks, lens = _split_no_filter(tokenizer, text_proc, window_words, stride_words)
    else:
        chunks, lens = _split_and_measure_intro_conc(tokenizer, text_proc, window_words, stride_words)

    if len(chunks) == 0:
        chunks, lens = [text_proc], [_token_len(tokenizer, text_proc)]

    p_chunks = assets["predict_proba_chunk"](chunks)  # [K, C]

    if len(chunks) == 1:
        chosen_idx = 0
    else:
        if is_abstract:
            # ABSTRACT_ALL_MIXET: choose chunk with highest prob for the predicted class (uniform policy)
            chosen_idx = int(np.argmax(p_chunks[:, pred_idx]))
        else:
            # original: length-weighted support
            w = np.array(lens, dtype=np.float32); w = w / (w.sum() + 1e-12)
            chosen_idx = int(np.argmax(w * p_chunks[:, pred_idx]))

    chosen_chunk = chunks[chosen_idx]

    # --- SHAP on chosen chunk ---
    sv = assets["explainer"]([chosen_chunk], fixed_context=1)  # shape: [1, tokens, classes]
    sv0 = sv[0]  # token x class Explanation
    all_values = sv0.values
    token_values_for_class = all_values[:, pred_idx]

    if HIDE_NEGATIVE:
        token_values_for_class = np.where(token_values_for_class > 0, token_values_for_class, 0.0)

    token_data = getattr(sv, "data", None)
    if token_data is not None:
        token_data = token_data[0]

    base_vals = getattr(sv0, "base_values", None)
    base_val_for_class = None
    if base_vals is not None:
        try:
            base_val_for_class = float(base_vals[pred_idx]) if np.ndim(base_vals) > 0 else float(base_vals)
        except Exception:
            base_val_for_class = None

    red_only_exp = shap.Explanation(
        values=token_values_for_class,
        base_values=base_val_for_class,
        data=token_data,
        feature_names=getattr(sv0, "feature_names", None)
    )

    html = shap.plots.text(red_only_exp, display=False)

    if token_data is None:
        import re
        tokens = re.split(r"(\s+)", chosen_chunk)
    else:
        tokens = token_data

    contrib = token_values_for_class
    return probs_doc, pred_idx, pred_label_str, html, tokens, contrib
