# app.py — updated for RoBERTa (6 labels) + optional hiding of unconfigured sections
import os
import re
import html as _html
import gradio as gr
import numpy as np
import pandas as pd
import traceback

import model as mymodel  # your updated RoBERTa-only model.py (6 labels)

# Build section choices from configured checkpoints (hide ones without a valid path)
def _available_sections():
    secs = []
    for section, cfg in mymodel.SECTION_CONFIG.items():
        p = cfg.get("BEST_CKPT")
        if p and os.path.isdir(p):
            secs.append(section)
    # If none are found (e.g., first run on a new machine), fall back to all
    return secs or ["Abstract", "Introduction", "Conclusion"]

SECTIONS = _available_sections()
DEFAULT_SECTION = SECTIONS[0]

LABEL_TO_BINARY = mymodel.LABEL_TO_BINARY  # 6-label mapping (0=Human, 1..5=AI)

CSS = """
body {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    background: white !important;
}
.gradio-container { background: white !important; }
.primary-btn {
    background-color: #8a4431 !important;
    border-color: #8a4431 !important;
    color: #fff !important;
    transition: background 0.2s;
}
.primary-btn:hover { background-color: #6c3323 !important; border-color: #6c3323 !important; }
.primary-btn:active { background-color: #a85a3a !important; border-color: #a85a3a !important; }
textarea { background-color: white !important; color: black !important; }
.gr-input, .gr-text-input, .gr-text-area { color: black !important; }
::placeholder { color: #555 !important; opacity: 1 !important; }
/* Spinner only, no background */
.spinner {
    display: block; margin: 0 auto;
    border: 3px solid #f3f3f3; border-top: 3px solid #8a4431;
    border-radius: 50%; width: 32px; height: 32px;
    animation: spin 1s linear infinite; background: transparent !important;
}
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
.logo-row, .logo-row .gr-panel { background: white !important; box-shadow: none !important; border-radius: 12px !important; }
.logo-row .gr-box { background: white !important; }
/* Custom header for class probabilities */
#probs-header {
    margin-top: 16px; margin-bottom: 0; text-align: left; color: #fff;
    background: #2c2c2f; padding: 8px 16px; border-radius: 8px 8px 0 0; font-size: 1.1em;
}
/* Custom Markdown/HTML table styling */
.markdown-table-container { width: 100%; display: flex; justify-content: center; margin-bottom: 0; }
.markdown-table-container table {
    width: 100%; border-collapse: separate; border-spacing: 0; background: #2c2c2f;
    border-radius: 0 0 12px 12px; overflow: hidden; margin-bottom: 0;
}
.markdown-table-container th, .markdown-table-container td {
    padding: 12px 18px; text-align: left; border-bottom: 1px solid #444; color: #fff; font-size: 1.08em;
}
.markdown-table-container th {
    background: #232326; font-weight: bold; font-size: 1.15em; border-bottom: 2px solid #8a4431;
}
.markdown-table-container tr:last-child td { border-bottom: none; }
.markdown-table-container td { background: #2c2c2f; }
.markdown-table-container tr:hover td { background: #393940; }

/* --- Chunking preview styles (high contrast) --- */
.chunk-legend { display:flex; gap:12px; align-items:center; margin:8px 0 6px; }
.legend-box { width:16px; height:16px; border-radius:4px; display:inline-block; vertical-align:middle; }
.legend-label { color:#e6e6e6; font-size:0.95em; }

.chunk-wrap {
  padding:10px 12px;
  border:1px solid #2a2a2a;
  border-radius:10px;
  background:#111;          /* dark background */
  color:#fff;               /* WHITE text inside preview */
  line-height:1.6;
  max-height:320px;
  overflow:auto;
  white-space:pre-wrap;
  word-wrap:break-word;
}

/* Alternating base colors for non-overlap chunk areas (darker, readable with white text) */
.c0 { background:#263b4d; }  /* deep blue */
.c1 { background:#2f4a36; }  /* deep green */
.c2 { background:#4a3a2a; }  /* deep brown/orange */
.c3 { background:#3b2f4d; }  /* deep purple */
.c4 { background:#4d2f42; }  /* deep magenta */
.c5 { background:#2f464c; }  /* deep teal */

/* Overlap segments: bright highlight + subtle outline, white text stays readable */
.overlap { background:#8a4431; outline:1px solid #a85a3a; color:#fff; }

/* Small chunk index tag */
.badge {
  display:inline-block;
  font-size:0.72em;
  color:#fff;               /* WHITE text on badge */
  background:#29486b;
  padding:1px 5px;
  border-radius:6px;
  margin-right:4px;
}

/* --- Make textbox text white too --- */
textarea, .gr-textbox textarea, .gr-text-area textarea {
  color:#fff !important;         /* WHITE text inside the input box */
  background:#1a1a1a !important; /* darker background so white text is readable */
}

/* Optional: lighter placeholder on dark background */
::placeholder {
  color:#bbbbbb !important;
  opacity:1 !important;
}

"""

def _empty():
    return (
        gr.update(visible=False),  # status
        gr.update(visible=False),  # results group
        "", "", None, ""           # outputs
    )

def _spinner():
    return (
        gr.update(visible=True, value="<div class='spinner' style='margin:20px auto;display:block;'></div>"),
        gr.update(visible=False),
        "", "", None, ""
    )

def _error(msg):
    return (
        gr.update(visible=True, value=f"<div class='status-message error'>⚠️ {msg}</div>"),
        gr.update(visible=False),
        "", "", None, ""
    )

# ---------- Chunking preview + analysis helpers ----------
def _tokenize_words_with_spaces(text: str):
    return re.split(r'(\s+)', text)

def _word_index_map(tokens):
    tok_to_word = {}
    w = 0
    for i, t in enumerate(tokens):
        if t.strip():
            tok_to_word[i] = w
            w += 1
    return tok_to_word, w  # total words

def _coverage_arrays(total_words: int, window: int, stride: int):
    starts = list(range(0, total_words, stride))
    coverage = [0] * total_words
    first_chunk_for_word = [-1] * total_words
    for ci, st in enumerate(starts):
        ed = min(total_words, st + window)
        if st >= ed:
            continue
        for w in range(st, ed):
            coverage[w] += 1
            if first_chunk_for_word[w] == -1:
                first_chunk_for_word[w] = ci
    return starts, coverage, first_chunk_for_word

def build_chunk_preview_html(text: str, window_words: int, stride_words: int) -> str:
    txt = (text or "").strip()
    if not txt:
        return "<div class='chunk-wrap'>(paste text to preview chunking)</div>"

    tokens = _tokenize_words_with_spaces(txt)
    tok_to_word, total_words = _word_index_map(tokens)
    if total_words == 0:
        return "<div class='chunk-wrap'>(no words)</div>"

    window = max(1, int(window_words))
    stride = max(1, int(stride_words))
    starts, coverage, first_chunk_for_word = _coverage_arrays(total_words, window, stride)
    chunk_word_starts = set(starts)

    out = []
    out.append("<div class='chunk-legend'>")
    out.append("<span class='legend-box' style='background:#e9f3ff;'></span><span class='legend-label'>Chunk window (non-overlap)</span>")
    out.append("<span class='legend-box' style='background:#ffd9d9; outline:1px solid #ffb3b3;'></span><span class='legend-label'>Overlap (window − stride)</span>")
    out.append("</div>")

    out.append("<div class='chunk-wrap'>")
    current_w = -1
    for i, tok in enumerate(tokens):
        esc = _html.escape(tok)
        if tok.strip():  # word
            current_w += 1
            if current_w in chunk_word_starts:
                ci = starts.index(current_w)
                out.append(f"<span class='badge'>chunk {ci}</span>")
            if coverage[current_w] >= 2:
                cls = "overlap"
            else:
                ci = first_chunk_for_word[current_w]
                cls = f"c{ci % 6}" if ci >= 0 else "c0"
            out.append(f"<span class='{cls}'>{esc}</span>")
        else:
            out.append(esc)
    out.append("</div>")
    return "".join(out)

def build_chunk_analysis_table(text: str, window_words: int, stride_words: int) -> pd.DataFrame:
    txt = (text or "").strip()
    if not txt:
        return pd.DataFrame(columns=["chunk", "start_word", "end_word", "words_in_chunk", "unique_words", "overlap_words"])

    tokens = _tokenize_words_with_spaces(txt)
    tok_to_word, total_words = _word_index_map(tokens)
    if total_words == 0:
        return pd.DataFrame(columns=["chunk", "start_word", "end_word", "words_in_chunk", "unique_words", "overlap_words"])

    window = max(1, int(window_words))
    stride = max(1, int(stride_words))
    starts, coverage, _first = _coverage_arrays(total_words, window, stride)

    rows = []
    for ci, st in enumerate(starts):
        ed = min(total_words, st + window)
        if st >= ed:
            continue
        words_in_chunk = ed - st
        # unique vs overlap per chunk measured on coverage
        unique = sum(1 for w in range(st, ed) if coverage[w] == 1)
        overlap = words_in_chunk - unique
        rows.append({
            "chunk": ci,
            "start_word": st,
            "end_word": ed,  # exclusive
            "words_in_chunk": words_in_chunk,
            "unique_words": unique,
            "overlap_words": overlap
        })
    return pd.DataFrame(rows)

def update_preview_enabled(enable, text, window_words, stride_words):
    """Toggle visibility & content based on checkbox."""
    if not enable:
        return (
            gr.update(visible=False),  # preview HTML
            gr.update(visible=False),  # analysis table
        )
    # enabled: compute both
    html = build_chunk_preview_html(text or "", int(window_words), int(stride_words))
    df = build_chunk_analysis_table(text or "", int(window_words), int(stride_words))
    # Render table as HTML to keep your dark table style consistent
    df_html = f'<div class="markdown-table-container">{df.to_html(index=False, border=0)}</div>'
    return (
        gr.update(visible=True, value=html),
        gr.update(visible=True, value=df_html),
    )

def update_preview_live(enable, text, window_words, stride_words):
    """Live updates when enabled; otherwise do nothing visible."""
    if not enable:
        return gr.update(), gr.update()
    html = build_chunk_preview_html(text or "", int(window_words), int(stride_words))
    df = build_chunk_analysis_table(text or "", int(window_words), int(stride_words))
    df_html = f'<div class="markdown-table-container">{df.to_html(index=False, border=0)}</div>'
    return gr.update(value=html), gr.update(value=df_html)

# ---------- Main analyze ----------
def analyze(section, text, window_words, stride_words):
    try:
        txt = (text or "").strip()
        if not txt:
            yield _empty()
            return

        yield _spinner()

        # Load assets (will raise a clear ValueError if Conclusion path is still a placeholder)
        assets = mymodel.get_assets(section)

        labels_human = assets["labels_human"]
        ww = int(window_words)
        sw = int(stride_words)

        probs, pred_idx, pred_label_str, html_obj, tokens, contrib = mymodel.shap_for_pred_class(
            txt, assets, window_words=ww, stride_words=sw
        )
        pred_binary = LABEL_TO_BINARY.get(pred_idx, "Unknown")

        # Format probabilities as percentages
        pct = (np.array(probs, dtype=float) * 100.0).round(2)
        df = pd.DataFrame({
            "Label": labels_human,
            "Probability": [f"{p:.2f}%" for p in pct]
        })
        df_html = f'<div class="markdown-table-container">{df.to_html(index=False, border=0)}</div>'

        # SHAP HTML
        html_plot = getattr(html_obj, "data", html_obj if isinstance(html_obj, str) else "<div>SHAP output unavailable.</div>")
        predicted_label_field = f"{pred_label_str}"

        yield (
            gr.update(visible=False),
            gr.update(visible=True),
            predicted_label_field,
            pred_binary,
            df_html,
            html_plot
        )

    except Exception as e:
        msg = str(e)
        if "Checkpoint path for section" in msg:
            yield _error(msg + " (Set the correct BEST_CKPT_* path or update model.py.)")
        else:
            err = traceback.format_exc(limit=1).splitlines()[-1]
            yield _error(f"Something went wrong: {err}")

def on_clear():
    return (*_empty(), gr.update(value=""))

def on_section_change(_):
    return (*_empty(), gr.update(value=""))

with gr.Blocks(title="AI Detector", css=CSS, theme=gr.themes.Default()) as demo:
    with gr.Row(elem_classes=["logo-row"]):
        gr.Image(
            value="/Users/sm/Developer/Master_V2/8. UI & SHAP/static/ecis_2025.png",
            label=None, show_label=False, height=140, container=False, width=600,
            show_download_button=False, show_fullscreen_button=False, elem_id="ecis-logo"
        )
        gr.Image(
            value="/Users/sm/Developer/Master_V2/8. UI & SHAP/static/ais_logo.png",
            label=None, show_label=False, height=140, container=False, width=600,
            show_download_button=False, show_fullscreen_button=False, elem_id="ais-logo"
        )

    gr.Markdown(
        """
        <div style="text-align:center;">
            <h2 style="margin:14px 0 0 0; color:#29486b;">AIS AI-Generated Section Detector</h2>
            <div style="font-size:1.1em; color:#8a4431; margin-top:6px;">
                Detects AI in
                <b style="color:#29486b;">Abstract</b>,
                <b style="color:#29486b;">Introduction</b>,
                and <b style="color:#29486b;">Conclusion</b>
                sections of academic papers.
            </div>
        </div>
        """
    )

    section_sel = gr.Radio(choices=SECTIONS, value=DEFAULT_SECTION, label="Paper section")

    # Hidden chunk params (kept for internal use) --- not visible to users
    chunk_window = gr.State(int(os.environ.get("CHUNK_WINDOW_WORDS", "300")))
    chunk_stride = gr.State(int(os.environ.get("CHUNK_STRIDE_WORDS", "170")))

    text_in = gr.Textbox(
        lines=10,
        label="Paste your text",
        placeholder="Paste a section."
    )

    with gr.Row():
        btn = gr.Button("Analyze", variant="primary", elem_classes=["primary-btn"])
        clear_btn = gr.Button("Clear")

    status = gr.HTML(visible=True)

    with gr.Group(visible=False) as results_group:
        with gr.Row():
            pred_label = gr.Textbox(label="Predicted label")
            verdict = gr.Textbox(label='Binary verdict ("AI" or "Human")')
        gr.Markdown("### Class probabilities", elem_id="probs-header")
        probs_table = gr.HTML(label=None)  # Use HTML for full control
        shap_html = gr.HTML(label="SHAP Explanation")

    # Predict
    btn.click(
        analyze,
        inputs=[section_sel, text_in, chunk_window, chunk_stride],
        outputs=[status, results_group, pred_label, verdict, probs_table, shap_html],
        queue=True
    )

    # Clear
    clear_btn.click(
        on_clear,
        inputs=[],
        outputs=[status, results_group, pred_label, verdict, probs_table, shap_html, text_in]
    )

    # Section change
    section_sel.change(
        on_section_change,
        inputs=[section_sel],
        outputs=[status, results_group, pred_label, verdict, probs_table, shap_html, text_in]
    )

    # (chunk preview UI removed)

if __name__ == "__main__":
    demo.launch(share=True)
