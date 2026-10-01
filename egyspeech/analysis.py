"""Dataset analysis: statistics + Plotly figures (used by the analysis step and notebook)."""

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from egyspeech.config import Section
from egyspeech.io import iter_jsonl_dir, read_json, read_jsonl
from egyspeech.paths import Layout
from egyspeech.text import latin_words, strip_tags

TEMPLATE = "plotly_white"
COLORS = ["#2563eb", "#f97316", "#10b981", "#e11d48", "#8b5cf6", "#0ea5e9", "#eab308", "#64748b"]
AR_STOP = set(
    "في من على الى إلى عن ان إن أن اللي ده دي دا دى كان كانت هو هي هما احنا انا انت انتي انتو و ف يا ما مش "
    "لا ايه إيه ازاي كده كدا بقى بس يعني او أو مع لو كل عشان علشان لما حاجة حاجه كمان برضه برده هنا هناك "
    "اه آه ايوه طب طيب تاني بعد قبل لحد عند عنده عندي فيه فيها منه منها بيه بيها ليه ليها له لها اني انه "
    "انها اللى الي دول دلوقتي النهارده جدا اوي قوي اكتر اكثر زي مثلا هو ليه the a to and of is in it you that".split()
)


def gini(values: list[float]) -> float:
    x = np.sort(np.asarray(values, dtype=float))
    if len(x) == 0 or x.sum() == 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load_frames(cfg: Section) -> dict[str, pd.DataFrame]:
    lay = Layout(cfg.work_dir)
    final = []
    for split in ("train", "validation", "test"):
        for r in read_jsonl(lay.split(split)):
            r["split"] = split
            final.append(r)
    segmented = list(iter_jsonl_dir(lay.meta / "chunks"))
    quality = {r["id"]: r for r in iter_jsonl_dir(lay.meta / "quality")}
    for r in segmented:
        r.update(quality.get(r["id"], {}))
    trans = list(iter_jsonl_dir(lay.meta / "transcripts" / cfg.transcription.backend))
    return {
        "final": pd.DataFrame(final),
        "segmented": pd.DataFrame(segmented),
        "filtered": pd.DataFrame(read_jsonl(lay.filtered)),
        "transcripts": pd.DataFrame(trans),
        "videos": pd.DataFrame(read_jsonl(lay.videos)),
    }


def _hours(df: pd.DataFrame) -> float:
    return float(df["duration"].sum() / 3600) if len(df) and "duration" in df else 0.0


def downloaded_hours(cfg: Section) -> float:
    return sum(v.get("duration") or 0 for v in read_jsonl(Layout(cfg.work_dir).videos)) / 3600


def build(cfg: Section) -> tuple[dict, dict]:
    """(figures by name, stats dict)."""
    import plotly.express as px
    import plotly.graph_objects as go

    lay = Layout(cfg.work_dir)
    d = load_frames(cfg)
    final, seg, filt, trans = d["final"], d["segmented"], d["filtered"], d["transcripts"]
    figs: dict[str, go.Figure] = {}
    stats: dict = {}
    if final.empty:
        raise SystemExit("final dataset is empty: run the balance step first")

    # ---- funnel
    balance = read_json(lay.final / "balance_summary.json")
    stages = [
        ("downloaded", downloaded_hours(cfg)),
        ("single-speaker clips", _hours(seg)),
        ("quality filter", _hours(filt)),
        ("transcribed + aligned", balance["candidate_hours"]),
        ("final (balanced)", _hours(final)),
    ]
    figs["funnel"] = go.Figure(go.Funnel(y=[s for s, _ in stages], x=[round(h, 1) for _, h in stages],
                                         textinfo="value+percent initial", marker={"color": COLORS[:5]}))
    figs["funnel"].update_layout(title="Pipeline funnel (hours)", template=TEMPLATE)

    # ---- durations
    figs["durations"] = px.histogram(final, x="duration", nbins=50, color="split", template=TEMPLATE,
                                     title="Clip duration (s)", color_discrete_sequence=COLORS)

    # ---- quality before / after
    q_cols = ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "utmos"]
    qa = pd.concat([seg.assign(stage="all clips")[q_cols + ["stage"]],
                    final.assign(stage="final")[q_cols + ["stage"]]]).melt(id_vars="stage", var_name="metric")
    figs["quality"] = px.box(qa, x="metric", y="value", color="stage", template=TEMPLATE,
                             title="Quality scores: all segmented clips vs final dataset",
                             color_discrete_sequence=COLORS)

    report = read_json(lay.filter_report)
    rej = pd.DataFrame(sorted(report["rejections"].items(), key=lambda kv: -kv[1]), columns=["reason", "clips"])
    figs["rejections"] = px.bar(rej, x="reason", y="clips", template=TEMPLATE, title="Quality-filter rejections",
                                color_discrete_sequence=COLORS)

    # ---- speakers
    spk_final = final.groupby("speaker_id")["duration"].sum().sort_values(ascending=False) / 3600
    speakers = read_json(lay.speakers)["speakers"]
    spk_before = pd.Series({s["id"]: s["hours"] for s in speakers}).sort_values(ascending=False)
    top = pd.DataFrame({"before balancing": spk_before.head(40), "final": spk_final.reindex(spk_before.head(40).index)})
    figs["speaker_hours"] = px.bar(top, barmode="group", template=TEMPLATE, color_discrete_sequence=COLORS,
                                   title="Hours per speaker (40 largest), before and after balancing",
                                   labels={"value": "hours", "index": "speaker"})
    lorenz = go.Figure()
    for name, series, color in (("before balancing", spk_before, COLORS[1]), ("final", spk_final, COLORS[0])):
        x = np.sort(series.values)
        cum = np.concatenate([[0], np.cumsum(x) / x.sum()]) if x.sum() else [0, 1]
        lorenz.add_trace(go.Scatter(x=np.linspace(0, 1, len(cum)), y=cum, name=f"{name} (Gini {gini(x):.2f})",
                                    line={"color": color}))
    lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="perfect balance", line={"dash": "dot", "color": "#94a3b8"}))
    lorenz.update_layout(title="Speaker balance (Lorenz curve)", xaxis_title="share of speakers",
                         yaxis_title="share of hours", template=TEMPLATE)
    figs["speaker_balance"] = lorenz
    figs["speakers_per_hours"] = px.histogram(spk_final.reset_index(name="hours"), x="hours", nbins=40,
                                              template=TEMPLATE, title="Distribution of hours per speaker (final)",
                                              color_discrete_sequence=COLORS)

    # ---- gender
    g = final.groupby("gender").agg(hours=("duration", lambda s: s.sum() / 3600),
                                    speakers=("speaker_id", "nunique")).reset_index()
    figs["gender"] = go.Figure()
    from plotly.subplots import make_subplots

    figs["gender"] = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                                   subplot_titles=["hours", "speakers"])
    figs["gender"].add_trace(go.Pie(labels=g["gender"], values=g["hours"], hole=0.45,
                                    marker={"colors": [COLORS[4], COLORS[0]]}), 1, 1)
    figs["gender"].add_trace(go.Pie(labels=g["gender"], values=g["speakers"], hole=0.45,
                                    marker={"colors": [COLORS[4], COLORS[0]]}), 1, 2)
    figs["gender"].update_layout(title="Gender distribution", template=TEMPLATE)

    # ---- code-switching
    cs = final.assign(kind=np.where(final["code_switched"], "Arabic + English (code-switched)", "pure Arabic"))
    cs_h = cs.groupby("kind")["duration"].sum() / 3600
    figs["code_switching"] = go.Figure(go.Pie(labels=cs_h.index, values=cs_h.values, hole=0.45,
                                              marker={"colors": [COLORS[1], COLORS[0]]}))
    figs["code_switching"].update_layout(title="Pure Arabic vs code-switched (hours)", template=TEMPLATE)
    eng = Counter(w.lower() for t in final["text"] for w in latin_words(t))
    eng_df = pd.DataFrame(eng.most_common(30), columns=["word", "count"])
    figs["english_words"] = px.bar(eng_df, x="count", y="word", orientation="h", template=TEMPLATE,
                                   title="Most frequent English words", color_discrete_sequence=COLORS)
    figs["english_words"].update_yaxes(autorange="reversed")

    # ---- tags
    stats["tags"] = {}
    if "tags" in final and final["tags"].notna().any():
        tag_counts = Counter(t for tags in final["tags"].dropna() for t in tags)
        tdf = pd.DataFrame(sorted(tag_counts.items(), key=lambda kv: -kv[1]), columns=["tag", "count"])
        tdf["per_hour"] = tdf["count"] / max(_hours(final), 1e-9)
        figs["tags"] = px.bar(tdf, x="tag", y="count", template=TEMPLATE, text="count",
                              title="Paralinguistic tag distribution", color_discrete_sequence=COLORS)
        stats["tags"] = dict(tag_counts)
        tagged = final["tags"].dropna().map(len)
        stats["clips_with_tags"] = int((tagged > 0).sum())

    # ---- speaking rate, words
    rate = trans[trans["ok"]]["chars_per_sec"] if len(trans) else pd.Series(dtype=float)
    figs["speaking_rate"] = px.histogram(rate, nbins=50, template=TEMPLATE, title="Speaking rate (letters / s)",
                                         color_discrete_sequence=COLORS)
    words = Counter(w for t in final["text"] for w in re.findall(r"[ء-ي]+", strip_tags(t))
                    if w not in AR_STOP and len(w) > 2)
    wdf = pd.DataFrame(words.most_common(40), columns=["word", "count"])
    figs["top_words"] = px.bar(wdf, x="count", y="word", orientation="h", template=TEMPLATE,
                               title="Most frequent content words", color_discrete_sequence=COLORS)
    figs["top_words"].update_yaxes(autorange="reversed")

    # ---- channels
    ch = final.groupby("channel")["duration"].sum().sort_values(ascending=False).head(40) / 3600
    figs["channels"] = px.treemap(ch.reset_index(name="hours"), path=["channel"], values="hours",
                                  title="Hours per channel (40 largest)", template=TEMPLATE)

    # ---- audio provenance / cuts
    src = final["source"].value_counts()
    figs["provenance"] = go.Figure(go.Pie(labels=["original audio" if s == "original" else "music removed"
                                                  for s in src.index], values=src.values, hole=0.45,
                                          marker={"colors": [COLORS[2], COLORS[3]]}))
    figs["provenance"].update_layout(title="Clip audio source", template=TEMPLATE)

    # ---- topics (semantic clusters)
    figs.update(topic_figures(final, cfg))

    stats.update({
        "hours": round(_hours(final), 2),
        "clips": int(len(final)),
        "speakers": int(final["speaker_id"].nunique()),
        "speakers_before_balancing": len(speakers),
        "videos": int(final["video_id"].nunique()),
        "channels": int(final["channel"].nunique()),
        "splits": {s: {"hours": round(_hours(final[final["split"] == s]), 2),
                       "clips": int((final["split"] == s).sum())} for s in ("train", "validation", "test")},
        "gender_hours": {r["gender"]: round(r["hours"], 2) for _, r in g.iterrows()},
        "gender_speakers": {r["gender"]: int(r["speakers"]) for _, r in g.iterrows()},
        "code_switched_hours": round(float(cs_h.get("Arabic + English (code-switched)", 0.0)), 2),
        "pure_arabic_hours": round(float(cs_h.get("pure Arabic", 0.0)), 2),
        "gini_before": round(gini(spk_before.values), 3),
        "gini_final": round(gini(spk_final.values), 3),
        "max_speaker_share": round(float(spk_final.max() / spk_final.sum()), 4),
        "duration_sec": {"mean": round(float(final["duration"].mean()), 2),
                         "median": round(float(final["duration"].median()), 2),
                         "min": round(float(final["duration"].min()), 2),
                         "max": round(float(final["duration"].max()), 2)},
        "quality_mean": {c: round(float(final[c].mean()), 3) for c in q_cols},
        "funnel_hours": {s: round(h, 2) for s, h in stages},
    })
    return figs, stats


def topic_figures(final: pd.DataFrame, cfg: Section) -> dict:
    """Semantic topics: multilingual sentence embeddings -> KMeans -> TF-IDF keywords."""
    import plotly.express as px
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [strip_tags(t) for t in final["text"]]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), size=min(len(texts), 20000), replace=False)
    sample = [texts[i] for i in idx]
    k = max(2, min(cfg.analysis.topic_clusters, len(sample) // 5))
    model = SentenceTransformer(cfg.analysis.embedding_model)
    emb = model.encode(sample, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    tfidf = TfidfVectorizer(token_pattern=r"[ء-ي]{3,}|[A-Za-z]{3,}", max_features=20000,
                            stop_words=list(AR_STOP))
    X = tfidf.fit_transform(sample)
    vocab = np.array(tfidf.get_feature_names_out())
    names = {}
    for c in range(k):
        rows = X[labels == c]
        if rows.shape[0] == 0:
            names[c] = f"topic {c}"
            continue
        top = np.asarray(rows.mean(axis=0)).ravel().argsort()[::-1][:4]
        names[c] = f"{c}: " + "، ".join(vocab[top])
    durations = final["duration"].to_numpy()[idx]
    tdf = pd.DataFrame({"topic": [names[l] for l in labels], "hours": durations / 3600})
    size = tdf.groupby("topic")["hours"].sum().sort_values(ascending=True)
    bar = px.bar(size, orientation="h", template=TEMPLATE, color_discrete_sequence=COLORS,
                 title="Topics (semantic clusters of transcripts, labelled by top keywords)",
                 labels={"value": "hours (sample)", "topic": ""})
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    pts = rng.choice(len(sample), size=min(len(sample), 4000), replace=False)
    scatter = px.scatter(x=xy[pts, 0], y=xy[pts, 1], color=[names[labels[i]] for i in pts],
                         hover_name=[sample[i][:80] for i in pts], template=TEMPLATE, opacity=0.6,
                         title="Semantic map of transcripts (PCA of sentence embeddings)")
    scatter.update_layout(showlegend=False)
    return {"topics": bar, "semantic_map": scatter}


def save(figs: dict, out_dir: Path, png: bool) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings = []
    for name, fig in figs.items():
        fig.write_html(str(out_dir / f"{name}.html"), include_plotlyjs="cdn")
        if png:
            try:
                fig.write_image(str(out_dir / f"{name}.png"), width=1200, height=700, scale=2)
            except Exception as exc:  # noqa: BLE001 - kaleido/Chrome missing: html is enough
                warnings.append(f"{name}.png: {exc}")
                png = False
    return warnings
