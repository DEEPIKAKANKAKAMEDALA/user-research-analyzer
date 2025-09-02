# app.py — Upload (CSV/Excel) → Clean → Embed → Auto-K KMeans → Summarize with Gemini → Dashboard
import os, json, re, string
import pandas as pd, numpy as np, streamlit as st, plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastembed import TextEmbedding
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import google.generativeai as genai

st.set_page_config(page_title="AI User Research Analyzer", layout="wide")
st.title("AI-Powered User Research Analyzer")
st.caption("Upload a CSV or Excel with a text column (survey answers or reviews). Optional: source, user_segment, created_at.")

# ---- Helpers
@st.cache_resource
def ensure_nltk():
    try: nltk.data.find("tokenizers/punkt")
    except LookupError: nltk.download("punkt")
    try: nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try: nltk.download("punkt_tab")
        except Exception: pass
    try: nltk.data.find("corpora/stopwords")
    except LookupError: nltk.download("stopwords")

def read_table(upload) -> pd.DataFrame:
    name = upload.name.lower()
    if name.endswith(".csv"): return pd.read_csv(upload)
    return pd.read_excel(upload)

def basic_clean(s: str, sw, punct):
    if not isinstance(s, str): return ""
    s = s.lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)
    s = re.sub(r"[\r\n\t]+", " ", s)
    s = re.sub(r"[^\x00-\x7F]+", " ", s)
    toks = [w for w in word_tokenize(s) if w not in punct]
    toks = [w for w in toks if w.isalpha() and w not in sw]
    return " ".join(toks)

def tfidf_keywords(texts, top_k=6):
    vec = TfidfVectorizer(max_features=6000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    idx = scores.argsort()[::-1][:top_k]
    return np.array(vec.get_feature_names_out())[idx].tolist()

def pick_reps(df_cluster, k=8):
    return (df_cluster.assign(_len=df_cluster["text"].astype(str).str.len())
            .sort_values("_len")
            .head(min(k, len(df_cluster)))["text"].astype(str).tolist())

def choose_k(emb, n):
    if n < 3: return 1
    if n < 10: return 2
    k_min, k_max = 2, min(10, max(3, n // 3))
    best_k, best_score = 2, -1
    for k in range(k_min, k_max+1):
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=256)
        labels = km.fit_predict(emb)
        if len(set(labels)) < 2: continue
        try:
            score = silhouette_score(emb, labels, metric="cosine")
            if score > best_score: best_k, best_score = k, score
        except Exception:
            pass
    return best_k

@st.cache_resource
def load_embedder():
    # Small, accurate, fast CPU model
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# fastembed returns a generator of lists -> turn into a numpy array
emb_gen = load_embedder().embed(df["text_clean"].tolist())
emb = np.array(list(emb_gen), dtype="float32")

# (optional) normalize like sentence-transformers did
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

def gemini_label(quotes: list[str], api_key: str) -> dict:
    if not api_key: return {"label":"Theme", "bullets":[]}
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "You are helping a product team synthesize user feedback.\n"
        "Given the sample quotes below, produce:\n"
        "1) A short 3-5 word theme label\n"
        "2) 3 concise bullet insights summarizing the core pain/opportunity\n\n"
        "Quotes:\n" + "\n".join(f"{i+1}. {q}" for i,q in enumerate(quotes)) +
        "\n\nReturn strict JSON with keys: label, bullets"
    )
    resp = model.generate_content(prompt)
    txt = (resp.text or "").strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    lines = [l.strip("- ").strip() for l in txt.splitlines() if l.strip()]
    return {"label": (lines[0][:60] if lines else "Theme"), "bullets": lines[1:4] if len(lines)>1 else []}

def run_pipeline(df_in: pd.DataFrame, text_col: str, source_col: str|None, seg_col: str|None, api_key: str):
    ensure_nltk()
    sw, punct = set(stopwords.words("english")), set(string.punctuation)

    df = df_in.copy()
    df["text"] = df[text_col].astype(str)
    df["source"] = df[source_col].astype(str) if source_col else "unknown"
    df["user_segment"] = df[seg_col].astype(str) if seg_col else "unknown"
    if "id" not in df.columns: df["id"] = np.arange(1, len(df)+1)

    df["text_clean"] = df["text"].map(lambda s: basic_clean(s, sw, punct))
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    if len(df) == 0: raise ValueError("No valid rows after cleaning.")

    emb = load_embedder().encode(df["text_clean"].tolist(), normalize_embeddings=True, show_progress_bar=False)

    k = choose_k(emb, len(df))
    if k == 1:
        df["cluster"] = 0
    else:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto", batch_size=256)
        df["cluster"] = km.fit_predict(emb)

    summaries = []
    for cid, g in df.groupby("cluster"):
        quotes = pick_reps(g, k=8)
        lab = gemini_label(quotes, api_key)
        label = lab.get("label") or "Theme"
        bullets = lab.get("bullets", [])
        top_kws = tfidf_keywords(g["text_clean"].tolist(), top_k=6)
        seg_counts = g["user_segment"].value_counts().to_dict()
        summaries.append({
            "cluster_id": int(cid),
            "size": int(len(g)),
            "label": label,
            "bullets": bullets,
            "top_keywords": top_kws,
            "sample_quotes": quotes,
            "segments": seg_counts
        })

    id2label = {s["cluster_id"]: s["label"] for s in summaries}
    df["cluster_label"] = df["cluster"].map(id2label)
    return df, summaries

# ---- Sidebar: upload + mapping
st.sidebar.header("Upload & Configure")
# Prefer secret; if not present, allow manual input
gemini_key = st.secrets.get("GEMINI_API_KEY", "")
if not gemini_key:
    gemini_key = st.sidebar.text_input("Gemini API Key (optional)", type="password",
                                       help="Get one at aistudio.google.com/app/apikey. Leave blank to use keyword-only labels.")

uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv","xlsx","xls"])
use_sample = st.sidebar.toggle("Use sample data", value=not bool(uploaded))

df_preview = None
if use_sample:
    df_preview = pd.DataFrame({
        "id":[1,2,3,4,5,6],
        "text":[
            "Signup is confusing, can't find the continue button",
            "Dashboard slow when filtering by date",
            "Exports fail sometimes from reports",
            "After onboarding I don't know next step",
            "Mobile app crashes on checkout",
            "Too many steps to invite teammates"
        ],
        "source":["survey","review","interview","survey","support","NPS"],
        "user_segment":["Free","Pro","Pro","Free","Gen Z","SMB"]
    })
elif uploaded:
    try: df_preview = read_table(uploaded)
    except Exception as e: st.error(f"Could not read the file: {e}")

if df_preview is not None:
    st.subheader("Preview")
    st.dataframe(df_preview.head(20), use_container_width=True)

    st.sidebar.subheader("Column mapping")
    cols = df_preview.columns.tolist()
    text_col = st.sidebar.selectbox("Text column", cols, index=cols.index("text") if "text" in cols else 0)
    source_col = st.sidebar.selectbox("Source (optional)", ["(none)"]+cols, index=(cols.index("source")+1 if "source" in cols else 0))
    seg_col = st.sidebar.selectbox("User segment (optional)", ["(none)"]+cols, index=(cols.index("user_segment")+1 if "user_segment" in cols else 0))
    source_col = None if source_col=="(none)" else source_col
    seg_col = None if seg_col=="(none)" else seg_col

    if st.sidebar.button("Analyze"):
        try:
            with st.spinner("Crunching feedback…"):
                df_out, summaries = run_pipeline(df_preview, text_col, source_col, seg_col, gemini_key)

            c1,c2,c3 = st.columns(3)
            c1.metric("Feedback items", f"{len(df_out):,}")
            c2.metric("Themes", df_out["cluster"].nunique())
            c3.metric("Sources", df_out["source"].nunique() if "source" in df_out.columns else 0)

            st.sidebar.header("Filters")
            segs = ["(all)"] + sorted(df_out["user_segment"].dropna().unique().tolist())
            srcs = ["(all)"] + sorted(df_out["source"].dropna().unique().tolist())
            seg_f = st.sidebar.selectbox("Filter by segment", segs)
            src_f = st.sidebar.selectbox("Filter by source", srcs)
            search = st.sidebar.text_input("Search text")

            q = df_out.copy()
            if seg_f != "(all)": q = q[q["user_segment"] == seg_f]
            if src_f != "(all)": q = q[q["source"] == src_f]
            if search.strip(): q = q[q["text"].str.contains(search, case=False, na=False)]

            st.subheader("Top pain points / themes")
            theme_counts = (q.groupby("cluster_label")["id"].count().sort_values(ascending=False).reset_index(name="count"))
            st.plotly_chart(px.bar(theme_counts, x="cluster_label", y="count"), use_container_width=True)

            st.subheader("Theme drill-down")
            if not theme_counts.empty:
                picked = st.selectbox("Choose a theme", theme_counts["cluster_label"])
                cid = int(df_out[df_out["cluster_label"] == picked]["cluster"].iloc[0])
                srow = [s for s in summaries if s["cluster_id"] == cid][0]

                st.markdown(f"### {srow['label']}")
                bullets = srow.get("bullets", [])
                if isinstance(bullets, list) and bullets:
                    st.write("- " + "\n- ".join(bullets))
                else:
                    st.caption("Keyword-only theme (no Gemini key provided).")

                seg_break = (q[q["cluster_label"] == picked]
                             .groupby("user_segment")["id"].count().reset_index(name="count").sort_values("count", ascending=False))
                if not seg_break.empty:
                    st.plotly_chart(px.bar(seg_break, x="user_segment", y="count"), use_container_width=True)

                st.write("**Sample quotes**")
                cols_show = [c for c in ["id","text","source","user_segment"] if c in df_out.columns]
                st.dataframe(df_out[df_out["cluster_label"] == picked][cols_show].head(50), use_container_width=True)

            st.sidebar.header("Downloads")
            st.sidebar.download_button("Download clusters.csv", data=df_out.to_csv(index=False), file_name="clusters.csv")
            st.sidebar.download_button("Download summaries.json", data=json.dumps(summaries, indent=2), file_name="summaries.json")

        except Exception as e:
            st.exception(e)
else:
    st.info("Upload a file or toggle 'Use sample data' in the sidebar.")
