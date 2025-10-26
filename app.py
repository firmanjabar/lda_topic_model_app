# app.py
# Streamlit LDA Topic Modeling App (Indonesia & English)
# Dependencies: streamlit, pandas, gensim, nltk, numpy, pyLDAvis

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

import nltk
from nltk.corpus import stopwords

from gensim import corpora
from gensim.models import LdaModel
from gensim.models.phrases import Phrases, Phraser

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import streamlit.components.v1 as components

st.set_page_config(page_title="LDA Topic Modeling (ID & EN)", page_icon="🧵", layout="wide")

# ------------- Ensure NLTK resources -------------
@st.cache_resource(show_spinner=False)
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

ensure_nltk()

# Built-in minimal Indonesian stopwords fallback
STOP_ID_FALLBACK = {
    "yang","dan","di","ke","dari","untuk","pada","dengan","adalah","itu","ini","ada",
    "atau","tidak","saya","kami","kita","anda","dia","mereka","sebagai","sebuah","para",
    "dalam","akan","lagi","serta","atau","karena","juga","bagi","oleh"
}

def get_stopwords(lang_mode: str):
    """lang_mode: 'id','en','mix'"""
    sw_en = set(stopwords.words("english"))
    try:
        sw_id = set(stopwords.words("indonesian"))
    except OSError:
        sw_id = STOP_ID_FALLBACK
    if lang_mode == "id":
        return sw_id
    elif lang_mode == "en":
        return sw_en
    else:
        return sw_en | sw_id

def simple_tokenize(text: str, lowercase=True, rm_digits=True, rm_punct=True, min_len=2):
    if lowercase:
        text = text.lower()
    if rm_digits:
        text = re.sub(r"\d+", " ", text)
    if rm_punct:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    toks = [t for t in text.split() if len(t) >= min_len]
    return toks

def build_bigrams(tokens_list, threshold=10, min_count=5):
    phrases = Phrases(tokens_list, min_count=min_count, threshold=threshold, delimiter=b'_')
    bigram = Phraser(phrases)
    return [bigram[doc] for doc in tokens_list]

# ---------------- UI ----------------
st.title("🧵 LDA Topic Modeling — Bahasa Indonesia & English")
st.markdown(
    """
Aplikasi ini membangun **model LDA** dari korpus Indonesia/Inggris atau campuran.
- Input: **CSV** (kolom teks), **TXT** (satu dokumen per baris), atau **tempel teks**
- Preprocess: lowercase, hapus angka/tanda baca, stopwords (ID/EN), bigram opsional
- Output: topik (term teratas), distribusi topik per dokumen, & visualisasi **pyLDAvis**
    """
)

with st.sidebar:
    st.header("⚙️ Pengaturan")
    lang_mode = st.selectbox("Bahasa dokumen", ["id (Indonesia)", "en (English)", "mix (Campuran)"], index=2)
    lang_code = "id" if lang_mode.startswith("id") else ("en" if lang_mode.startswith("en") else "mix")
    lowercase = st.checkbox("Lowercase", True)
    rm_digits = st.checkbox("Hapus angka", False)
    rm_punct = st.checkbox("Hapus tanda baca", True)
    use_stop = st.checkbox("Buang stopwords", True)
    min_token_len = st.number_input("Panjang token minimum", 1, 10, 2)
    use_bigrams = st.checkbox("Gunakan bigrams (otomatis)", True)

    st.markdown("---")
    st.subheader("LDA Params")
    num_topics = st.slider("Jumlah topik", 2, 30, 8)
    passes = st.slider("passes (iterasi korpus)", 1, 50, 8)
    iters = st.slider("iterations (per dokumen)", 25, 400, 100, step=25)
    alpha = st.selectbox("alpha", ["symmetric", "asymmetric", "auto"], index=2)
    beta = st.selectbox("eta (beta)", ["auto", "symmetric"], index=0)
    random_state = st.number_input("random_state", 0, 9999, 42)

    st.markdown("---")
    st.subheader("Filter Vocabulary")
    no_below = st.number_input("no_below (min dokumen)", 1, 50, 2)
    no_above = st.slider("no_above (max prop dokumen)", 0.1, 1.0, 0.9, step=0.05)

st.subheader("1) Muat Korpus")
tab_csv, tab_txt, tab_paste = st.tabs(["📤 CSV", "📄 TXT", "📝 Tempel Teks"])

docs = []
doc_ids = []

with tab_csv:
    up = st.file_uploader("Unggah CSV (harus ada kolom teks)", type=["csv"])
    text_col = st.text_input("Nama kolom teks", value="text")
    if up is not None:
        try:
            df = pd.read_csv(up)
        except Exception:
            up.seek(0)
            df = pd.read_csv(io.BytesIO(up.read()), encoding_errors="ignore")
        if text_col not in df.columns:
            st.error(f"Kolom '{text_col}' tidak ada. Kolom tersedia: {list(df.columns)}")
        else:
            docs = df[text_col].astype("string").fillna("").tolist()
            doc_ids = df.index.astype(str).tolist()
            st.success(f"Memuat {len(docs)} dokumen dari CSV.")
            st.dataframe(df.head(10), use_container_width=True)

with tab_txt:
    up_txt = st.file_uploader("Unggah TXT (satu dokumen per baris)", type=["txt"])
    if up_txt is not None:
        t = up_txt.read().decode("utf-8", errors="ignore")
        lines = [line.strip() for line in t.splitlines() if line.strip()]
        docs = lines
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        st.success(f"Memuat {len(docs)} dokumen dari TXT.")

with tab_paste:
    txt = st.text_area("Tempel teks (satu dokumen per baris)", height=180, placeholder="Dokumen 1...\nDokumen 2...\nDokumen 3...")
    if txt.strip():
        docs = [line.strip() for line in txt.splitlines() if line.strip()]
        doc_ids = [f"doc_{i+1}" for i in range(len(docs))]
        st.success(f"Memuat {len(docs)} dokumen dari input tempel.")

if not docs:
    st.info("Unggah/Tempel korpus terlebih dahulu.")
    st.stop()

st.subheader("2) Pra-proses & Bangun Model")
if st.button("🚀 Jalankan LDA"):
    sw = get_stopwords(lang_code) if use_stop else set()

    # Tokenize
    tokens_list = [simple_tokenize(d, lowercase=lowercase, rm_digits=rm_digits, rm_punct=rm_punct, min_len=min_token_len) for d in docs]
    if use_stop:
        tokens_list = [[t for t in toks if t not in sw] for toks in tokens_list]

    # Optional bigrams
    if use_bigrams:
        tokens_list = build_bigrams(tokens_list, threshold=10, min_count=5)

    # Dictionary & Corpus
    dictionary = corpora.Dictionary(tokens_list)
    dictionary.filter_extremes(no_below=int(no_below), no_above=float(no_above))
    corpus = [dictionary.doc2bow(toks) for toks in tokens_list]

    if len(dictionary) == 0:
        st.error("Kosakata kosong setelah filter. Coba turunkan 'no_below' atau naikkan 'no_above'.")
        st.stop()

    # Train LDA
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=int(num_topics),
        random_state=int(random_state),
        passes=int(passes),
        iterations=int(iters),
        alpha=alpha,
        eta=beta,
        eval_every=None,
        per_word_topics=False
    )

    st.success(f"Model selesai. Vocabulary size: {len(dictionary)}, Docs: {len(corpus)}")

    # --------- Show topics
    st.markdown("### 3) Topik (Top Terms)")
    topn = st.slider("Top-N terms per topic", 5, 20, 10)
    topics = lda.show_topics(num_topics=int(num_topics), num_words=int(topn), formatted=False)
    rows = []
    for tid, terms in topics:
        rows.append({
            "topic_id": tid,
            "terms": ", ".join([w for (w, p) in terms]),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --------- Doc-topic distribution
    st.markdown("### 4) Distribusi Topik per Dokumen")
    doc_rows = []
    for i, bow in enumerate(corpus):
        doc_topics = lda.get_document_topics(bow, minimum_probability=0.0)
        # get dominant topic
        dom_topic, dom_prob = sorted(doc_topics, key=lambda x: x[1], reverse=True)[0]
        row = {"doc_id": doc_ids[i] if doc_ids else i, "dominant_topic": dom_topic, "score": float(dom_prob)}
        # add per-topic columns
        for (tid, prob) in doc_topics:
            row[f"topic_{tid}"] = float(prob)
        doc_rows.append(row)
    df_doc = pd.DataFrame(doc_rows)
    st.dataframe(df_doc, use_container_width=True, height=360)

    # Downloads
    st.download_button("⬇️ Unduh Distribusi Topik (CSV)", data=df_doc.to_csv(index=False).encode("utf-8"),
                       file_name="doc_topic_distribution.csv", mime="text/csv")

    # Save model & dictionary to bytes for user download
    import tempfile, os, pickle
    with tempfile.TemporaryDirectory() as tmpdir:
        dict_path = os.path.join(tmpdir, "dictionary.gensim")
        model_path = os.path.join(tmpdir, "lda_model.gensim")
        dictionary.save(dict_path)
        lda.save(model_path)
        with open(dict_path, "rb") as f:
            dict_bytes = f.read()
        with open(model_path, "rb") as f:
            model_bytes = f.read()

    st.download_button("⬇️ Unduh Dictionary (.gensim)", data=dict_bytes, file_name="dictionary.gensim")
    st.download_button("⬇️ Unduh LDA Model (.gensim)", data=model_bytes, file_name="lda_model.gensim")

    # --------- pyLDAvis
    st.markdown("### 5) Visualisasi Interaktif (pyLDAvis)")
    try:
        prepared = gensimvis.prepare(lda, corpus, dictionary)
        html = pyLDAvis.prepared_data_to_html(prepared)
        components.html(html, height=800, scrolling=True)
        # Allow download of HTML
        st.download_button("⬇️ Unduh pyLDAvis (HTML)",
                           data=html.encode("utf-8"),
                           file_name="pyLDAvis.html",
                           mime="text/html")
    except Exception as e:
        st.warning(f"Tidak dapat menampilkan pyLDAvis: {e}")

else:
    st.info("Atur parameter di sidebar lalu klik **🚀 Jalankan LDA**.")
