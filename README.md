# LDA Topic Modeling App (Streamlit) — Indonesia & English

Aplikasi web untuk membangun **LDA Topic Model** pada korpus **bahasa Indonesia**, **Inggris**, atau **campuran**.

## Fitur
- Input: **CSV** (kolom teks), **TXT** (satu dokumen per baris), atau **tempel teks**
- Preprocess: lowercase, hapus angka/tanda baca, stopwords (ID/EN), bigram opsional
- LDA parameterizable: jumlah topik, passes, iterations, alpha/eta
- Output: daftar topik (top terms), distribusi topik per dokumen (CSV)
- Visualisasi interaktif **pyLDAvis** (bisa diunduh HTML)
- Unduh **dictionary** & **model** (.gensim) untuk pemakaian ulang

## Instalasi (disarankan virtual env)
### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Menjalankan
```bash
streamlit run app.py
```
Buka URL yang tampil (umumnya `http://localhost:8501`).

## Cara pakai singkat
1. Muat data dari CSV/TXT/Tempel teks.  
2. Pilih bahasa dokumen (**id/en/mix**) dan atur opsi pra-proses.  
3. Atur parameter LDA (jumlah topik, passes, iterations, alpha/eta).  
4. Klik **Jalankan LDA** untuk membangun model.  
5. Lihat **Topik**, **Distribusi Topik per Dokumen**, dan **pyLDAvis**.  
6. Unduh **CSV**, **model**, **dictionary**, dan **pyLDAvis HTML** bila perlu.

## Catatan
- Stopwords Indonesia menggunakan NLTK; bila korpus tidak tersedia, aplikasi memakai fallback built-in.
- Jika pyLDAvis tidak tampil di beberapa environment, tetap bisa diunduh dalam bentuk **HTML**.
