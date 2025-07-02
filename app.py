import streamlit as st
import os
import requests
import pickle
import zipfile
import re
import string
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
import torch
from transformers import BertTokenizer, BertModel
import gdown

# --- Fungsi untuk download file dari Google Drive ---
def download_file_from_google_drive(id, destination):
    URL = f"https://drive.google.com/uc?export=download&id={id}"
    session = requests.Session()
    response = session.get(URL, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    with st.spinner(f"Mengunduh {os.path.basename(destination)}... (hanya sekali)"):
        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
    st.success(f"{os.path.basename(destination)} berhasil diunduh!")

# --- Fungsi Load Model (dengan cache agar tidak di-load berulang kali) ---
@st.cache_resource
def load_all_resources():
    # Cek dan Unduh NLTK Stopwords
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    # Membuat direktori jika belum ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('bert_model_tersimpan', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    # GANTI DENGAN FILE ID ANDA DARI GOOGLE DRIVE
    file_ids = {
        "rf": "1gWShpKmz8AiieIYmWf0o9htA2FHDQyhP",
        "svm": "1vR_OVC6Bb1LbZygNTD5rvIhEFQpSx08A",
        "knn": "1g0reDvjxifd1YFwuaJnIZ9kntwAvFbbw",
        "bert_zip": "1yjnWAV8vFPl5BJe8JHnOQig3Wq1BXX2N",
        "slang": "19_jiwSxQ4rmtPK3Swe4jgssbb93QBpkz"
    }

    # Path tujuan file
    paths = {
        "rf": "models/random_forest_model.pkl",
        "svm": "models/svm_model.pkl",
        "knn": "models/knn_model.pkl",
        "bert_zip": "bert_model_tersimpan.zip",
        "bert_dir": "bert_model_tersimpan",
        "slang": "data/slang.txt"
    }

    # Download file jika belum ada menggunakan gdown
    with st.spinner("Mempersiapkan model saat pertama kali dijalankan..."):
        if not os.path.exists(paths["rf"]): gdown.download(id=file_ids["rf"], output=paths["rf"], quiet=True)
        if not os.path.exists(paths["svm"]): gdown.download(id=file_ids["svm"], output=paths["svm"], quiet=True)
        if not os.path.exists(paths["knn"]): gdown.download(id=file_ids["knn"], output=paths["knn"], quiet=True)
        
        # Proses download dan unzip model BERT
        if not os.path.exists(os.path.join(paths["bert_dir"], "tokenizer")): 
            gdown.download(id=file_ids["bert_zip"], output=paths["bert_zip"], quiet=True)
            with zipfile.ZipFile(paths["bert_zip"], 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(paths["bert_zip"])
        
        if not os.path.exists(paths["slang"]): gdown.download(id=file_ids["slang"], output=paths["slang"], quiet=True)

    # Load semua model dan resource
    with open(paths['rf'], 'rb') as file: rf_model = pickle.load(file)
    with open(paths['svm'], 'rb') as file: svm_model = pickle.load(file)
    with open(paths['knn'], 'rb') as file: knn_model = pickle.load(file)
    models = {'Random Forest': rf_model, 'SVM': svm_model, 'KNN': knn_model}
    
    tokenizer = BertTokenizer.from_pretrained(os.path.join(paths["bert_dir"], 'tokenizer'))
    bert_model = BertModel.from_pretrained(os.path.join(paths["bert_dir"], 'model'))

    normalisasi_dict = {}
    with open(paths["slang"], "r", encoding="utf-8") as file:
        for line in file:
            if ":" in line:
                f = line.strip().split(":")
                normalisasi_dict[f[0].strip()] = f[1].strip()

    indo_stopwords = set(stopwords.words('indonesian'))
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    return models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer


# --- Fungsi Preprocessing ---
def preprocess_text(text, normalisasi_dict, indo_stopwords, stemmer):
    text = text.lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    for slang, baku in normalisasi_dict.items():
        pattern = r'\b' + re.escape(slang) + r'\b'
        text = re.sub(pattern, baku, text, flags=re.IGNORECASE)
    tokens = text.split()
    text = ' '.join([word for word in tokens if word not in indo_stopwords])
    text = stemmer.stem(text)
    return text

def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding.reshape(1, -1)

# Panggil fungsi utama untuk load semua resource
models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer = load_all_resources()

# --- Antarmuka Streamlit ---
st.set_page_config(page_title="Aplikasi Klasifikasi Sentimen", page_icon="💬")
st.title("💬 Aplikasi Klasifikasi Sentimen")
st.write("Pilih model dan masukkan teks untuk melakukan prediksi sentimen.")

model_choice = st.selectbox("Pilih Model", ('Random Forest', 'SVM', 'KNN'))
user_input = st.text_area("Masukkan teks untuk prediksi sentimen", "Programnya bagus, saya suka sekali.", height=150)

if st.button("Prediksi Sentimen", use_container_width=True):
    if user_input:
        with st.spinner(f'Memproses dengan model {model_choice}...'):
            selected_model = models[model_choice]
            cleaned_text = preprocess_text(user_input, normalisasi_dict, indo_stopwords, stemmer)
            bert_features = get_bert_embedding(cleaned_text, tokenizer, bert_model)
            prediction = selected_model.predict(bert_features)
            sentiment = "Positif" if prediction[0] == 1 else "Negatif"
            
            if sentiment == "Positif":
                st.success(f"Prediksi Sentimen: **{sentiment}**")
            else:
                st.error(f"Prediksi Sentimen: **{sentiment}**")
            
            if hasattr(selected_model, 'predict_proba'):
                prediction_proba = selected_model.predict_proba(bert_features)
                st.progress(prediction_proba[0][1])
                st.write(f"Positif: `{prediction_proba[0][1]:.2%}` | Negatif: `{prediction_proba[0][0]:.2%}`")