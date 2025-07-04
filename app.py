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
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for modern UI ---
st.markdown("""
<style>
    /* General Styling */
    .stApp {
        background-color: #f0f2f6;
        font-family: 'Inter', sans-serif;
    }

    /* Main Title */
    h1 {
        color: #1e3a8a; /* Dark Blue */
        text-align: center;
        font-weight: 800;
    }

    /* Subheader */
    .st-emotion-cache-10trblm {
        color: #4b5563; /* Gray */
        text-align: center;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        border: 2px solid #1e3a8a;
        background-color: #1e3a8a;
        color: white;
        font-weight: 600;
        transition: all 0.3s ease-in-out;
        padding: 10px 25px;
    }
    .stButton > button:hover {
        background-color: white;
        color: #1e3a8a;
        border-color: #1e3a8a;
    }
    .stButton > button:active {
        transform: scale(0.98);
    }

    /* Text Area */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid #d1d5db;
        background-color: #ffffff;
        min-height: 200px;
        font-size: 16px;
    }
    .stTextArea textarea:focus {
        border-color: #1e3a8a;
        box-shadow: 0 0 0 2px #bfdbfe;
    }
    
    /* Selectbox */
    .stSelectbox div[data-baseweb="select"] > div {
        border-radius: 15px;
        border: 2px solid #d1d5db;
        background-color: #ffffff;
    }
    .stSelectbox div[data-baseweb="select"] > div:focus-within {
        border-color: #1e3a8a;
    }

    /* Result Cards */
    .result-card {
        background-color: white;
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-left: 10px solid;
    }
    .result-card-positif {
        border-left-color: #22c55e; /* Green */
    }
    .result-card-negatif {
        border-left-color: #ef4444; /* Red */
    }
    .result-sentiment {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .result-sentiment.positif {
        color: #22c55e;
    }
    .result-sentiment.negatif {
        color: #ef4444;
    }

    /* Progress Bar */
    .stProgress > div > div > div > div {
        background-color: #22c55e;
    }

</style>
""", unsafe_allow_html=True)


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
        with st.spinner("Mengunduh NLTK Stopwords..."):
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
    with st.spinner("Mempersiapkan model saat pertama kali dijalankan... Ini mungkin memakan waktu beberapa menit."):
        if not os.path.exists(paths["rf"]): gdown.download(id=file_ids["rf"], output=paths["rf"], quiet=True)
        if not os.path.exists(paths["svm"]): gdown.download(id=file_ids["svm"], output=paths["svm"], quiet=True)
        if not os.path.exists(paths["knn"]): gdown.download(id=file_ids["knn"], output=paths["knn"], quiet=True)
        
        # Proses download dan unzip model BERT
        if not os.path.exists(os.path.join(paths["bert_dir"], "config.json")): 
            gdown.download(id=file_ids["bert_zip"], output=paths["bert_zip"], quiet=True)
            with zipfile.ZipFile(paths["bert_zip"], 'r') as zip_ref:
                zip_ref.extractall()
            os.remove(paths["bert_zip"])
        
        if not os.path.exists(paths["slang"]): gdown.download(id=file_ids["slang"], output=paths["slang"], quiet=True)

    # Load semua model dan resource
    with open(paths['rf'], 'rb') as file: rf_model = pickle.load(file)
    with open(paths['svm'], 'rb') as file: svm_model = pickle.load(file)
    with open(paths['knn'], 'rb') as file: knn_model = pickle.load(file)
    models = {'Random Forest': rf_model, 'Support Vector Machine (SVM)': svm_model, 'K-Nearest Neighbors (KNN)': knn_model}
    
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
try:
    models, tokenizer, bert_model, normalisasi_dict, indo_stopwords, stemmer = load_all_resources()

    # --- Antarmuka Streamlit ---
    st.title("🤖 Aplikasi Analisis Sentimen Teks")
    st.markdown("<p style='text-align: center; color: #4b5563;'>Analisis sentimen pada ulasan atau teks berbahasa Indonesia menggunakan model Machine Learning dan BERT.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([0.6, 0.4], gap="large")

    with col1:
        st.subheader("Masukkan Teks Anda")
        model_choice = st.selectbox("Pilih Model Klasifikasi", ('Random Forest', 'Support Vector Machine (SVM)', 'K-Nearest Neighbors (KNN)'))
        user_input = st.text_area("Ketik atau tempel teks di sini...", "Pelayanannya sangat ramah dan makanannya luar biasa enak!", height=210, label_visibility="collapsed")
        
        predict_button = st.button("Analisis Sentimen", use_container_width=True, type="primary")

    with col2:
        st.subheader("Hasil Analisis")
        if predict_button:
            if user_input:
                with st.spinner(f'Menganalisis dengan model {model_choice}...'):
                    time.sleep(1) # Simulasi proses
                    selected_model = models[model_choice]
                    cleaned_text = preprocess_text(user_input, normalisasi_dict, indo_stopwords, stemmer)
                    
                    if not cleaned_text.strip():
                        st.warning("Teks tidak mengandung kata yang dapat diproses setelah preprocessing. Coba masukkan teks lain.")
                    else:
                        bert_features = get_bert_embedding(cleaned_text, tokenizer, bert_model)
                        prediction = selected_model.predict(bert_features)
                        sentiment = "Positif" if prediction[0] == 1 else "Negatif"
                        
                        # Display result card
                        card_class = "result-card-positif" if sentiment == "Positif" else "result-card-negatif"
                        sentiment_class = "positif" if sentiment == "Positif" else "negatif"
                        icon = "😊" if sentiment == "Positif" else "😠"

                        st.markdown(f"""
                        <div class="result-card {card_class}">
                            <div class="result-sentiment {sentiment_class}">{icon} Prediksi: {sentiment}</div>
                            <p style="color: #4b5563;">Model yang digunakan: <b>{model_choice}</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Display probability
                        if hasattr(selected_model, 'predict_proba'):
                            prediction_proba = selected_model.predict_proba(bert_features)
                            
                            st.write("Tingkat Keyakinan:")
                            
                            # Custom progress bar with labels
                            prob_pos = prediction_proba[0][1]
                            prob_neg = prediction_proba[0][0]
                            
                            st.write(f"Positif: **{prob_pos:.2%}**")
                            st.progress(prob_pos)
                            
                            st.write(f"Negatif: **{prob_neg:.2%}**")
                            st.progress(prob_neg)
            else:
                st.warning("Harap masukkan teks untuk dianalisis.")
        else:
            st.info("Pilih model dan masukkan teks, lalu klik tombol 'Analisis Sentimen' untuk melihat hasilnya di sini.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau menjalankan aplikasi: {e}")
    st.info("Coba muat ulang halaman. Jika masalah berlanjut, pastikan file model dapat diakses.")

