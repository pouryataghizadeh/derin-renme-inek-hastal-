import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

# Sayfa ayarlarını geniş ekran yapıyoruz
st.set_page_config(page_title="VetAI - İnek Hastalığı Teşhisi", page_icon="🩺", layout="wide")

# Özel CSS ile daha modern ve havalı bir görünüm
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 20px;
        color: #64748B;
        text-align: center;
        margin-bottom: 40px;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        border-color: #1D4ED8;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # Güvenli modda (safe_mode=False) yükleyerek TFOpLambda hatasını aşmayı deniyoruz
    return tf.keras.models.load_model("best_cow_model.h5", compile=False, safe_mode=False) 

model = load_model()
IMG_SIZE = 384
class_names = ["Şap Hastalığı (Foot-and-Mouth)", "Sağlıklı (Healthy)", "Yumru Deri Hastalığı (Lumpy)"]

# Başlık Kısmı
st.markdown('<div class="main-header">🩺 VetAI Teşhis Platformu</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Yapay Zeka Destekli Dermatolojik Erken Teşhis Sistemi</div>', unsafe_allow_html=True)

# Sol tarafa bir bilgi menüsü ekliyoruz
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004141.png", width=120)
    st.header("Sistem Hakkında")
    st.write("Bu yapay zeka destekli model, ineklerdeki yaygın cilt hastalıklarını yüksek doğrulukla tespit etmek için EfficientNetB4 mimarisi ile eğitilmiştir.")
    st.info("💡 **Nasıl Kullanılır?**\n\nAnaliz etmek istediğiniz lezyonun veya bölgenin net bir fotoğrafını yükleyin ve klinik destek sonucunu saniyeler içinde alın.")
    st.write("---")
    st.caption("© 2026 VetAI Diagnostic Systems")

st.write("---")

# Sayfayı iki kolona bölüyoruz: Sol taraf yükleme, sağ taraf analiz
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 Vaka Görüntüsü Yükle")
    uploaded_file = st.file_uploader("Lütfen analize gönderilecek fotoğrafı seçin", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("🔬 Yapay Zeka Analizi")
    
    if uploaded_file is None:
        st.info("Lütfen sol taraftan bir fotoğraf yükleyin. Sistem analize hazır bekliyor.")
    else:
        # Fotoğrafı Göster
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Sisteme Yüklenen Vaka", use_container_width=True)
        
        # Butona tıklandığında analizi başlat
        if st.button("Görüntüyü Analiz Et 🚀"):
            
            # Havalı bir ilerleme çubuğu animasyonu
            progress_text = "Yapay zeka ağları görüntüyü işliyor..."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01) # Animasyon hızı
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(0.3)
            my_bar.empty() # İşlem bitince çubuğu gizle

            # Modeli Hazırla
            img = image.resize((IMG_SIZE, IMG_SIZE))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)

            # Tahmin Yap
            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            # Sonuçları Renklendirerek Göster
            st.write("---")
            if "Sağlıklı" in class_names[index]:
                st.success(f"**Teşhis Sonucu:** {class_names[index]}")
                st.balloons() # Ekranda balonlar uçar :)
            else:
                st.error(f"**Dikkat, Tespit Edilen Hastalık:** {class_names[index]}")
            
            # Modern Metrik Kartı
            st.metric(label="Model Güven (Eminlik) Skoru", value=f"%{confidence:.2f}")

            # Ekstra Bilgi Kutusu
            with st.expander("📋 Sonraki Adımlar ve Klinik Uyarı"):
                st.write("Bu sistem **ikinci bir görüş** sağlamak amacıyla tasarlanmıştır. Lütfen yapay zeka sonuçlarını fiziksel muayene bulguları ile eşleştirerek nihai tanıyı koyunuz.")
