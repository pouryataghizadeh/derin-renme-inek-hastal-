import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import time

# Sayfa ayarları - Geniş ekran ve profesyonel başlık
st.set_page_config(page_title="VetAI - Klinik Teşhis Platformu", page_icon="🔬", layout="wide")

# Gelişmiş Özel CSS Tasarımı
st.markdown("""
    <style>
    /* Ana Arka plan ve Yazı tipleri */
    .stApp {
        background-color: #f8fafc;
    }
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        text-align: center;
        padding-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        padding-bottom: 2rem;
    }
    /* Sonuç Kartları */
    .result-card-healthy {
        background-color: #dcfce7;
        border-left: 6px solid #22c55e;
        padding: 20px;
        border-radius: 8px;
        color: #166534;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .result-card-disease {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 20px;
        border-radius: 8px;
        color: #991b1b;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    /* Dosya Yükleme Alanı Düzenlemesi */
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff;
        border: 2px dashed #94a3b8;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL YÜKLEME ---
custom_objects = {"TFOpLambda": tf.keras.layers.Lambda}

@st.cache_resource
def load_model():
    # Hatayı çözdüğümüz özel yükleme ayarları
    return tf.keras.models.load_model("best_cow_model.h5", custom_objects=custom_objects, compile=False)

model = load_model()
IMG_SIZE = 384
class_names = ["Şap Hastalığı (Foot-and-Mouth)", "Sağlıklı (Healthy)", "Yumru Deri Hastalığı (Lumpy)"]

# --- ANA SAYFA ---
st.markdown('<div class="hero-title">🔬 VetAI Diagnostic Systems</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Büyükbaş Hayvan Dermatolojisi İçin Yapay Zeka Destekli Erken Teşhis Platformu</div>', unsafe_allow_html=True)

# Üst Bilgi Kartları (Dashboard görünümü)
m1, m2, m3 = st.columns(3)
m1.metric("Desteklenen Hastalık Türü", "2 (Şap, Yumru Deri)")
m2.metric("Ortalama Analiz Süresi", "< 1.5 Saniye")
m3.metric("Sistem Altyapısı", "EfficientNetB4 (Deep Learning)")

st.write("---")

# --- İÇERİK BÖLÜMÜ ---
col_left, col_right = st.columns([1, 1.2], gap="large")

with col_left:
    st.subheader("📁 Vaka Görüntüsü Yükleme")
    uploaded_file = st.file_uploader("Klinik değerlendirme için net bir lezyon fotoğrafı yükleyin", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Yüklenen Vaka Görüntüsü", use_container_width=True)

with col_right:
    st.subheader("📊 Yapay Zeka Analiz Raporu")
    
    if uploaded_file is None:
        st.info("Sistem analize başlamak için sol taraftan görüntü yüklemenizi bekliyor...")
    else:
        # Dikkat çeken işlem butonu
        if st.button("Görüntüyü İşle ve Raporla", type="primary", use_container_width=True):
            
            with st.spinner('Derin öğrenme ağı katmanları inceleniyor...'):
                time.sleep(1) # Ekranda bir şeylerin işlendiğini hissettiren profesyonel kısa bekleme
                
                # Modeli Hazırla
                img = image.resize((IMG_SIZE, IMG_SIZE))
                img_array = np.array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)
                
                # Tahmin Yap
                prediction = model.predict(img_array)
                index = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
            # Sonuç Gösterimi
            st.write("### Klinik Teşhis Sonucu")
            
            # Hastalık ve Sağlık Durumuna Göre Özel Kutu
            if "Sağlıklı" in class_names[index]:
                st.markdown(f"""
                <div class="result-card-healthy">
                    <h3 style="margin:0;">✅ {class_names[index]}</h3>
                    <p style="margin:0; margin-top:5px;">Hayvanın incelenen cilt bölgesinde herhangi bir patolojik veya viral bulguya rastlanmadı.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-disease">
                    <h3 style="margin:0;">⚠️ {class_names[index]}</h3>
                    <p style="margin:0; margin-top:5px;">Klinik inceleme önerilir. Sistemin tespit ettiği hastalık riski yüksektir ve izolasyon değerlendirilmelidir.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.write("#### Güven Skoru (Model Confidence)")
            # Havalı Güvenlik Barı
            st.progress(int(confidence), text=f"**%{confidence:.2f}** Doğruluk Olasılığı")
            
            # Alt Uyarı Notu
            with st.expander("📄 Hekim Bilgilendirme ve Yasal Not"):
                st.warning("Uyarı: Bu sistem nihai tıbbi teşhis koymaz, veteriner hekimlere klinik karar destek mekanizması (DSS) olarak hizmet vermek üzere tasarlanmıştır. Tedavi süreçleri, fiziksel muayene ve laboratuvar testleri ile doğrulanmalıdır.")
