"""
MVP - Interfaz Streamlit para identificación de cerdas
Uso: streamlit run app.py
"""

import streamlit as st
import os
import sys
import numpy as np
from PIL import Image
import tempfile

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_identificacion_cerdos.h5")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset_procesado")

CLASS_NAMES = sorted([
    d for d in os.listdir(DATASET_PATH) 
    if os.path.isdir(os.path.join(DATASET_PATH, d))
])

CLASS_MAPPING = {i: name for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

THRESHOLD = 0.50

@st.cache_resource
def load_model():
    import tensorflow as tf
    st.write("🔄 Cargando modelo...")
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

def preprocess_image(image, target_size=(224, 224)):
    img = image.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_top3(model, image):
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    
    top_indices = np.argsort(predictions)[::-1][:3]
    
    results = []
    for idx in top_indices:
        confidence = float(predictions[idx])
        is_unknown = confidence < THRESHOLD
        
        results.append({
            "class_id": int(idx),
            "class_name": CLASS_MAPPING[idx] if not is_unknown else "Desconocido",
            "confidence": confidence,
            "is_unknown": is_unknown
        })
    
    return results

def main():
    st.set_page_config(
        page_title="Identificación de Cerdas",
        page_icon="🐷",
        layout="centered"
    )
    
    st.title("🐷 Identificación Biométrica de Cerdas")
    st.markdown("---")
    
    st.info(f"**Clases disponibles:** {', '.join(CLASS_NAMES)}")
    st.info(f"**Umbral de confianza:** {THRESHOLD:.0%}")
    
    st.markdown("### 📸 Subir foto")
    
    uploaded_file = st.file_uploader(
        "Selecciona una imagen de la cerda",
        type=['jpg', 'jpeg', 'png', 'webp']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Foto subida", use_container_width=True)
        
        with col2:
            if st.button("🔍 Identificar", type="primary"):
                with st.spinner("Analizando imagen..."):
                    try:
                        model = load_model()
                        results = predict_top3(model, image)
                        
                        st.markdown("### 📊 Resultados Top 3")
                        
                        for i, r in enumerate(results, 1):
                            confidence_pct = r["confidence"] * 100
                            
                            if r["is_unknown"]:
                                status = "⚠️ DESCONOCIDO"
                                color = "red"
                            else:
                                status = "✅"
                                color = "green"
                            
                            st.markdown(f"""
                            **{i}. {r['class_name']}** {status}
                             - Confianza: {confidence_pct:.1f}%
                            """)
                        
                        st.markdown("---")
                        st.markdown("### 👤 Confirmación")
                        
                        selected = st.radio(
                            "Selecciona la cerda correcta:",
                            options=[r["class_name"] for r in results],
                            horizontal=True
                        )
                        
                        if selected:
                            if st.button("✓ Confirmar", type="secondary"):
                                st.success(f"✅ Confirmado: {selected}")
                                
                                with open("logs/confirmaciones.csv", "a") as f:
                                    import datetime
                                    f.write(f"{datetime.datetime.now()},{uploaded_file.name},{selected}\n")
                                
                                st.balloons()
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
    
    st.markdown("---")
    st.markdown("*MVP v1.0 - Porci Integral*")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    main()
