import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import os
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, GlobalAveragePooling2D)
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    .main-header h1 { color: #e94560; font-size: 2.2rem; margin-bottom: 0.3rem; }
    .main-header p { color: #a8a8b3; font-size: 1rem; }
    .result-box {
        padding: 1.5rem; border-radius: 12px;
        text-align: center; margin: 1rem 0;
    }
    .normal-box { background: linear-gradient(135deg, #d4edda, #c3e6cb); border: 2px solid #28a745; }
    .pneumonia-box { background: linear-gradient(135deg, #f8d7da, #f5c6cb); border: 2px solid #dc3545; }
    .metric-card {
        background: #f8f9fa; padding: 1rem; border-radius: 10px;
        text-align: center; border: 1px solid #dee2e6;
    }
    .metric-card h3, .metric-card h4, .metric-card p { color: #000000 !important; }
    .info-box {
        background: #e8f4f8; padding: 1rem; border-radius: 10px;
        border-left: 4px solid #17a2b8; margin: 1rem 0;
    }
    .info-box, .info-box strong, .info-box p { color: #000000 !important; }
    .result-box h2, .result-box p { color: #000000 !important; }

    /* MAIN CONTENT — black text */
    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4,
    [data-testid="stMainBlockContainer"] p,
    [data-testid="stMainBlockContainer"] strong,
    [data-testid="stMainBlockContainer"] li,
    [data-testid="stMainBlockContainer"] span,
    [data-testid="stMainBlockContainer"] label,
    [data-testid="stMainBlockContainer"] .stMarkdown { color: #000000 !important; }

    /* SIDEBAR — white text */
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] strong,
    [data-testid="stSidebar"] .stMarkdown { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL ARCHITECTURE BUILDERS (MUST MATCH TRAINING)
# ============================================================

IMG_SIZE = 128

def build_mobilenet_architecture():
    base = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_SIZE, IMG_SIZE, 3), alpha=0.75)
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_resnet50_architecture():
    base = ResNet50(weights='imagenet', include_top=False,
                    input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base.layers:
        layer.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=output)
    model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ============================================================
# MODEL LOADING — REBUILD ARCHITECTURE + LOAD WEIGHTS
# ============================================================

@st.cache_resource
def load_models():
    """Rebuild architectures locally and load saved weights."""
    loaded = {}

    # MobileNetV2
    if os.path.exists('mobilenet_pneumonia.weights.h5'):
        m = build_mobilenet_architecture()
        m.load_weights('mobilenet_pneumonia.weights.h5')
        loaded['MobileNetV2'] = m
        print("Loaded: MobileNetV2")

    # ResNet50
    if os.path.exists('resnet50_pneumonia.weights.h5'):
        m = build_resnet50_architecture()
        m.load_weights('resnet50_pneumonia.weights.h5')
        loaded['ResNet50'] = m
        print("Loaded: ResNet50")

    # Fine-tuned best model
    if os.path.exists('best_model_finetuned_pneumonia.weights.h5'):
        config = load_config()
        best_name = config.get('best_model', 'MobileNetV2') if config else 'MobileNetV2'

        if best_name == 'MobileNetV2':
            m = build_mobilenet_architecture()
            for layer in m.layers[-20:]:
                layer.trainable = True
        else:
            m = build_resnet50_architecture()
            for layer in m.layers[-15:]:
                layer.trainable = True

        m.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])
        m.load_weights('best_model_finetuned_pneumonia.weights.h5')
        loaded['Best Model (Fine-Tuned)'] = m
        print(f"Loaded: Fine-Tuned {best_name}")

    return loaded


@st.cache_data
def load_config():
    if os.path.exists('model_config.json'):
        with open('model_config.json', 'r') as f:
            return json.load(f)
    return None

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return image, img_array, img_batch


def get_gradcam_heatmap(model, img_array_batch, last_conv_layer_name):
    try:
        grad_model = Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, pred = grad_model(tf.cast(img_array_batch, tf.float32))
            loss = pred[:, 0]
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy().astype(np.float32)
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None


def create_gradcam_overlay(img_array, heatmap, img_size=128):
    heatmap_resized = cv2.resize(heatmap, (img_size, img_size))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap_color / 255.0) * 0.4 + img_array
    superimposed = np.clip(superimposed, 0, 1)
    return superimposed, heatmap_resized


def get_last_conv_layer(model_name):
    conv_layers = {
        'MobileNetV2': 'Conv_1',
        'ResNet50': 'conv5_block3_out',
        'Best Model (Fine-Tuned)': 'Conv_1'
    }
    # If fine-tuned model is ResNet, adjust
    config = load_config()
    if model_name == 'Best Model (Fine-Tuned)' and config:
        if config.get('best_model') == 'ResNet50':
            return 'conv5_block3_out'
    return conv_layers.get(model_name, 'Conv_1')


def create_confidence_gauge(confidence, prediction):
    color = '#28a745' if prediction == 'NORMAL' else '#dc3545'
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={'suffix': '%', 'font': {'size': 40}},
        title={'text': f'Prediction: {prediction}', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': 'white', 'borderwidth': 2,
            'steps': [
                {'range': [0, 50], 'color': '#e8f5e9'},
                {'range': [50, 100], 'color': '#ffebee'}
            ],
            'threshold': {'line': {'color': 'black', 'width': 3},
                          'thickness': 0.8, 'value': 50}
        }
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=20, l=30, r=30))
    return fig

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    models = load_models()
    config = load_config()

    if models:
        selected_model_name = st.selectbox(
            "🔬 Select Model", list(models.keys()),
            index=len(models) - 1
        )
        st.markdown("---")
        show_gradcam = st.checkbox("🔥 Show Grad-CAM", value=True)
        show_details = st.checkbox("📊 Show Detailed Metrics", value=True)

        st.markdown("---")
        st.markdown("### 📋 Model Info")
        if config and 'results' in config:
            for model_name, metrics in config['results'].items():
                with st.expander(f"📈 {model_name}"):
                    for metric, value in metrics.items():
                        st.write(f"**{metric}:** {value:.4f}")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This system uses deep learning models trained on the
        [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
        to classify chest X-ray images as **Normal** or **Pneumonia**.

        **Models Available:**
        - MobileNetV2 (Transfer Learning)
        - ResNet50 (Transfer Learning)
        - Fine-Tuned Best Model
        """)
    else:
        st.error("No models found. Place .weights.h5 files in the app directory.")

# ============================================================
# MAIN PAGE
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🫁 Pneumonia Detection System</h1>
    <p>Automated Chest X-Ray Analysis using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>📌 How to use:</strong> Upload a chest X-ray image (JPEG/PNG) using the uploader below.
    The system will classify it as <strong>Normal</strong> or <strong>Pneumonia</strong> and display
    a Grad-CAM heatmap showing which regions influenced the prediction.
</div>
""", unsafe_allow_html=True)

st.markdown("")

uploaded_file = st.file_uploader(
    "📤 Upload a Chest X-Ray Image",
    type=['jpg', 'jpeg', 'png'],
    help="Upload a frontal chest X-ray image for pneumonia classification."
)

if uploaded_file is not None and models:
    original_image, img_array, img_batch = preprocess_image(uploaded_file)
    selected_model = models[selected_model_name]

    with st.spinner("🔍 Analysing X-ray..."):
        prediction_prob = float(selected_model.predict(img_batch, verbose=0)[0][0])

    pred_label = 'PNEUMONIA' if prediction_prob >= 0.5 else 'NORMAL'
    confidence = prediction_prob if pred_label == 'PNEUMONIA' else 1 - prediction_prob

    # Results
    st.markdown("---")
    st.markdown("## 🩺 Diagnosis Result")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(original_image, caption="Uploaded Chest X-Ray", use_container_width=True)

    with col2:
        gauge_fig = create_confidence_gauge(confidence, pred_label)
        st.plotly_chart(gauge_fig, use_container_width=True)

        if pred_label == 'NORMAL':
            st.markdown("""
            <div class="result-box normal-box">
                <h2 style="color: #155724;">✅ NORMAL</h2>
                <p style="color: #155724;">No signs of pneumonia detected.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-box pneumonia-box">
                <h2 style="color: #721c24;">⚠️ PNEUMONIA DETECTED</h2>
                <p style="color: #721c24;">Signs of pneumonia have been identified. Please consult a medical professional.</p>
            </div>
            """, unsafe_allow_html=True)

    # Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.markdown("## 🔥 Grad-CAM Explainability")
        st.markdown("""
        <div class="info-box">
            Grad-CAM highlights the regions of the X-ray that most influenced the model's prediction.
            <strong>Red/yellow areas</strong> indicate high importance, while <strong>blue areas</strong>
            indicate low importance.
        </div>
        """, unsafe_allow_html=True)

        last_conv_layer = get_last_conv_layer(selected_model_name)
        heatmap = get_gradcam_heatmap(selected_model, img_batch, last_conv_layer)

        if heatmap is not None:
            overlay, heatmap_resized = create_gradcam_overlay(img_array, heatmap, IMG_SIZE)

            gcol1, gcol2, gcol3 = st.columns(3)
            with gcol1:
                st.markdown("**Original X-Ray**")
                st.image(img_array, use_container_width=True)
            with gcol2:
                st.markdown("**Grad-CAM Heatmap**")
                heatmap_colour = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)
                st.image(heatmap_colour, use_container_width=True)
            with gcol3:
                st.markdown("**Overlay**")
                st.image(overlay, use_container_width=True)

    # Details
    if show_details:
        st.markdown("---")
        st.markdown("## 📊 Prediction Details")

        dcol1, dcol2, dcol3, dcol4 = st.columns(4)
        with dcol1:
            st.markdown(f"""<div class="metric-card"><h3>🏷️ Model</h3>
            <p style="font-size:1.1rem;font-weight:bold;">{selected_model_name}</p></div>""",
            unsafe_allow_html=True)
        with dcol2:
            st.markdown(f"""<div class="metric-card"><h3>📋 Prediction</h3>
            <p style="font-size:1.1rem;font-weight:bold;">{pred_label}</p></div>""",
            unsafe_allow_html=True)
        with dcol3:
            st.markdown(f"""<div class="metric-card"><h3>🎯 Confidence</h3>
            <p style="font-size:1.1rem;font-weight:bold;">{confidence:.2%}</p></div>""",
            unsafe_allow_html=True)
        with dcol4:
            st.markdown(f"""<div class="metric-card"><h3>📐 Raw Score</h3>
            <p style="font-size:1.1rem;font-weight:bold;">{prediction_prob:.4f}</p></div>""",
            unsafe_allow_html=True)

        # Probability bar chart
        st.markdown("### Probability Distribution")
        prob_fig = go.Figure()
        prob_fig.add_trace(go.Bar(
            x=['Normal', 'Pneumonia'],
            y=[1 - prediction_prob, prediction_prob],
            marker_color=['#28a745', '#dc3545'],
            text=[f'{(1-prediction_prob)*100:.1f}%', f'{prediction_prob*100:.1f}%'],
            textposition='auto', textfont_size=16
        ))
        prob_fig.update_layout(height=300, yaxis_title='Probability',
                               yaxis_range=[0, 1], margin=dict(t=20, b=40))
        st.plotly_chart(prob_fig, use_container_width=True)

    # Disclaimer
    st.markdown("---")
    st.warning("""
    ⚠️ **Medical Disclaimer:** This tool is designed for research and educational purposes only.
    It should NOT be used as a substitute for professional medical diagnosis.
    Always consult a qualified healthcare professional for medical decisions.
    """)

elif not models:
    st.error("⚠️ No models loaded. Please ensure .weights.h5 files are in the application directory.")
else:
    st.markdown("")
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown("""<div class="metric-card"><h3>📤</h3><h4>Step 1</h4>
        <p>Upload a chest X-ray image</p></div>""", unsafe_allow_html=True)
    with lcol2:
        st.markdown("""<div class="metric-card"><h3>🤖</h3><h4>Step 2</h4>
        <p>AI analyses the image</p></div>""", unsafe_allow_html=True)
    with lcol3:
        st.markdown("""<div class="metric-card"><h3>🩺</h3><h4>Step 3</h4>
        <p>View diagnosis and Grad-CAM</p></div>""", unsafe_allow_html=True)