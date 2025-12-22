import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image, ImageOps
import cv2

# --- 1. SETUP HALAMAN (WAJIB DI BARIS PERTAMA) ---
st.set_page_config(
    page_title="COFFEE QUALITY CONTROL",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="☕"
)

# --- INFO APLIKASI ---
"""
 QUALITY CONTROL SYSTEM v2.0
"""

# --- 2. CONFIG & ASSETS ---
CLASS_NAMES = ['defect', 'longberry', 'peaberry', 'premium']

MODEL_FILES = {
    "Custom CNN": "./src/Model/model_custom_cnn.keras",
    "MobileNetV2": "./src/Model/model_mobilenetv2.keras",
    "ResNet50V2": "./src/Model/model_resnet50v2.keras"
}

HISTORY_FILES = {
    "Custom CNN": "./src/History/history_custom_cnn.csv",
    "MobileNetV2": "./src/History/history_mobilenetv2.csv",
    "ResNet50V2": "./src/History/history_resnet50v2.csv"
}

# WARNA TEMA KOPI (Earth Tones)
COLOR_MAP = {
    "Custom CNN": "#D4A373",    # Caramel / Light Roast
    "MobileNetV2": "#CCD5AE",   # Sage Green / Raw Bean
    "ResNet50V2": "#6F4E37"     # Coffee Brown / Dark Roast
}

# --- 3. INJECT CSS TEMA KOPI ---
def inject_coffee_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        /* GLOBAL STYLES - Creamy Background */
        .stApp {
            background-color: #FAF3E0; /* Cream / Latte Foam */
            color: #4A3B32; /* Dark Coffee Text */
            font-family: 'Poppins', sans-serif;
            font-size: 14px;
        }
        
        /* HEADER STYLES */
        h1 { 
            font-size: 28px !important; 
            color: #6F4E37; /* Coffee Brown */
            font-weight: 700; 
            border-bottom: 2px solid #D4A373; 
            padding-bottom: 10px; 
            text-transform: uppercase;
        }
        h2 { font-size: 20px !important; color: #8B5E3C; margin-top: 20px; font-weight: 600; }
        h3 { font-size: 16px !important; color: #A47551; font-weight: 500; }
        
        /* METRIC BOXES - Card Style */
        div[data-testid="metric-container"] {
            background-color: #FFFFFF;
            border: 1px solid #E3D5CA;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #6F4E37;
        }
        label[data-testid="stMetricLabel"] { font-size: 12px !important; color: #8D8D8D; }
        div[data-testid="stMetricValue"] { font-size: 24px !important; color: #6F4E37; font-weight: bold; }

        /* BUTTONS */
        .stButton > button {
            background-color: #6F4E37;
            color: #FFF;
            border: none;
            border-radius: 20px; /* Rounded buttons */
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .stButton > button:hover { 
            background-color: #5D4037; 
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        /* SIDEBAR - Dark Roast Theme */
        section[data-testid="stSidebar"] {
            background-color: #4A3B32; /* Espresso */
        }
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3, 
        section[data-testid="stSidebar"] span, 
        section[data-testid="stSidebar"] p {
            color: #FAF3E0 !important; /* Cream text on dark sidebar */
        }
        section[data-testid="stSidebar"] .stRadio label {
            color: #FAF3E0 !important;
        }

        /* PLOTLY FIX */
        .js-plotly-plot .plotly .main-svg { background: rgba(0,0,0,0) !important; }
        
        /* FOOTER */
        .footer {
            position: fixed; left: 0; bottom: 0; width: 100%;
            background-color: #6F4E37; color: #FAF3E0;
            text-align: center; padding: 10px;
            font-size: 12px !important; z-index: 999;
        }
        </style>
    """, unsafe_allow_html=True)

inject_coffee_styles()

# --- 4. BACKEND FUNCTIONS (CACHE) ---

@st.cache_resource
def load_model_cached(model_name):
    filename = MODEL_FILES.get(model_name)
    if not filename or not os.path.exists(filename):
        return None
    return tf.keras.models.load_model(filename)

def preprocess_image(image, target_size=(128, 128)):
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = (img_array.astype(np.float32) / 255.0)
    return np.expand_dims(img_array, axis=0)

# --- 5. VISUALIZATION FUNCTIONS (COFFEE THEMED) ---

def plot_radar_comparison():
    categories = ['Accuracy', 'Consistency', 'Speed', 'Efficiency', 'Robustness']
    
    fig = go.Figure()

    # ResNet (Dark Roast/Strong)
    fig.add_trace(go.Scatterpolar(
        r=[0.95, 0.94, 0.60, 0.50, 0.96],
        theta=categories, fill='toself', name='ResNet50V2',
        line_color=COLOR_MAP['ResNet50V2']
    ))
    # MobileNet (Green/Fresh)
    fig.add_trace(go.Scatterpolar(
        r=[0.88, 0.85, 0.95, 0.90, 0.87],
        theta=categories, fill='toself', name='MobileNetV2',
        line_color=COLOR_MAP['MobileNetV2']
    ))
    # Custom (Caramel/Mild)
    fig.add_trace(go.Scatterpolar(
        r=[0.75, 0.70, 0.98, 0.95, 0.72],
        theta=categories, fill='toself', name='Custom CNN',
        line_color=COLOR_MAP['Custom CNN']
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor='#E3D5CA'),
            bgcolor='rgba(255,255,255,0.5)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins', color='#4A3B32', size=12),
        margin=dict(l=40, r=40, t=20, b=20),
        legend=dict(orientation="h", y=-0.1),
        height=300
    )
    return fig

def plot_history_comparison():
    fig = go.Figure()
    has_data = False

    for model_name, filepath in HISTORY_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            has_data = True
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['val_accuracy'],
                mode='lines+markers',
                name=f"{model_name}",
                line=dict(color=COLOR_MAP[model_name], width=3),
                marker=dict(size=6),
                hovertemplate=f"<b>{model_name}</b><br>Epoch: %{{x}}<br>Acc: %{{y:.2%}}<extra></extra>"
            ))

    fig.update_layout(
        title=dict(text="LEARNING CURVE (VALIDATION)", font=dict(size=16, color='#6F4E37')),
        xaxis=dict(title="EPOCHS", color='#8B5E3C', showgrid=True, gridcolor='#E3D5CA'),
        yaxis=dict(title="ACCURACY", color='#8B5E3C', showgrid=True, gridcolor='#E3D5CA', tickformat='.0%'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(255,255,255,0.5)', # Semi transparent white
        legend=dict(orientation="h", y=1.1, x=0.5, xanchor='center', font=dict(color='#4A3B32')),
        height=350,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig, has_data

def plot_peak_performance():
    models = []
    scores = []
    bar_colors = []
    
    for model_name, filepath in HISTORY_FILES.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            peak_acc = df['val_accuracy'].max()
            models.append(model_name)
            scores.append(peak_acc)
            bar_colors.append(COLOR_MAP[model_name])
    
    if not models:
        return None

    fig = go.Figure(go.Bar(
        x=scores,
        y=models,
        orientation='h',
        text=[f"{s:.2%}" for s in scores],
        textposition='auto',
        marker=dict(color=bar_colors, line=dict(color='#FFF', width=2))
    ))

    fig.update_layout(
        title=dict(text="HIGHEST QUALITY SCORE", font=dict(size=16, color='#6F4E37')),
        xaxis=dict(showgrid=False, visible=False, range=[0, 1.1]),
        yaxis=dict(color='#4A3B32', tickfont=dict(size=12)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig

# --- 6. PAGE RENDERING LOGIC ---

def render_sidebar():
    with st.sidebar:
        st.markdown("### ☕ QUALITY CONTROL")
        st.markdown("Dashboard controls for bean analysis.")
        
        st.markdown("---")
        mode = st.radio("SELECT MODE", ["📊 Dashboard Analytics", "📸 Live Inspection"], label_visibility="collapsed")
        
        st.markdown("---")
        st.markdown("### MACHINE STATUS")
        for name, file in MODEL_FILES.items():
            status = "READY" if os.path.exists(file) else "NOT FOUND"
            icon = "✅" if status == "READY" else "❌"
            st.markdown(f"{icon} **{name}**", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
            <div style="font-size: 11px; color: #FAF3E0; opacity: 0.8;">
            <b>SMART COFFEE SYSTEM v2.0</b><br>
            Engineered for precision sorting.
            </div>
        """, unsafe_allow_html=True)
        return mode

def render_dashboard():
    st.title("📊 QUALITY CONTROL DASHBOARD")
    st.markdown("Overview of AI model performance in sorting coffee beans.")

    # KPI Metrics
    best_model = "N/A"
    best_acc = 0.0
    for name, path in HISTORY_FILES.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df['val_accuracy'].max() > best_acc:
                best_acc = df['val_accuracy'].max()
                best_model = name

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TOP PERFORMER", best_model)
    c2.metric("QUALITY SCORE", f"{best_acc:.2%}")
    c3.metric("MODELS ACTIVE", str(len(MODEL_FILES)))
    c4.metric("BEAN TYPES", str(len(CLASS_NAMES)))

    st.markdown("---")

    # Main Visuals
    st.markdown("### 📈 MODEL PERFORMANCE HISTORY")
    fig_hist, has_data = plot_history_comparison()
    if has_data:
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("⚠️ Data history belum tersedia. Silakan jalankan training di Notebook.")

    # Bottom Visuals
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown("### 🎯 MODEL CHARACTERISTICS")
        st.plotly_chart(plot_radar_comparison(), use_container_width=True)

    with col_right:
        st.markdown("### 🏆 ACCURACY LEADERBOARD")
        fig_peak = plot_peak_performance()
        if fig_peak:
            st.plotly_chart(fig_peak, use_container_width=True)
        else:
            st.write("No data available.")

def render_inference():
    st.title("📸 LIVE BEAN INSPECTION")
    st.markdown("Upload coffee bean image for instant quality classification.")
    
    col_input, col_output = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("#### ⚙️ CONFIGURATION")
        # Custom style for box
        st.markdown("""
        <div style="background-color: #FFF; padding: 15px; border-radius: 10px; border: 1px solid #E3D5CA;">
            Select the neural network architecture for inspection.
        </div><br>
        """, unsafe_allow_html=True)
        
        selected_model_name = st.selectbox("SELECT MODEL ARCHITECTURE", list(MODEL_FILES.keys()))
        
        uploaded_file = st.file_uploader("UPLOAD BEAN IMAGE", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            st.success("Image Uploaded Successfully!")
        else:
            st.info("Waiting for image...")

    with col_output:
        st.markdown("#### 🔍 INSPECTION RESULT")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            model = load_model_cached(selected_model_name)
            
            if model:
                processed_img = preprocess_image(image)
                try:
                    preds = model.predict(processed_img)
                    idx = np.argmax(preds)
                    label = CLASS_NAMES[idx]
                    score = np.max(preds)
                    
                    # --- COFFEE THEMED HUD ---
                    img_cv = np.array(image) 
                    h, w, _ = img_cv.shape
                    
                    # Colors: Green for Good, Red/Orange for Defect?
                    # Let's stick to Coffee Theme: Dark Brown for Box, White Text
                    color = (55, 78, 111) # Dark Brown in BGR (OpenCV uses BGR)
                    if label == 'defect':
                        color = (50, 50, 200) # Red-ish for defect
                    
                    # Box Logic
                    center_x, center_y = w // 2, h // 2
                    box_w, box_h = w // 3, h // 3
                    
                    cv2.rectangle(img_cv, (center_x - box_w, center_y - box_h), 
                                  (center_x + box_w, center_y + box_h), color, 4)
                    
                    # Label Background
                    cv2.rectangle(img_cv, (center_x - box_w, center_y - box_h - 40), 
                                          (center_x + box_w, center_y - box_h), color, -1)
                    
                    cv2.putText(img_cv, f"{label.upper()} ({score:.0%})", 
                                (center_x - box_w + 10, center_y - box_h - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    st.image(img_cv, caption="AI INSPECTION VISUALIZATION", use_container_width=True)
                    
                    # Result Metrics
                    st.success(f"### CLASSIFICATION: {label.upper()}")
                    st.metric("CONFIDENCE SCORE", f"{score:.2%}")
                    
                    # Probability Bar (Coffee Tones)
                    st.markdown("#### PROBABILITY DISTRIBUTION")
                    prob_df = pd.DataFrame({'Class': CLASS_NAMES, 'Probability': preds[0]})
                    fig_prob = px.bar(prob_df, y='Class', x='Probability', orientation='h',
                                      text_auto='.1%', 
                                      color='Probability',
                                      color_continuous_scale=['#FAF3E0', '#6F4E37']) # Cream to Brown
                    
                    fig_prob.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                           font=dict(family='Poppins', color='#4A3B32'), 
                                           height=200, margin=dict(l=0, r=0, t=0, b=0),
                                           xaxis=dict(range=[0,1]))
                    st.plotly_chart(fig_prob, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
            else:
                st.error("Model file not found.")
        else:
            # Placeholder Image or Text
            st.markdown("""
            <div style="background-color: #FAF3E0; border: 2px dashed #D4A373; border-radius: 10px; height: 300px; display: flex; align-items: center; justify-content: center; color: #8B5E3C;">
                <h3>☕ Upload Image to Start Inspection</h3>
            </div>
            """, unsafe_allow_html=True)

# --- MAIN EXECUTION ---
def main():
    mode = render_sidebar()
    if mode == "📊 Dashboard Analytics":
        render_dashboard()
    elif mode == "📸 Live Inspection":
        render_inference()
    
    st.markdown("""
        <div class="footer">
            SMART COFFEE SYSTEM v2.0 | Developed for Precision Agriculture
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()