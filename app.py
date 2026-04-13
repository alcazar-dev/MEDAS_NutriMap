import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# ── Windows UTF-8 fix for kmapper ─────────────────────────────────────────────
import pathlib
_orig_read_text = pathlib.Path.read_text
def _read_text_utf8(self, encoding=None, errors=None, **kw):
    return _orig_read_text(self, encoding=encoding or 'utf-8', errors=errors or 'replace', **kw)
pathlib.Path.read_text = _read_text_utf8

import kmapper as km  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriMap · Clasificador de dieta",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        line-height: 1.15;
        color: #1a1a2e;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        font-weight: 300;
        margin-top: -0.5rem;
    }
    .diet-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #e8e4f0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .diet-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .badge-plant { background: #e8f5e9; color: #2e7d32; }
    .badge-med { background: #e3f2fd; color: #1565c0; }
    .badge-lowcarb { background: #fff3e0; color: #e65100; }
    .badge-therapeutic { background: #f3e5f5; color: #6a1b9a; }

    .result-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .result-hero h2 { color: white; font-family: 'DM Serif Display', serif; font-size: 2rem; }
    .result-hero p { color: rgba(255,255,255,0.85); }

    .metric-pill {
        background: #f8f7ff;
        border: 1px solid #e0dbf7;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .metric-pill .val { font-size: 1.5rem; font-weight: 600; color: #4c3d8f; }
    .metric-pill .lbl { font-size: 0.75rem; color: #999; text-transform: uppercase; letter-spacing: 0.05em; }

    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #1a1a2e;
        border-bottom: 2px solid #f0edf8;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }

    div[data-testid="stSidebar"] { background: #faf9ff; }
    .sidebar-logo {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #4c3d8f;
        text-align: center;
        padding: 1rem 0;
    }
    .import-box {
        background: #f8f7ff;
        border: 2px dashed #c5bbf0;
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .cluster-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 0.5rem;
    }
    .cluster-chip {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Data persistence ──────────────────────────────────────────────────────────
DATA_FILE = "survey_responses.json"

def load_responses():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_response(entry):
    responses = load_responses()
    responses.append(entry)
    with open(DATA_FILE, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

def save_all_responses(responses):
    with open(DATA_FILE, "w") as f:
        json.dump(responses, f, indent=2, ensure_ascii=False)

# ── Import helpers ────────────────────────────────────────────────────────────
REQUIRED_FEATURE_COLS = [
    "meat_freq", "veg_freq", "fruit_freq", "grain_freq",
    "fat_pref", "medical_cond", "env_concern", "sodium_concern",
    "protein_pref", "variety_pref"
]

def normalize_imported_row(row: dict) -> dict | None:
    """
    Convert a flat dict (from CSV/JSON import) into the internal response format.
    Returns None if the row is missing required feature columns.
    """
    answers = {}
    for col in REQUIRED_FEATURE_COLS:
        if col not in row:
            return None
        try:
            answers[col] = int(float(row[col]))
        except (ValueError, TypeError):
            return None

    result = classify_diet(answers)

    return {
        "timestamp": str(row.get("timestamp", datetime.now().isoformat())),
        "name": str(row.get("name", "Imported")),
        "age": int(float(row.get("age", 30))),
        "gender": str(row.get("gender", "Unknown")),
        "answers": answers,
        "allergies": row.get("allergies", []) if isinstance(row.get("allergies"), list)
                     else [x.strip() for x in str(row.get("allergies", "")).split(",") if x.strip()],
        "goals": row.get("goals", []) if isinstance(row.get("goals"), list)
                 else [x.strip() for x in str(row.get("goals", "")).split(",") if x.strip()],
        "diet_key": result["diet_key"],
        "diet_name": result["diet_name"],
        "confidence": result["confidence"],
        "scores": result["scores"],
        "feature_vector": result["feature_vector"],
        "_imported": True,
    }

# ── Diet classification engine ────────────────────────────────────────────────
DIET_PROFILES = {
    "plant_based": {
        "name": "Plant-Based",
        "subtitle": "Vegetarian / Vegan",
        "emoji": "⚫",
        "badge_class": "badge-plant",
        "color": "#2e7d32",
        "bg": "#e8f5e9",
        "description": "Focused on fruits, vegetables, legumes, and whole grains. Eliminates or minimizes animal products.",
        "foods": ["Legumes & beans", "Whole grains", "Nuts & seeds", "Leafy greens", "Seasonal fruits"],
        "avoid": ["Red meat", "Poultry", "Dairy (vegan)", "Processed foods"],
        "health_focus": ["Cardiovascular health", "Weight management", "Environmental sustainability"],
        "profile": np.array([0, 5, 5, 5, 1, 0, 4, 3, 2, 2])
    },
    "mediterranean": {
        "name": "Mediterranean",
        "subtitle": "Heart-Healthy",
        "emoji": "⚫",
        "badge_class": "badge-med",
        "color": "#1565c0",
        "bg": "#e3f2fd",
        "description": "Rich in olive oil, fish, vegetables, and moderate wine. Inspired by southern European dietary patterns.",
        "foods": ["Olive oil", "Fish & seafood", "Whole grains", "Legumes", "Fresh vegetables", "Red wine (moderate)"],
        "avoid": ["Processed meats", "Refined sugars", "Processed foods"],
        "health_focus": ["Heart health", "Brain health", "Longevity", "Anti-inflammatory"],
        "profile": np.array([3, 4, 3, 4, 2, 1, 3, 3, 3, 3])
    },
    "low_carb": {
        "name": "Low-Carb",
        "subtitle": "Ketogenic / Atkins",
        "emoji": "⚫",
        "badge_class": "badge-lowcarb",
        "color": "#e65100",
        "bg": "#fff3e0",
        "description": "Restricts carbohydrates to induce ketosis. High in protein and healthy fats.",
        "foods": ["Meat & poultry", "Fish", "Eggs", "Cheese", "Nuts", "Low-carb vegetables"],
        "avoid": ["Bread & pasta", "Rice", "Sugar", "Most fruits", "Starchy vegetables"],
        "health_focus": ["Weight loss", "Blood sugar control", "Mental clarity", "Energy stability"],
        "profile": np.array([5, 1, 1, 2, 4, 2, 2, 1, 4, 1])
    },
    "therapeutic": {
        "name": "Therapeutic",
        "subtitle": "Medical / Clinical",
        "emoji": "⚫",
        "badge_class": "badge-therapeutic",
        "color": "#6a1b9a",
        "bg": "#f3e5f5",
        "description": "Medically tailored for specific conditions: lower sodium, texture-modified, renal, or diabetic diets.",
        "foods": ["Low-sodium options", "Soft/pureed foods", "Controlled portions", "Physician-approved items"],
        "avoid": ["High-sodium foods", "Hard textures (if texture-modified)", "Trigger foods for condition"],
        "health_focus": ["Disease management", "Medication interaction", "Clinical nutrition"],
        "profile": np.array([2, 2, 2, 3, 2, 5, 1, 4, 2, 3])
    }
}

def classify_diet(answers: dict) -> dict:
    vec = np.array([
        answers.get("meat_freq", 0),
        answers.get("veg_freq", 0),
        answers.get("fruit_freq", 0),
        answers.get("grain_freq", 0),
        answers.get("fat_pref", 0),
        answers.get("medical_cond", 0),
        answers.get("env_concern", 0),
        answers.get("sodium_concern", 0),
        answers.get("protein_pref", 0),
        answers.get("variety_pref", 0),
    ], dtype=float)

    scores = {}
    for key, diet in DIET_PROFILES.items():
        profile = diet["profile"].astype(float)
        dot = np.dot(vec, profile)
        norm = np.linalg.norm(vec) * np.linalg.norm(profile)
        scores[key] = (dot / norm) if norm > 0 else 0

    best = max(scores, key=scores.get)
    confidence = round(scores[best] * 100, 1)
    scores_pct = {k: round(v * 100, 1) for k, v in scores.items()}

    return {
        "diet_key": best,
        "diet_name": DIET_PROFILES[best]["name"],
        "confidence": confidence,
        "scores": scores_pct,
        "feature_vector": vec.tolist()
    }

# ── Cluster-to-diet labeling ──────────────────────────────────────────────────
# Each cluster gets labeled by which food features dominate it.
# We compare the cluster centroid against the diet profile vectors.

CLUSTER_PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
                   "#f39c12", "#16a085", "#d35400", "#2c3e50"]

def label_cluster_from_centroid(centroid: np.ndarray) -> tuple[str, str]:
    """
    Given a cluster centroid in feature space, return (diet_key, diet_name)
    by cosine similarity to the four DIET_PROFILES vectors.
    """
    best_key = None
    best_sim = -1
    for key, diet in DIET_PROFILES.items():
        profile = diet["profile"].astype(float)
        norm = np.linalg.norm(centroid) * np.linalg.norm(profile)
        sim = np.dot(centroid, profile) / norm if norm > 0 else 0
        if sim > best_sim:
            best_sim = sim
            best_key = key
    return best_key, DIET_PROFILES[best_key]["name"]

def describe_cluster(centroid: np.ndarray) -> str:
    """
    Build a short human-readable description of a cluster's dominant food pattern.
    """
    labels = ["Meat", "Veg", "Fruit", "Grains", "Fat", "Medical", "Env", "Sodium", "Protein", "Variety"]
    top_idx = np.argsort(centroid)[::-1][:3]
    top_features = [labels[i] for i in top_idx if centroid[i] >= 2.5]
    if top_features:
        return "High: " + ", ".join(top_features)
    return "Balanced / low intake"

# ── Topology analysis (cluster-driven) ───────────────────────────────────────
def run_topology_analysis(df_features: pd.DataFrame, n_clusters: int = 4):
    """
    Run KMapper + PCA.
    Returns (X_pca, graph, pca, kmeans_model, cluster_labels_per_node)
    where cluster_labels_per_node maps node_id → (diet_key, diet_name, description).
    """
    if len(df_features) < 5:
        return None, None, None, None, {}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)

    # PCA lens
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Fit a global KMeans so cluster centroids are in *original* feature scale
    # (not scaled) — more interpretable for food labels
    kmeans_global = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_global.fit(df_features.values)

    # KMapper — uses KMeans internally per cover patch
    mapper = km.KeplerMapper(verbose=0)
    lens = mapper.fit_transform(X_pca, projection=[0])

    graph = mapper.map(
        lens,
        X_scaled,
        clusterer=KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        cover=km.Cover(n_cubes=8, perc_overlap=0.4)
    )

    # For each node compute the centroid of its member vectors (raw scale)
    # then assign diet label via cosine similarity
    cluster_labels_per_node = {}
    for node_id, member_indices in graph.get("nodes", {}).items():
        valid = [i for i in member_indices if i < len(df_features)]
        if valid:
            centroid = df_features.values[valid].mean(axis=0)
            diet_key, diet_name = label_cluster_from_centroid(centroid)
            desc = describe_cluster(centroid)
            cluster_labels_per_node[node_id] = {
                "diet_key": diet_key,
                "diet_name": diet_name,
                "description": desc,
                "centroid": centroid,
                "n_members": len(valid),
            }

    return X_pca, graph, pca, kmeans_global, cluster_labels_per_node

# -- Data Management with Session State ----------------------------------------
def get_active_data():
    # Priorizamos el archivo subido por el usuario
    if uploaded_file is not None:
        try:
            import_df = pd.read_csv(uploaded_file)
            raw_rows = import_df.to_dict(orient="records")
            valid_entries = []
            for row in raw_rows:
                # Reutilizamos tu lógica de normalización
                entry = normalize_imported_row(row)
                if entry:
                    valid_entries.append(entry)
            return valid_entries
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV: {e}")
            return []
    # Si no hay carga, regresamos la base de datos local
    return load_responses()

# Inicializar estados de sesión para evitar cálculos redundantes
if 'topology_results' not in st.session_state:
    st.session_state.topology_results = None
if 'last_data_hash' not in st.session_state:
    st.session_state.last_data_hash = None


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌿 NutriMap</div>', unsafe_allow_html=True)
    
    st.markdown("### Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu CSV de NutriMap", type=["csv"], help="Exporta desde tu Excel y súbelo aquí.")
    
    st.markdown("---")
    page = st.radio(
        "Navegación",
        ["⚪ Dashboard", "⚪ Topology", "⚪ Survey", "📥 Import Data", "📁 Export Data"],
        label_visibility="collapsed"
    )
    
    responses = get_active_data()
    # Generamos un hash simple para saber si los datos cambiaron
    current_data_hash = hash(str(responses))
    
    st.markdown("---")
    st.markdown(f"**Registros activos:** {len(responses)}")

# =============================================================================
# PAGE: SURVEY
# =============================================================================
if "📋 Survey" in page:
    col_title, _ = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="main-title">Food Preferences Survey</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Answer honestly — we\'ll find your ideal dietary pattern.</p>', unsafe_allow_html=True)

    st.markdown("---")

    with st.form("survey_form"):
        st.markdown('<p class="section-header">About you</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name (optional)", placeholder="Anonymous")
        with c2:
            age = st.number_input("Age", min_value=10, max_value=100, value=30)
        with c3:
            gender = st.selectbox("Gender", ["Prefer not to say", "Female", "Male", "Non-binary", "Other"])

        st.markdown('<p class="section-header">Dietary habits</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            meat_freq = st.select_slider(
                "How often do you eat meat (beef, pork, chicken)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Never", "Rarely", "1-2x/week", "3-4x/week", "Daily", "Multiple/day"][x],
                value=2
            )
            veg_freq = st.select_slider(
                "How often do you eat vegetables?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often", "Daily", "Every meal"][x],
                value=3
            )
            fruit_freq = st.select_slider(
                "How often do you eat fresh fruit?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Never", "Rarely", "Sometimes", "Often", "Daily", "Multiple/day"][x],
                value=3
            )
            grain_freq = st.select_slider(
                "Grains, legumes & beans consumption?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["None", "Very low", "Low", "Moderate", "High", "Very high"][x],
                value=3
            )

        with c2:
            fat_pref = st.select_slider(
                "Preference for fats (butter, oils, fatty meats)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Avoid all fats", "Very low fat", "Low fat", "Moderate", "High fat", "Keto-level"][x],
                value=2
            )
            protein_pref = st.select_slider(
                "Importance of protein in your diet?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Not important", "Low priority", "Below average", "Average", "High priority", "Primary focus"][x],
                value=3
            )
            variety_pref = st.select_slider(
                "How much dietary variety do you prefer?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Very rigid", "Little variety", "Some variety", "Moderate", "High variety", "Maximally varied"][x],
                value=3
            )
            env_concern = st.select_slider(
                "How important is environmental impact in food choices?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Not at all", "Slightly", "Somewhat", "Moderately", "Very important", "Primary driver"][x],
                value=2
            )

        st.markdown('<p class="section-header">Health & medical</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sodium_concern = st.select_slider(
                "Need to limit sodium / salt intake?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["No concern", "Slight care", "Some care", "Moderate care", "High concern", "Medical restriction"][x],
                value=1
            )
            medical_cond = st.select_slider(
                "Do you have medical dietary conditions (diabetes, renal, dysphagia, etc.)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["None", "Minor", "Mild condition", "Moderate", "Significant", "Severe/complex"][x],
                value=0
            )
        with c2:
            allergies = st.multiselect(
                "Allergies or intolerances",
                ["Gluten", "Lactose/Dairy", "Nuts", "Eggs", "Soy", "Shellfish", "None"],
                default=["None"]
            )
            goals = st.multiselect(
                "Primary health goals",
                ["Weight loss", "Muscle gain", "Heart health", "Blood sugar control",
                 "Anti-inflammatory", "Digestive health", "Energy levels", "General wellness"],
                default=["General wellness"]
            )

        st.markdown("---")
        submitted = st.form_submit_button("Classify my diet", use_container_width=True)

    if submitted:
        answers = {
            "meat_freq": meat_freq, "veg_freq": veg_freq,
            "fruit_freq": fruit_freq, "grain_freq": grain_freq,
            "fat_pref": fat_pref, "medical_cond": medical_cond,
            "env_concern": env_concern, "sodium_concern": sodium_concern,
            "protein_pref": protein_pref, "variety_pref": variety_pref
        }

        result = classify_diet(answers)
        diet_info = DIET_PROFILES[result["diet_key"]]

        entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name or "Anonymous",
            "age": int(age),
            "gender": gender,
            "answers": answers,
            "allergies": allergies,
            "goals": goals,
            "diet_key": result["diet_key"],
            "diet_name": result["diet_name"],
            "confidence": result["confidence"],
            "scores": result["scores"],
            "feature_vector": result["feature_vector"]
        }
        save_response(entry)

        st.success("Response saved! Here is your result:")
        st.markdown("---")

        st.markdown(f"""
        <div class="result-hero">
            <div style="font-size:3rem; margin-bottom:0.5rem;">{diet_info['emoji']}</div>
            <h2>{diet_info['name']}</h2>
            <p style="font-size:1.1rem;">{diet_info['subtitle']}</p>
            <p style="font-size:0.9rem; margin-top:1rem;">{diet_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Similarity to each diet profile")
        cols = st.columns(4)
        for i, (key, diet) in enumerate(DIET_PROFILES.items()):
            with cols[i]:
                score = result["scores"][key]
                is_best = key == result["diet_key"]
                st.markdown(f"""
                <div class="metric-pill" style="border: {'2px solid ' + diet['color'] if is_best else '1px solid #e0dbf7'}">
                    <div style="font-size:1.5rem;">{diet['emoji']}</div>
                    <div class="val" style="color:{diet['color']}">{score}%</div>
                    <div class="lbl">{diet['name']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Recommended foods**")
            for f in diet_info["foods"]:
                st.markdown(f"- {f}")
        with c2:
            st.markdown("**Limit or avoid**")
            for f in diet_info["avoid"]:
                st.markdown(f"- {f}")
        with c3:
            st.markdown("**Health focus**")
            for f in diet_info["health_focus"]:
                st.markdown(f"- {f}")


# =============================================================================
# PAGE: DASHBOARD
# =============================================================================
elif "Dashboard" in page:
    st.markdown('<h1 class="main-title">Statistics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Aggregated insights from all survey responses.</p>', unsafe_allow_html=True)
    st.markdown("---")

    responses = load_responses()

    if len(responses) == 0:
        st.info("No responses yet. Complete the survey or import data to start collecting.")
        st.stop()

    # 1. Definimos las columnas que usaremos (usando tu lista global)
    feature_cols = REQUIRED_FEATURE_COLS 
    
    # 2. Extraemos los nombres de las dietas de los datos actuales para los labels del gráfico
    diet_names = [r["diet_name"] for r in responses]
    
    # 3. Preparamos el DataFrame de respuestas (necesario para el cálculo y el cache)
    answers_df = pd.json_normalize([r["answers"] for r in responses])

    df = pd.DataFrame(responses)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Responses", len(df))
    with c2:
        most_common = df["diet_key"].value_counts().idxmax()
        st.metric("Most Common Diet", DIET_PROFILES[most_common]["name"])
    with c3:
        avg_conf = round(df["confidence"].mean(), 1) if "confidence" in df.columns else "—"
        st.metric("Avg. Confidence", f"{avg_conf}%")
    with c4:
        today_count = sum(1 for r in responses if r["timestamp"][:10] == datetime.now().date().isoformat())
        st.metric("Responses Today", today_count)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Diet distribution")
        diet_counts = df["diet_key"].value_counts().reset_index()
        diet_counts.columns = ["diet", "count"]
        diet_counts["name"] = diet_counts["diet"].map(lambda k: DIET_PROFILES[k]["name"] + " " + DIET_PROFILES[k]["emoji"])
        colors = [DIET_PROFILES[k]["color"] for k in diet_counts["diet"]]
        fig = px.pie(diet_counts, values="count", names="name",
                     color_discrete_sequence=colors, hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10), height=300,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Age distribution by diet")
        if "age" in df.columns:
            fig2 = px.box(df, x="diet_name", y="age",
                          color="diet_name",
                          color_discrete_map={DIET_PROFILES[k]["name"]: DIET_PROFILES[k]["color"] for k in DIET_PROFILES})
            fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                                xaxis_title="", yaxis_title="Age")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Average feature values per diet")
    feature_cols = ["meat_freq", "veg_freq", "fruit_freq", "grain_freq",
                    "fat_pref", "medical_cond", "env_concern", "sodium_concern",
                    "protein_pref", "variety_pref"]
    feature_labels = ["Meat", "Vegetables", "Fruit", "Grains/Legumes",
                      "Fat Preference", "Medical Conditions", "Env. Concern",
                      "Sodium Concern", "Protein Pref.", "Variety Pref."]

    answers_df = pd.json_normalize(df["answers"])
    answers_df["diet_name"] = df["diet_name"].values

    heatmap_data = answers_df.groupby("diet_name")[feature_cols].mean()
    heatmap_data.columns = feature_labels

    fig3 = px.imshow(heatmap_data.T, color_continuous_scale="RdYlGn",
                     aspect="auto", text_auto=".1f")
    fig3.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top health goals")
        all_goals = [g for r in responses for g in r.get("goals", [])]
        goal_counts = pd.Series(all_goals).value_counts().head(8)
        fig4 = px.bar(x=goal_counts.values, y=goal_counts.index, orientation="h",
                      color_discrete_sequence=["#667eea"])
        fig4.update_layout(height=280, margin=dict(t=10, b=10),
                            xaxis_title="Count", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        st.markdown("#### Allergy / intolerance frequency")
        all_allergies = [a for r in responses for a in r.get("allergies", []) if a != "None"]
        if all_allergies:
            allergy_counts = pd.Series(all_allergies).value_counts()
            fig5 = px.bar(x=allergy_counts.index, y=allergy_counts.values,
                          color_discrete_sequence=["#764ba2"])
            fig5.update_layout(height=280, margin=dict(t=10, b=10),
                                xaxis_title="", yaxis_title="Count")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No allergies reported yet.")

    st.markdown("#### Recent responses")
    display_cols = ["timestamp", "name", "age", "gender", "diet_name", "confidence"]
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[available].tail(20).rename(columns={
            "timestamp": "Date", "name": "Name", "age": "Age",
            "gender": "Gender", "diet_name": "Diet", "confidence": "Match %"
        }).iloc[::-1],
        use_container_width=True
    )


# =============================================================================
# PAGE: TOPOLOGY (cluster-based mapper)
# =============================================================================
elif "⚪ Topology" in page:
    st.markdown('<h1 class="main-title">Topological Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KMapper + PCA · cada nodo coloreado por el <em>cluster de patrones alimentarios</em> que predomina en él.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Usamos la función de carga dinámica que definimos antes
    responses = get_active_data()

    if len(responses) < 5:
        st.warning(f"Se necesitan al menos 5 respuestas para el análisis topológico. Actualmente: {len(responses)}")
        st.stop()

    # --- ESTO SOLUCIONA LOS ERRORES DE PYLANCE ---
    # 1. Definimos feature_cols usando tu lista global predefinida [cite: 27]
    feature_cols = REQUIRED_FEATURE_COLS 
    
    # 2. Generamos el DataFrame necesario para PCA/KMapper 
    answers_df = pd.json_normalize([r["answers"] for r in responses])
    
    # 3. Extraemos los nombres de las dietas para el hover de los gráficos [cite: 91]
    diet_names = [r["diet_name"] for r in responses]
    # ---------------------------------------------

    # --- Parámetros de la Topología ---
    with st.expander("Topology parameters", expanded=False):
        n_clusters = st.slider("Number of food-pattern clusters (KMeans)", 2, 8, 4)
        n_cubes = st.slider("KMapper cover cubes", 4, 20, 8)
        perc_overlap = st.slider("Cover overlap %", 0.1, 0.8, 0.4, step=0.05)

    # --- Lógica de Cache (Session State) ---
    # Asegúrate de que current_data_hash se genere en el sidebar o aquí mismo
    current_data_hash = hash(str(responses))
    params_key = f"{current_data_hash}_{n_clusters}_{n_cubes}_{perc_overlap}"

    if st.session_state.topology_results is None or st.session_state.last_data_hash != params_key:
        with st.spinner("Ejecutando algoritmos..."):
            results = run_topology_analysis(
                answers_df[feature_cols], 
                n_clusters=n_clusters
            )
            st.session_state.topology_results = results
            st.session_state.last_data_hash = params_key

    # Recuperamos los resultados del cache
    X_pca, graph, pca, kmeans_model, cluster_labels_per_node = st.session_state.topology_results

    if X_pca is None:
        st.error("No se pudo ejecutar el análisis. Verifica los datos.")
        st.stop()

    # ── Cluster centroid radar (what each cluster eats) ───────────────────────
    st.markdown("#### Food-pattern clusters · what each cluster eats")
    centroids = kmeans_model.cluster_centers_
    feature_labels_short = ["Meat", "Veg", "Fruit", "Grains", "Fat", "Medical", "Env", "Sodium", "Protein", "Variety"]

    centroid_df = pd.DataFrame(centroids, columns=feature_labels_short)
    centroid_df.index = [f"Cluster {i}" for i in range(len(centroids))]

    # Assign diet label to each cluster centroid
    cluster_diet_labels = {}
    for i, row in centroid_df.iterrows():
        diet_key, diet_name = label_cluster_from_centroid(row.values)
        cluster_diet_labels[i] = {"diet_key": diet_key, "diet_name": diet_name,
                                   "emoji": DIET_PROFILES[diet_key]["emoji"],
                                   "color": DIET_PROFILES[diet_key]["color"]}

    # Heatmap of centroids
    fig_cent = px.imshow(
        centroid_df.T,
        color_continuous_scale="YlOrRd",
        text_auto=".1f",
        aspect="auto",
        labels=dict(color="Avg score")
    )
    # Annotate columns with diet labels
    cluster_annotations = [
        f"{cluster_diet_labels[f'Cluster {i}']['emoji']} {cluster_diet_labels[f'Cluster {i}']['diet_name']}"
        for i in range(len(centroids))
    ]
    fig_cent.update_layout(
        height=360,
        margin=dict(t=40, b=20),
        xaxis=dict(
            tickvals=list(range(len(centroids))),
            ticktext=cluster_annotations
        )
    )
    st.plotly_chart(fig_cent, use_container_width=True)
    st.caption("Cada columna es un cluster de patrones alimentarios. El emoji/etiqueta indica el tipo de dieta más cercano al centroide del cluster.")

    # ── PCA scatter — colored by cluster ────────────────────────────────────
    st.markdown("#### PCA · Espacio de preferencias alimentarias (color = cluster)")
    col_info, col_plot = st.columns([1, 3])

    # Assign each respondent to a KMeans cluster
    global_cluster_ids = kmeans_model.predict(answers_df[feature_cols].values)
    cluster_diet_names = [
        f"{cluster_diet_labels[f'Cluster {c}']['emoji']} {cluster_diet_labels[f'Cluster {c}']['diet_name']} (C{c})"
        for c in global_cluster_ids
    ]
    cluster_colors_map = {
        f"{cluster_diet_labels[f'Cluster {c}']['emoji']} {cluster_diet_labels[f'Cluster {c}']['diet_name']} (C{c})":
        cluster_diet_labels[f'Cluster {c}']['color']
        for c in range(n_clusters)
    }

    with col_info:
        var = pca.explained_variance_ratio_
        st.markdown(f"**PC1:** {round(var[0]*100, 1)}% variance")
        st.markdown(f"**PC2:** {round(var[1]*100, 1)}% variance")
        st.markdown(f"**Total:** {round(sum(var)*100, 1)}%")
        st.markdown("---")
        st.caption("Cada punto = un respondente. Color = cluster alimentario asignado → tipo de dieta.")

    with col_plot:
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Cluster"] = cluster_diet_names
        pca_df["Diet (assigned)"] = diet_names

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="Cluster",
            color_discrete_map=cluster_colors_map,
            hover_data={"Diet (assigned)": True, "Cluster": True},
            labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
        )
        fig_pca.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="white")))
        fig_pca.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── KMapper graph — colored by cluster diet ───────────────────────────────
    st.markdown("#### KMapper · Topological graph (nodos coloreados por cluster alimentario)")

    if graph and graph.get("nodes"):
        nodes = list(graph["nodes"].keys())
        edges = []
        for src, targets in graph.get("links", {}).items():
            for tgt in targets:
                edges.append((src, tgt))

        n = len(nodes)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {node: (np.cos(a), np.sin(a)) for node, a in zip(nodes, angles)}

        node_colors, node_sizes, node_hover = [], [], []

        for node in nodes:
            info = cluster_labels_per_node.get(node, {})
            diet_key = info.get("diet_key", "plant_based")
            n_members = info.get("n_members", 1)
            desc = info.get("description", "")
            diet_name = info.get("diet_name", "Unknown")
            emoji = DIET_PROFILES[diet_key]["emoji"]
            color = DIET_PROFILES[diet_key]["color"]

            node_colors.append(color)
            node_sizes.append(max(12, n_members * 6))
            node_hover.append(
                f"{emoji} {diet_name}<br>n={n_members}<br>{desc}"
            )

        edge_x, edge_y = [], []
        for src, tgt in edges:
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[nd][0] for nd in nodes]
        node_y = [pos[nd][1] for nd in nodes]

        fig_mapper = go.Figure()
        fig_mapper.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color="#ddd", width=1.2), hoverinfo="none"
        ))
        fig_mapper.add_trace(go.Scatter(
            x=node_x, y=node_y, mode="markers",
            marker=dict(size=node_sizes, color=node_colors,
                        line=dict(color="white", width=2)),
            text=node_hover,
            hovertemplate="%{text}<extra></extra>"
        ))

        # Diet legend
        seen = set()
        for node in nodes:
            info = cluster_labels_per_node.get(node, {})
            dk = info.get("diet_key", "plant_based")
            if dk not in seen:
                seen.add(dk)
                d = DIET_PROFILES[dk]
                fig_mapper.add_trace(go.Scatter(
                    x=[None], y=[None], mode="markers",
                    marker=dict(color=d["color"], size=13),
                    name=f"{d['emoji']} {d['name']}"
                ))

        fig_mapper.update_layout(
            height=480,
            showlegend=True,
            legend=dict(orientation="h", y=-0.15, font=dict(size=12)),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(t=20, b=90),
            plot_bgcolor="rgba(248,247,255,0.6)"
        )
        st.plotly_chart(fig_mapper, use_container_width=True)

        st.caption(
            f"**{len(nodes)} nodos · {len(edges)} aristas.** "
            "Tamaño del nodo = número de respondentes. "
            "Color = tipo de dieta inferido del centroide alimentario del cluster que domina el nodo. "
            "Pasa el cursor sobre un nodo para ver detalles."
        )
    else:
        st.info("El grafo KMapper está vacío — intenta recolectar respuestas más diversas o ajusta los parámetros.")

    # ── PCA loadings heatmap ───────────────────────────────────────────────────
    st.markdown("#### PCA loadings · qué features separan los datos")
    loading_df = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=["Meat", "Vegetables", "Fruit", "Grains/Legumes", "Fat Pref.",
               "Medical Cond.", "Env. Concern", "Sodium Concern", "Protein Pref.", "Variety Pref."]
    )
    fig_load = px.imshow(loading_df, color_continuous_scale="RdBu_r",
                         text_auto=".2f", aspect="auto",
                         color_continuous_midpoint=0)
    fig_load.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_load, use_container_width=True)


# =============================================================================
# PAGE: IMPORT DATA  (NEW)
# =============================================================================
elif "Import Data" in page:
    st.markdown('<h1 class="main-title">Import Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Carga respuestas previas desde CSV o JSON para alimentar la base de datos.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Format reference ──────────────────────────────────────────────────────
    with st.expander("Formato esperado de columnas", expanded=False):
        st.markdown("""
**Columnas requeridas** (valores enteros 0–5):

| Columna | Descripción |
|---|---|
| `meat_freq` | Frecuencia de consumo de carne |
| `veg_freq` | Frecuencia de vegetales |
| `fruit_freq` | Frecuencia de fruta |
| `grain_freq` | Consumo de granos/leguminosas |
| `fat_pref` | Preferencia de grasas |
| `medical_cond` | Condición médica dietética |
| `env_concern` | Preocupación ambiental |
| `sodium_concern` | Restricción de sodio |
| `protein_pref` | Preferencia de proteína |
| `variety_pref` | Variedad dietética |

**Columnas opcionales:** `name`, `age`, `gender`, `timestamp`, `allergies`, `goals`

Para CSV: `allergies` y `goals` pueden ser strings separados por comas, e.g. `"Gluten,Nuts"`.
""")
        # Sample CSV download
        sample_data = {
            "name": ["Alice", "Bob"],
            "age": [28, 45],
            "gender": ["Female", "Male"],
            "meat_freq": [0, 5],
            "veg_freq": [5, 1],
            "fruit_freq": [4, 2],
            "grain_freq": [5, 1],
            "fat_pref": [1, 4],
            "medical_cond": [0, 2],
            "env_concern": [4, 1],
            "sodium_concern": [1, 3],
            "protein_pref": [2, 4],
            "variety_pref": [4, 2],
            "allergies": ["Gluten", "None"],
            "goals": ["Heart health,General wellness", "Weight loss"],
        }
        st.download_button(
            "Descargar CSV de ejemplo",
            pd.DataFrame(sample_data).to_csv(index=False),
            "nutrimap_sample.csv",
            "text/csv"
        )

    st.markdown("---")

    # ── Upload widget ──────────────────────────────────────────────────────────
    st.markdown('<div class="import-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV o JSON",
        type=["csv", "json"],
        help="CSV con columnas separadas por coma, o JSON exportado por NutriMap (array de objetos)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()

        try:
            if file_ext == "csv":
                import_df = pd.read_csv(uploaded_file)
                raw_rows = import_df.to_dict(orient="records")
            else:  # json
                raw_data = json.load(uploaded_file)
                # Accept both array-of-objects and array-of-full-NutriMap-entries
                if isinstance(raw_data, list):
                    raw_rows = raw_data
                else:
                    st.error("El JSON debe ser un array de objetos.")
                    st.stop()

            st.success(f"Archivo cargado: **{len(raw_rows)} filas** encontradas.")

            # Preview
            preview_df = pd.DataFrame(raw_rows).head(5)
            st.markdown("**Vista previa (primeras 5 filas):**")
            st.dataframe(preview_df, use_container_width=True)

            # Validate & normalize
            valid_entries, skipped = [], []
            existing_responses = load_responses()
            existing_ts = {r.get("timestamp") for r in existing_responses}

            for i, row in enumerate(raw_rows):
                # If it already looks like a full NutriMap response (has diet_key),
                # skip reclassification but still normalize
                if "diet_key" in row and "answers" in row:
                    ts = str(row.get("timestamp", ""))
                    if ts in existing_ts:
                        skipped.append(i)
                        continue
                    valid_entries.append(row)
                else:
                    entry = normalize_imported_row(row)
                    if entry is None:
                        skipped.append(i)
                        continue
                    ts = entry["timestamp"]
                    if ts in existing_ts:
                        skipped.append(i)
                        continue
                    valid_entries.append(entry)

            st.markdown(f"**{len(valid_entries)}** filas válidas · **{len(skipped)}** filas omitidas (faltan columnas o duplicadas)")

            if skipped:
                with st.expander(f"Ver {len(skipped)} filas omitidas"):
                    st.write(skipped)

            if valid_entries:
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    if st.button(f"⬆Importar {len(valid_entries)} respuestas", use_container_width=True):
                        merged = existing_responses + valid_entries
                        save_all_responses(merged)
                        st.success(f"{len(valid_entries)} respuestas importadas correctamente. Total en BD: {len(merged)}")
                        st.balloons()
                with col_b:
                    st.info("Las respuestas importadas serán clasificadas automáticamente y aparecerán en Dashboard y Topology.")
            else:
                st.warning("No hay filas válidas para importar.")

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")

    # ── Danger zone: clear DB ─────────────────────────────────────────────────
    with st.expander("Limpiar base de datos"):
        st.warning("Esto borrará **todas** las respuestas almacenadas. Esta acción no se puede deshacer.")
        confirm = st.text_input("Escribe BORRAR para confirmar:")
        if st.button("Borrar todas las respuestas"):
            if confirm.strip().upper() == "BORRAR":
                save_all_responses([])
                st.success("Base de datos limpiada.")
            else:
                st.error("Confirmación incorrecta.")


# =============================================================================
# PAGE: EXPORT DATA
# =============================================================================
elif "Export Data" in page:
    st.markdown('<h1 class="main-title">Export Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Descarga todas las respuestas recolectadas.</p>', unsafe_allow_html=True)
    st.markdown("---")

    responses = load_responses()

    if not responses:
        st.info("No hay respuestas aún.")
        st.stop()

    df = pd.DataFrame(responses)

    st.markdown(f"**{len(df)} respuestas totales disponibles.**")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇ Descargar CSV",
            csv,
            "nutrimap_responses.csv",
            "text/csv",
            use_container_width=True
        )
    with c2:
        json_str = json.dumps(responses, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇ Descargar JSON",
            json_str,
            "nutrimap_responses.json",
            "application/json",
            use_container_width=True
        )