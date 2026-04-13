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

# ── Corrección UTF-8 para Windows con kmapper ─────────────────────────────────
import pathlib
_orig_read_text = pathlib.Path.read_text
def _read_text_utf8(self, encoding=None, errors=None, **kw):
    return _orig_read_text(self, encoding=encoding or 'utf-8', errors=errors or 'replace', **kw)
pathlib.Path.read_text = _read_text_utf8

import kmapper as km  # noqa: E402

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriMap · Clasificador de dieta",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
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

# ── Persistencia de datos ──────────────────────────────────────────────────────
DATA_FILE = "survey_responses.json"

def cargar_respuestas():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def guardar_respuesta(entrada):
    respuestas = cargar_respuestas()
    respuestas.append(entrada)
    with open(DATA_FILE, "w") as f:
        json.dump(respuestas, f, indent=2, ensure_ascii=False)

def guardar_todas_respuestas(respuestas):
    with open(DATA_FILE, "w") as f:
        json.dump(respuestas, f, indent=2, ensure_ascii=False)

# ── Auxiliares de importación ─────────────────────────────────────────────────
REQUIRED_FEATURE_COLS = [
    "meat_freq", "veg_freq", "fruit_freq", "grain_freq",
    "fat_pref", "medical_cond", "env_concern", "sodium_concern",
    "protein_pref", "variety_pref"
]

def normalizar_fila_importada(fila: dict) -> dict | None:
    """
    Convierte un dict plano (de importación CSV/JSON) al formato interno de respuesta.
    Retorna None si le faltan columnas de características requeridas.
    """
    respuestas = {}
    for col in REQUIRED_FEATURE_COLS:
        if col not in fila:
            return None
        try:
            respuestas[col] = int(float(fila[col]))
        except (ValueError, TypeError):
            return None

    resultado = clasificar_dieta(respuestas)

    return {
        "timestamp": str(fila.get("timestamp", datetime.now().isoformat())),
        "name": str(fila.get("name", "Importado")),
        "age": int(float(fila.get("age", 30))),
        "gender": str(fila.get("gender", "Desconocido")),
        "answers": respuestas,
        "allergies": fila.get("allergies", []) if isinstance(fila.get("allergies"), list)
                     else [x.strip() for x in str(fila.get("allergies", "")).split(",") if x.strip()],
        "goals": fila.get("goals", []) if isinstance(fila.get("goals"), list)
                 else [x.strip() for x in str(fila.get("goals", "")).split(",") if x.strip()],
        "diet_key": resultado["diet_key"],
        "diet_name": resultado["diet_name"],
        "confidence": resultado["confidence"],
        "scores": resultado["scores"],
        "feature_vector": resultado["feature_vector"],
        "_imported": True,
    }

# ── Motor de clasificación de dietas ─────────────────────────────────────────
DIET_PROFILES = {
    "plant_based": {
        "name": "Basada en Plantas",
        "subtitle": "Vegetariana / Vegana",
        "emoji": "⚫",
        "badge_class": "badge-plant",
        "color": "#2e7d32",
        "bg": "#e8f5e9",
        "description": "Centrada en frutas, verduras, legumbres y cereales integrales. Elimina o minimiza los productos de origen animal.",
        "foods": ["Legumbres y frijoles", "Cereales integrales", "Nueces y semillas", "Verduras de hoja", "Frutas de temporada"],
        "avoid": ["Carne roja", "Aves de corral", "Lácteos (vegano)", "Alimentos procesados"],
        "health_focus": ["Salud cardiovascular", "Control de peso", "Sostenibilidad ambiental"],
        "profile": np.array([0, 5, 5, 5, 1, 0, 4, 3, 2, 2])
    },
    "mediterranean": {
        "name": "Mediterránea",
        "subtitle": "Saludable para el Corazón",
        "emoji": "⚫",
        "badge_class": "badge-med",
        "color": "#1565c0",
        "bg": "#e3f2fd",
        "description": "Rica en aceite de oliva, pescado, verduras y vino moderado. Inspirada en los patrones alimentarios del sur de Europa.",
        "foods": ["Aceite de oliva", "Pescado y mariscos", "Cereales integrales", "Legumbres", "Verduras frescas", "Vino tinto (moderado)"],
        "avoid": ["Carnes procesadas", "Azúcares refinados", "Alimentos ultraprocesados"],
        "health_focus": ["Salud del corazón", "Salud cerebral", "Longevidad", "Antiinflamatorio"],
        "profile": np.array([3, 4, 3, 4, 2, 1, 3, 3, 3, 3])
    },
    "low_carb": {
        "name": "Baja en Carbohidratos",
        "subtitle": "Cetogénica / Atkins",
        "emoji": "⚫",
        "badge_class": "badge-lowcarb",
        "color": "#e65100",
        "bg": "#fff3e0",
        "description": "Restringe los carbohidratos para inducir la cetosis. Alta en proteínas y grasas saludables.",
        "foods": ["Carnes y aves", "Pescado", "Huevos", "Queso", "Nueces", "Verduras bajas en carbohidratos"],
        "avoid": ["Pan y pasta", "Arroz", "Azúcar", "La mayoría de las frutas", "Verduras feculentas"],
        "health_focus": ["Pérdida de peso", "Control del azúcar en sangre", "Claridad mental", "Estabilidad energética"],
        "profile": np.array([5, 1, 1, 2, 4, 2, 2, 1, 4, 1])
    },
    "therapeutic": {
        "name": "Terapéutica",
        "subtitle": "Médica / Clínica",
        "emoji": "⚫",
        "badge_class": "badge-therapeutic",
        "color": "#6a1b9a",
        "bg": "#f3e5f5",
        "description": "Diseñada médicamente para condiciones específicas: bajo en sodio, modificada en textura, renal o para diabéticos.",
        "foods": ["Opciones bajas en sodio", "Alimentos blandos/en puré", "Porciones controladas", "Alimentos aprobados por médico"],
        "avoid": ["Alimentos altos en sodio", "Texturas duras (si es modificada en textura)", "Alimentos desencadenantes de la condición"],
        "health_focus": ["Manejo de enfermedades", "Interacción con medicamentos", "Nutrición clínica"],
        "profile": np.array([2, 2, 2, 3, 2, 5, 1, 4, 2, 3])
    }
}

def clasificar_dieta(respuestas: dict) -> dict:
    vec = np.array([
        respuestas.get("meat_freq", 0),
        respuestas.get("veg_freq", 0),
        respuestas.get("fruit_freq", 0),
        respuestas.get("grain_freq", 0),
        respuestas.get("fat_pref", 0),
        respuestas.get("medical_cond", 0),
        respuestas.get("env_concern", 0),
        respuestas.get("sodium_concern", 0),
        respuestas.get("protein_pref", 0),
        respuestas.get("variety_pref", 0),
    ], dtype=float)

    puntuaciones = {}
    for clave, dieta in DIET_PROFILES.items():
        perfil = dieta["profile"].astype(float)
        producto_punto = np.dot(vec, perfil)
        norma = np.linalg.norm(vec) * np.linalg.norm(perfil)
        puntuaciones[clave] = (producto_punto / norma) if norma > 0 else 0

    mejor = max(puntuaciones, key=puntuaciones.get)
    confianza = round(puntuaciones[mejor] * 100, 1)
    puntuaciones_pct = {k: round(v * 100, 1) for k, v in puntuaciones.items()}

    return {
        "diet_key": mejor,
        "diet_name": DIET_PROFILES[mejor]["name"],
        "confidence": confianza,
        "scores": puntuaciones_pct,
        "feature_vector": vec.tolist()
    }

# ── Etiquetado de clústeres según dieta ───────────────────────────────────────
# Cada clúster se etiqueta según qué características alimentarias dominan en él.
# Comparamos el centroide del clúster con los vectores de perfil de dieta.

CLUSTER_PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#8e44ad",
                   "#f39c12", "#16a085", "#d35400", "#2c3e50"]

def etiquetar_cluster_desde_centroide(centroide: np.ndarray) -> tuple[str, str]:
    """
    Dado un centroide de clúster en el espacio de características, retorna (diet_key, diet_name)
    mediante similitud coseno con los cuatro vectores de DIET_PROFILES.
    """
    mejor_clave = None
    mejor_sim = -1
    for clave, dieta in DIET_PROFILES.items():
        perfil = dieta["profile"].astype(float)
        norma = np.linalg.norm(centroide) * np.linalg.norm(perfil)
        sim = np.dot(centroide, perfil) / norma if norma > 0 else 0
        if sim > mejor_sim:
            mejor_sim = sim
            mejor_clave = clave
    return mejor_clave, DIET_PROFILES[mejor_clave]["name"]

def describir_cluster(centroide: np.ndarray) -> str:
    """
    Genera una descripción corta y legible del patrón alimentario dominante de un clúster.
    """
    etiquetas = ["Carne", "Veg", "Fruta", "Cereales", "Grasa", "Médico", "Ambiental", "Sodio", "Proteína", "Variedad"]
    idx_top = np.argsort(centroide)[::-1][:3]
    características_top = [etiquetas[i] for i in idx_top if centroide[i] >= 2.5]
    if características_top:
        return "Alto: " + ", ".join(características_top)
    return "Equilibrado / ingesta baja"

# ── Análisis topológico (basado en clústeres) ─────────────────────────────────
def ejecutar_analisis_topologico(df_características: pd.DataFrame, n_clusters: int = 4):
    """
    Ejecuta KMapper + PCA.
    Retorna (X_pca, grafo, pca, kmeans_modelo, etiquetas_cluster_por_nodo)
    donde etiquetas_cluster_por_nodo mapea node_id → (diet_key, diet_name, descripción).
    """
    if len(df_características) < 5:
        return None, None, None, None, {}

    escalador = StandardScaler()
    X_escalado = escalador.fit_transform(df_características)

    # Lente PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_escalado)

    # KMeans global para que los centroides estén en la escala *original* de las características
    # (no escalada) — más interpretable para etiquetas de alimentos
    kmeans_global = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_global.fit(df_características.values)

    # KMapper — usa KMeans internamente por parche de cobertura
    mapper = km.KeplerMapper(verbose=0)
    lente = mapper.fit_transform(X_pca, projection=[0])

    grafo = mapper.map(
        lente,
        X_escalado,
        clusterer=KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        cover=km.Cover(n_cubes=8, perc_overlap=0.4)
    )

    # Para cada nodo se calcula el centroide de sus vectores miembro (escala original)
    # y se asigna la etiqueta de dieta por similitud coseno
    etiquetas_cluster_por_nodo = {}
    for node_id, índices_miembros in grafo.get("nodes", {}).items():
        válidos = [i for i in índices_miembros if i < len(df_características)]
        if válidos:
            centroide = df_características.values[válidos].mean(axis=0)
            diet_key, diet_name = etiquetar_cluster_desde_centroide(centroide)
            desc = describir_cluster(centroide)
            etiquetas_cluster_por_nodo[node_id] = {
                "diet_key": diet_key,
                "diet_name": diet_name,
                "description": desc,
                "centroid": centroide,
                "n_members": len(válidos),
            }

    return X_pca, grafo, pca, kmeans_global, etiquetas_cluster_por_nodo

# -- Gestión de datos con Session State ----------------------------------------
def obtener_datos_activos():
    # Priorizamos el archivo subido por el usuario
    if uploaded_file is not None:
        try:
            import_df = pd.read_csv(uploaded_file)
            filas_raw = import_df.to_dict(orient="records")
            entradas_válidas = []
            for fila in filas_raw:
                entrada = normalizar_fila_importada(fila)
                if entrada:
                    entradas_válidas.append(entrada)
            return entradas_válidas
        except Exception as e:
            st.sidebar.error(f"Error al leer CSV: {e}")
            return []
    # Si no hay carga, retornamos la base de datos local
    return cargar_respuestas()

# Inicializar estados de sesión para evitar cálculos redundantes
if 'topology_results' not in st.session_state:
    st.session_state.topology_results = None
if 'last_data_hash' not in st.session_state:
    st.session_state.last_data_hash = None


# ── Navegación en barra lateral ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🌿 NutriMap</div>', unsafe_allow_html=True)
    
    st.markdown("### Carga de Datos")
    uploaded_file = st.file_uploader("Sube tu CSV de NutriMap", type=["csv"], help="Exporta desde tu Excel y súbelo aquí.")
    
    st.markdown("---")
    page = st.radio(
        "Navegación",
        ["⚪ Panel de Control", "⚪ Topología", "⚪ Encuesta", "📥 Importar Datos", "📁 Exportar Datos"],
        label_visibility="collapsed"
    )
    
    responses = obtener_datos_activos()
    # Hash simple para detectar cambios en los datos
    current_data_hash = hash(str(responses))
    
    st.markdown("---")
    st.markdown(f"**Registros activos:** {len(responses)}")

# =============================================================================
# PÁGINA: ENCUESTA
# =============================================================================
if "⚪ Encuesta" in page:
    col_title, _ = st.columns([3, 1])
    with col_title:
        st.markdown('<h1 class="main-title">Encuesta de Preferencias Alimentarias</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Responde con honestidad — encontraremos tu patrón dietético ideal.</p>', unsafe_allow_html=True)

    st.markdown("---")

    with st.form("survey_form"):
        st.markdown('<p class="section-header">Acerca de ti</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Nombre (opcional)", placeholder="Anónimo")
        with c2:
            age = st.number_input("Edad", min_value=10, max_value=100, value=30)
        with c3:
            gender = st.selectbox("Género", ["Prefiero no decir", "Femenino", "Masculino", "No binario", "Otro"])

        st.markdown('<p class="section-header">Hábitos alimentarios</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            meat_freq = st.select_slider(
                "¿Con qué frecuencia comes carne (res, cerdo, pollo)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Nunca", "Raramente", "1-2x/semana", "3-4x/semana", "Diario", "Varias veces/día"][x],
                value=2
            )
            veg_freq = st.select_slider(
                "¿Con qué frecuencia comes verduras?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Nunca", "Raramente", "A veces", "Frecuentemente", "Diario", "En cada comida"][x],
                value=3
            )
            fruit_freq = st.select_slider(
                "¿Con qué frecuencia comes fruta fresca?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Nunca", "Raramente", "A veces", "Frecuentemente", "Diario", "Varias veces/día"][x],
                value=3
            )
            grain_freq = st.select_slider(
                "¿Cuánto consumes cereales, legumbres y frijoles?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Ninguno", "Muy poco", "Poco", "Moderado", "Alto", "Muy alto"][x],
                value=3
            )

        with c2:
            fat_pref = st.select_slider(
                "Preferencia por grasas (mantequilla, aceites, carnes grasas)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Evito todas las grasas", "Muy bajo en grasas", "Bajo en grasas", "Moderado", "Alto en grasas", "Nivel cetogénico"][x],
                value=2
            )
            protein_pref = st.select_slider(
                "¿Qué tan importante es la proteína en tu dieta?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["No importante", "Prioridad baja", "Por debajo del promedio", "Promedio", "Prioridad alta", "Enfoque principal"][x],
                value=3
            )
            variety_pref = st.select_slider(
                "¿Cuánta variedad dietética prefieres?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Muy rígida", "Poca variedad", "Algo de variedad", "Moderada", "Alta variedad", "Máxima variedad"][x],
                value=3
            )
            env_concern = st.select_slider(
                "¿Qué tan importante es el impacto ambiental en tus elecciones alimentarias?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Para nada", "Ligeramente", "Algo", "Moderadamente", "Muy importante", "Factor principal"][x],
                value=2
            )

        st.markdown('<p class="section-header">Salud y condiciones médicas</p>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            sodium_concern = st.select_slider(
                "¿Necesitas limitar el consumo de sodio / sal?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Sin preocupación", "Leve cuidado", "Algo de cuidado", "Cuidado moderado", "Alta preocupación", "Restricción médica"][x],
                value=1
            )
            medical_cond = st.select_slider(
                "¿Tienes condiciones médicas dietéticas (diabetes, renal, disfagia, etc.)?",
                options=[0, 1, 2, 3, 4, 5],
                format_func=lambda x: ["Ninguna", "Menor", "Condición leve", "Moderada", "Significativa", "Grave/compleja"][x],
                value=0
            )
        with c2:
            allergies = st.multiselect(
                "Alergias o intolerancias",
                ["Gluten", "Lactosa/Lácteos", "Nueces", "Huevos", "Soya", "Mariscos", "Ninguna"],
                default=["Ninguna"]
            )
            goals = st.multiselect(
                "Objetivos principales de salud",
                ["Pérdida de peso", "Ganancia muscular", "Salud del corazón", "Control del azúcar en sangre",
                 "Antiinflamatorio", "Salud digestiva", "Niveles de energía", "Bienestar general"],
                default=["Bienestar general"]
            )

        st.markdown("---")
        submitted = st.form_submit_button("Clasificar mi dieta", use_container_width=True)

    if submitted:
        answers = {
            "meat_freq": meat_freq, "veg_freq": veg_freq,
            "fruit_freq": fruit_freq, "grain_freq": grain_freq,
            "fat_pref": fat_pref, "medical_cond": medical_cond,
            "env_concern": env_concern, "sodium_concern": sodium_concern,
            "protein_pref": protein_pref, "variety_pref": variety_pref
        }

        resultado = clasificar_dieta(answers)
        diet_info = DIET_PROFILES[resultado["diet_key"]]

        entrada = {
            "timestamp": datetime.now().isoformat(),
            "name": name or "Anónimo",
            "age": int(age),
            "gender": gender,
            "answers": answers,
            "allergies": allergies,
            "goals": goals,
            "diet_key": resultado["diet_key"],
            "diet_name": resultado["diet_name"],
            "confidence": resultado["confidence"],
            "scores": resultado["scores"],
            "feature_vector": resultado["feature_vector"]
        }
        guardar_respuesta(entrada)

        st.success("¡Respuesta guardada! Aquí está tu resultado:")
        st.markdown("---")

        st.markdown(f"""
        <div class="result-hero">
            <div style="font-size:3rem; margin-bottom:0.5rem;">{diet_info['emoji']}</div>
            <h2>{diet_info['name']}</h2>
            <p style="font-size:1.1rem;">{diet_info['subtitle']}</p>
            <p style="font-size:0.9rem; margin-top:1rem;">{diet_info['description']}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Similitud con cada perfil de dieta")
        cols = st.columns(4)
        for i, (clave, dieta) in enumerate(DIET_PROFILES.items()):
            with cols[i]:
                puntuacion = resultado["scores"][clave]
                es_mejor = clave == resultado["diet_key"]
                st.markdown(f"""
                <div class="metric-pill" style="border: {'2px solid ' + dieta['color'] if es_mejor else '1px solid #e0dbf7'}">
                    <div style="font-size:1.5rem;">{dieta['emoji']}</div>
                    <div class="val" style="color:{dieta['color']}">{puntuacion}%</div>
                    <div class="lbl">{dieta['name']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Alimentos recomendados**")
            for f in diet_info["foods"]:
                st.markdown(f"- {f}")
        with c2:
            st.markdown("**Limitar o evitar**")
            for f in diet_info["avoid"]:
                st.markdown(f"- {f}")
        with c3:
            st.markdown("**Enfoque de salud**")
            for f in diet_info["health_focus"]:
                st.markdown(f"- {f}")

# =============================================================================
# PÁGINA: PANEL DE CONTROL
# =============================================================================
elif "Panel de Control" in page:
    st.markdown('<h1 class="main-title">Panel de Estadísticas</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Resumen agregado de todas las respuestas de la encuesta.</p>', unsafe_allow_html=True)
    st.markdown("---")

    responses = cargar_respuestas()

    if len(responses) == 0:
        st.info("Aún no hay respuestas. Completa la encuesta o importa datos para comenzar.")
        st.stop()

    # 1. Definimos las columnas que usaremos (usando la lista global)
    feature_cols = REQUIRED_FEATURE_COLS

    # 2. Extraemos los nombres de las dietas para las etiquetas del gráfico
    diet_names = [r["diet_name"] for r in responses]

    # 3. Preparamos el DataFrame de respuestas
    answers_df = pd.json_normalize([r["answers"] for r in responses])

    df = pd.DataFrame(responses)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total de Respuestas", len(df))
    with c2:
        más_común = df["diet_key"].value_counts().idxmax()
        st.metric("Dieta Más Común", DIET_PROFILES[más_común]["name"])
    with c3:
        conf_promedio = round(df["confidence"].mean(), 1) if "confidence" in df.columns else "—"
        st.metric("Confianza Promedio", f"{conf_promedio}%")
    with c4:
        hoy = sum(1 for r in responses if r["timestamp"][:10] == datetime.now().date().isoformat())
        st.metric("Respuestas Hoy", hoy)

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Distribución de dietas")
        conteo_dietas = df["diet_key"].value_counts().reset_index()
        conteo_dietas.columns = ["diet", "count"]
        conteo_dietas["name"] = conteo_dietas["diet"].map(lambda k: DIET_PROFILES[k]["name"] + " " + DIET_PROFILES[k]["emoji"])
        colores = [DIET_PROFILES[k]["color"] for k in conteo_dietas["diet"]]
        fig = px.pie(conteo_dietas, values="count", names="name",
                     color_discrete_sequence=colores, hole=0.45)
        fig.update_layout(margin=dict(t=10, b=10), height=300,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Distribución de edad por dieta")
        if "age" in df.columns:
            fig2 = px.box(df, x="diet_name", y="age",
                          color="diet_name",
                          color_discrete_map={DIET_PROFILES[k]["name"]: DIET_PROFILES[k]["color"] for k in DIET_PROFILES})
            fig2.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                                xaxis_title="", yaxis_title="Edad")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Valores promedio de características por dieta")
    feature_cols = ["meat_freq", "veg_freq", "fruit_freq", "grain_freq",
                    "fat_pref", "medical_cond", "env_concern", "sodium_concern",
                    "protein_pref", "variety_pref"]
    feature_labels = ["Carne", "Verduras", "Fruta", "Cereales/Legumbres",
                      "Preferencia de Grasa", "Condiciones Médicas", "Preocupación Ambiental",
                      "Preocupación por Sodio", "Pref. de Proteína", "Pref. de Variedad"]

    answers_df = pd.json_normalize(df["answers"])
    answers_df["diet_name"] = df["diet_name"].values

    datos_mapa_calor = answers_df.groupby("diet_name")[feature_cols].mean()
    datos_mapa_calor.columns = feature_labels

    fig3 = px.imshow(datos_mapa_calor.T, color_continuous_scale="RdYlGn",
                     aspect="auto", text_auto=".1f")
    fig3.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig3, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Objetivos de salud más frecuentes")
        todos_objetivos = [g for r in responses for g in r.get("goals", [])]
        conteo_objetivos = pd.Series(todos_objetivos).value_counts().head(8)
        fig4 = px.bar(x=conteo_objetivos.values, y=conteo_objetivos.index, orientation="h",
                      color_discrete_sequence=["#667eea"])
        fig4.update_layout(height=280, margin=dict(t=10, b=10),
                            xaxis_title="Cantidad", yaxis_title="")
        st.plotly_chart(fig4, use_container_width=True)

    with c2:
        st.markdown("#### Frecuencia de alergias / intolerancias")
        todas_alergias = [a for r in responses for a in r.get("allergies", []) if a != "Ninguna"]
        if todas_alergias:
            conteo_alergias = pd.Series(todas_alergias).value_counts()
            fig5 = px.bar(x=conteo_alergias.index, y=conteo_alergias.values,
                          color_discrete_sequence=["#764ba2"])
            fig5.update_layout(height=280, margin=dict(t=10, b=10),
                                xaxis_title="", yaxis_title="Cantidad")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Aún no se han reportado alergias.")

    st.markdown("#### Respuestas recientes")
    display_cols = ["timestamp", "name", "age", "gender", "diet_name", "confidence"]
    disponibles = [c for c in display_cols if c in df.columns]
    st.dataframe(
        df[disponibles].tail(20).rename(columns={
            "timestamp": "Fecha", "name": "Nombre", "age": "Edad",
            "gender": "Género", "diet_name": "Dieta", "confidence": "Coincidencia %"
        }).iloc[::-1],
        use_container_width=True
    )


# =============================================================================
# PÁGINA: TOPOLOGÍA (mapper basado en clústeres)
# =============================================================================
elif "⚪ Topología" in page:
    st.markdown('<h1 class="main-title">Análisis Topológico</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">KMapper + PCA · cada nodo coloreado por el <em>clúster de patrones alimentarios</em> que predomina en él.</p>', unsafe_allow_html=True)
    st.markdown("---")

    responses = obtener_datos_activos()

    if len(responses) < 5:
        st.warning(f"Se necesitan al menos 5 respuestas para el análisis topológico. Actualmente: {len(responses)}")
        st.stop()

    feature_cols = REQUIRED_FEATURE_COLS

    answers_df = pd.json_normalize([r["answers"] for r in responses])

    diet_names = [r["diet_name"] for r in responses]

    # --- Parámetros de la Topología ---
    with st.expander("Parámetros de topología", expanded=False):
        n_clusters = st.slider("Número de clústeres de patrones alimentarios (KMeans)", 2, 8, 4)
        n_cubes = st.slider("Cubos de cobertura KMapper", 4, 20, 8)
        perc_overlap = st.slider("Solapamiento de cobertura %", 0.1, 0.8, 0.4, step=0.05)

    # --- Lógica de caché (Session State) ---
    current_data_hash = hash(str(responses))
    params_key = f"{current_data_hash}_{n_clusters}_{n_cubes}_{perc_overlap}"

    if st.session_state.topology_results is None or st.session_state.last_data_hash != params_key:
        with st.spinner("Ejecutando algoritmos..."):
            resultados = ejecutar_analisis_topologico(
                answers_df[feature_cols],
                n_clusters=n_clusters
            )
            st.session_state.topology_results = resultados
            st.session_state.last_data_hash = params_key

    X_pca, grafo, pca, kmeans_model, etiquetas_cluster_por_nodo = st.session_state.topology_results

    if X_pca is None:
        st.error("No se pudo ejecutar el análisis. Verifica los datos.")
        st.stop()

    # ── Radar de centroides de clúster (qué come cada clúster) ───────────────
    st.markdown("#### Clústeres de patrones alimentarios · qué come cada clúster")
    centroides = kmeans_model.cluster_centers_
    feature_labels_short = ["Carne", "Veg", "Fruta", "Cereales", "Grasa", "Médico", "Ambiental", "Sodio", "Proteína", "Variedad"]

    centroid_df = pd.DataFrame(centroides, columns=feature_labels_short)
    centroid_df.index = [f"Clúster {i}" for i in range(len(centroides))]

    # Asignar etiqueta de dieta a cada centroide de clúster
    etiquetas_dieta_cluster = {}
    for i, fila in centroid_df.iterrows():
        diet_key, diet_name = etiquetar_cluster_desde_centroide(fila.values)
        etiquetas_dieta_cluster[i] = {"diet_key": diet_key, "diet_name": diet_name,
                                       "emoji": DIET_PROFILES[diet_key]["emoji"],
                                       "color": DIET_PROFILES[diet_key]["color"]}

    # Mapa de calor de centroides
    fig_cent = px.imshow(
        centroid_df.T,
        color_continuous_scale="YlOrRd",
        text_auto=".1f",
        aspect="auto",
        labels=dict(color="Puntuación prom.")
    )
    anotaciones_cluster = [
        f"{etiquetas_dieta_cluster[f'Clúster {i}']['emoji']} {etiquetas_dieta_cluster[f'Clúster {i}']['diet_name']}"
        for i in range(len(centroides))
    ]
    fig_cent.update_layout(
        height=360,
        margin=dict(t=40, b=20),
        xaxis=dict(
            tickvals=list(range(len(centroides))),
            ticktext=anotaciones_cluster
        )
    )
    st.plotly_chart(fig_cent, use_container_width=True)
    st.caption("Cada columna es un clúster de patrones alimentarios. El emoji/etiqueta indica el tipo de dieta más cercano al centroide del clúster.")

    # ── Dispersión PCA — coloreada por clúster ────────────────────────────────
    st.markdown("#### PCA · Espacio de preferencias alimentarias (color = clúster)")
    col_info, col_plot = st.columns([1, 3])

    # Asignar cada respondente a un clúster KMeans
    ids_cluster_global = kmeans_model.predict(answers_df[feature_cols].values)
    nombres_cluster_dieta = [
        f"{etiquetas_dieta_cluster[f'Clúster {c}']['emoji']} {etiquetas_dieta_cluster[f'Clúster {c}']['diet_name']} (C{c})"
        for c in ids_cluster_global
    ]
    mapa_colores_cluster = {
        f"{etiquetas_dieta_cluster[f'Clúster {c}']['emoji']} {etiquetas_dieta_cluster[f'Clúster {c}']['diet_name']} (C{c})":
        etiquetas_dieta_cluster[f'Clúster {c}']['color']
        for c in range(n_clusters)
    }

    with col_info:
        var = pca.explained_variance_ratio_
        st.markdown(f"**PC1:** {round(var[0]*100, 1)}% de varianza")
        st.markdown(f"**PC2:** {round(var[1]*100, 1)}% de varianza")
        st.markdown(f"**Total:** {round(sum(var)*100, 1)}%")
        st.markdown("---")
        st.caption("Cada punto = un respondente. Color = clúster alimentario asignado → tipo de dieta.")

    with col_plot:
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["Clúster"] = nombres_cluster_dieta
        pca_df["Dieta (asignada)"] = diet_names

        fig_pca = px.scatter(
            pca_df, x="PC1", y="PC2",
            color="Clúster",
            color_discrete_map=mapa_colores_cluster,
            hover_data={"Dieta (asignada)": True, "Clúster": True},
            labels={"PC1": "Componente Principal 1", "PC2": "Componente Principal 2"}
        )
        fig_pca.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color="white")))
        fig_pca.update_layout(height=400, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pca, use_container_width=True)

    # ── Grafo KMapper — coloreado por clúster de dieta ────────────────────────
    st.markdown("#### KMapper · Grafo topológico (nodos coloreados por clúster alimentario)")

    if grafo and grafo.get("nodes"):
        nodos = list(grafo["nodes"].keys())
        aristas = []
        for src, destinos in grafo.get("links", {}).items():
            for dst in destinos:
                aristas.append((src, dst))

        n = len(nodos)
        angulos = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pos = {nodo: (np.cos(a), np.sin(a)) for nodo, a in zip(nodos, angulos)}

        colores_nodo, tamaños_nodo, hover_nodo = [], [], []

        for nodo in nodos:
            info = etiquetas_cluster_por_nodo.get(nodo, {})
            diet_key = info.get("diet_key", "plant_based")
            n_miembros = info.get("n_members", 1)
            desc = info.get("description", "")
            diet_name = info.get("diet_name", "Desconocido")
            emoji = DIET_PROFILES[diet_key]["emoji"]
            color = DIET_PROFILES[diet_key]["color"]

            colores_nodo.append(color)
            tamaños_nodo.append(max(12, n_miembros * 6))
            hover_nodo.append(
                f"{emoji} {diet_name}<br>n={n_miembros}<br>{desc}"
            )

        arista_x, arista_y = [], []
        for src, dst in aristas:
            x0, y0 = pos[src]
            x1, y1 = pos[dst]
            arista_x += [x0, x1, None]
            arista_y += [y0, y1, None]

        nodo_x = [pos[nd][0] for nd in nodos]
        nodo_y = [pos[nd][1] for nd in nodos]

        fig_mapper = go.Figure()
        fig_mapper.add_trace(go.Scatter(
            x=arista_x, y=arista_y, mode="lines",
            line=dict(color="#ddd", width=1.2), hoverinfo="none"
        ))
        fig_mapper.add_trace(go.Scatter(
            x=nodo_x, y=nodo_y, mode="markers",
            marker=dict(size=tamaños_nodo, color=colores_nodo,
                        line=dict(color="white", width=2)),
            text=hover_nodo,
            hovertemplate="%{text}<extra></extra>"
        ))

        # Leyenda de dietas
        vistos = set()
        for nodo in nodos:
            info = etiquetas_cluster_por_nodo.get(nodo, {})
            dk = info.get("diet_key", "plant_based")
            if dk not in vistos:
                vistos.add(dk)
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
            f"**{len(nodos)} nodos · {len(aristas)} aristas.** "
            "Tamaño del nodo = número de respondentes. "
            "Color = tipo de dieta inferido del centroide alimentario del clúster que domina el nodo. "
            "Pasa el cursor sobre un nodo para ver detalles."
        )
    else:
        st.info("El grafo KMapper está vacío — intenta recolectar respuestas más diversas o ajusta los parámetros.")

    # ── Mapa de calor de cargas PCA ────────────────────────────────────────────
    st.markdown("#### Cargas PCA · qué características separan los datos")
    loading_df = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=["Carne", "Verduras", "Fruta", "Cereales/Legumbres", "Pref. de Grasa",
               "Cond. Médica", "Preoc. Ambiental", "Preoc. por Sodio", "Pref. de Proteína", "Pref. de Variedad"]
    )
    fig_load = px.imshow(loading_df, color_continuous_scale="RdBu_r",
                         text_auto=".2f", aspect="auto",
                         color_continuous_midpoint=0)
    fig_load.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_load, use_container_width=True)


# =============================================================================
# PÁGINA: IMPORTAR DATOS
# =============================================================================
elif "Importar Datos" in page:
    st.markdown('<h1 class="main-title">Importar Datos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Carga respuestas previas desde CSV o JSON para alimentar la base de datos.</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Referencia de formato ──────────────────────────────────────────────────
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

Para CSV: `allergies` y `goals` pueden ser strings separados por comas, p.ej. `"Gluten,Nueces"`.
""")
        # Descarga de CSV de ejemplo
        sample_data = {
            "name": ["Alice", "Bob"],
            "age": [28, 45],
            "gender": ["Femenino", "Masculino"],
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
            "allergies": ["Gluten", "Ninguna"],
            "goals": ["Salud del corazón,Bienestar general", "Pérdida de peso"],
        }
        st.download_button(
            "Descargar CSV de ejemplo",
            pd.DataFrame(sample_data).to_csv(index=False),
            "nutrimap_ejemplo.csv",
            "text/csv"
        )

    st.markdown("---")

    # ── Widget de carga ──────────────────────────────────────────────────────────
    st.markdown('<div class="import-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Sube tu archivo CSV o JSON",
        type=["csv", "json"],
        help="CSV con columnas separadas por coma, o JSON exportado por NutriMap (array de objetos)."
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        extensión = uploaded_file.name.rsplit(".", 1)[-1].lower()

        try:
            if extensión == "csv":
                import_df = pd.read_csv(uploaded_file)
                filas_raw = import_df.to_dict(orient="records")
            else:  # json
                datos_raw = json.load(uploaded_file)
                if isinstance(datos_raw, list):
                    filas_raw = datos_raw
                else:
                    st.error("El JSON debe ser un array de objetos.")
                    st.stop()

            st.success(f"Archivo cargado: **{len(filas_raw)} filas** encontradas.")

            # Vista previa
            preview_df = pd.DataFrame(filas_raw).head(5)
            st.markdown("**Vista previa (primeras 5 filas):**")
            st.dataframe(preview_df, use_container_width=True)

            # Validar y normalizar
            entradas_válidas, omitidas = [], []
            respuestas_existentes = cargar_respuestas()
            timestamps_existentes = {r.get("timestamp") for r in respuestas_existentes}

            for i, fila in enumerate(filas_raw):
                # Si ya parece una respuesta completa de NutriMap (tiene diet_key),
                # omitir la reclasificación pero normalizar de todas formas
                if "diet_key" in fila and "answers" in fila:
                    ts = str(fila.get("timestamp", ""))
                    if ts in timestamps_existentes:
                        omitidas.append(i)
                        continue
                    entradas_válidas.append(fila)
                else:
                    entrada = normalizar_fila_importada(fila)
                    if entrada is None:
                        omitidas.append(i)
                        continue
                    ts = entrada["timestamp"]
                    if ts in timestamps_existentes:
                        omitidas.append(i)
                        continue
                    entradas_válidas.append(entrada)

            st.markdown(f"**{len(entradas_válidas)}** filas válidas · **{len(omitidas)}** filas omitidas (faltan columnas o duplicadas)")

            if omitidas:
                with st.expander(f"Ver {len(omitidas)} filas omitidas"):
                    st.write(omitidas)

            if entradas_válidas:
                col_a, col_b = st.columns([1, 3])
                with col_a:
                    if st.button(f"⬆ Importar {len(entradas_válidas)} respuestas", use_container_width=True):
                        combinadas = respuestas_existentes + entradas_válidas
                        guardar_todas_respuestas(combinadas)
                        st.success(f"{len(entradas_válidas)} respuestas importadas correctamente. Total en BD: {len(combinadas)}")
                        st.balloons()
                with col_b:
                    st.info("Las respuestas importadas serán clasificadas automáticamente y aparecerán en el Panel de Control y en Topología.")
            else:
                st.warning("No hay filas válidas para importar.")

        except Exception as e:
            st.error(f"Error al procesar el archivo: {e}")

    st.markdown("---")

    # ── Zona peligrosa: limpiar BD ─────────────────────────────────────────────
    with st.expander("Limpiar base de datos"):
        st.warning("Esto borrará **todas** las respuestas almacenadas. Esta acción no se puede deshacer.")
        confirmar = st.text_input("Escribe BORRAR para confirmar:")
        if st.button("Borrar todas las respuestas"):
            if confirmar.strip().upper() == "BORRAR":
                guardar_todas_respuestas([])
                st.success("Base de datos limpiada.")
            else:
                st.error("Confirmación incorrecta.")


# =============================================================================
# PÁGINA: EXPORTAR DATOS
# =============================================================================
elif "Exportar Datos" in page:
    st.markdown('<h1 class="main-title">Exportar Datos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Descarga todas las respuestas recolectadas.</p>', unsafe_allow_html=True)
    st.markdown("---")

    responses = cargar_respuestas()

    if not responses:
        st.info("Aún no hay respuestas.")
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
            "nutrimap_respuestas.csv",
            "text/csv",
            use_container_width=True
        )
    with c2:
        json_str = json.dumps(responses, indent=2, ensure_ascii=False)
        st.download_button(
            "⬇ Descargar JSON",
            json_str,
            "nutrimap_respuestas.json",
            "application/json",
            use_container_width=True
        )
