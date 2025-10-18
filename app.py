# =====================================================
# 🎓 APP: Predicción de Empleabilidad Post-Graduación
# =====================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cloudpickle
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# ==================================
# 🧹 Clase CleanAndScale (para deserializar modelos)
# ==================================
class CleanAndScale(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=None):
        self.numeric_cols = numeric_cols or []
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.Q1 = X[self.numeric_cols].quantile(0.25)
        self.Q3 = X[self.numeric_cols].quantile(0.75)
        self.IQR = self.Q3 - self.Q1
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        mask = ~(
            (X[self.numeric_cols] < (self.Q1 - 1.5 * self.IQR)) |
            (X[self.numeric_cols] > (self.Q3 + 1.5 * self.IQR))
        ).any(axis=1)
        X_clean = X.copy()
        X_clean[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X_clean

# ==========================
# ⚙️ CONFIGURACIÓN INICIAL
# ==========================
st.set_page_config(page_title="Predicción de Empleabilidad", page_icon="🎓", layout="wide")
st.title("🎓 Predicción de Empleabilidad Post-Graduación")

st.markdown("""
Este panel interactivo compara **tres modelos supervisados** — *Random Forest, Decision Tree y Logistic Regression* —  
entrenados sobre factores académicos y personales que influyen en la **empleabilidad de un estudiante**.
""")

st.caption("Desarrollado por **Samantha Quintanchala** — Proyecto Final de Machine Learning")
st.info(f"🧠 Versión de Python: `{sys.version.split()[0]}`")

# ==========================
# 🔹 CARGA DE CLEANER Y MODELOS
# ==========================
st.sidebar.header("⚙️ Cargar Modelos")

try:
    with open("cleaner.pkl", "rb") as f:
        cleaner = cloudpickle.load(f)
    with open("modelo_Random_Forest.pkl", "rb") as f:
        model_rf = cloudpickle.load(f)
    with open("modelo_Decision_Tree.pkl", "rb") as f:
        model_dt = cloudpickle.load(f)
    with open("modelo_Logistic_Regression.pkl", "rb") as f:
        model_lr = cloudpickle.load(f)
    st.sidebar.success("✅ Modelos cargados correctamente.")
except Exception as e:
    st.sidebar.error(f"⚠️ Error al cargar modelos: {e}")
    st.stop()

# ==========================
# 🧮 ENTRADAS DEL USUARIO
# ==========================
st.sidebar.header("🧩 Factores de entrada")

IQ = st.sidebar.slider("IQ", 80, 160, 110)
Prev_Sem_Result = st.sidebar.slider("Promedio del semestre previo", 0.0, 10.0, 7.5)
CGPA = st.sidebar.slider("CGPA final", 0.0, 10.0, 7.0)
Academic_Performance = st.sidebar.slider("Desempeño académico (1-10)", 1, 10, 8)
Internship_Experience = st.sidebar.selectbox("¿Tuvo experiencia en prácticas?", ["No", "Sí"])
Extra_Curricular_Score = st.sidebar.slider("Actividades extracurriculares (1-10)", 1, 10, 5)
Communication_Skills = st.sidebar.slider("Habilidades de comunicación (1-10)", 1, 10, 7)
Projects_Completed = st.sidebar.slider("Proyectos completados", 0, 10, 3)

input_data = pd.DataFrame({
    "IQ": [IQ],
    "Prev_Sem_Result": [Prev_Sem_Result],
    "CGPA": [CGPA],
    "Academic_Performance": [Academic_Performance],
    "Internship_Experience": [1 if Internship_Experience == "Sí" else 0],
    "Extra_Curricular_Score": [Extra_Curricular_Score],
    "Communication_Skills": [Communication_Skills],
    "Projects_Completed": [Projects_Completed]
})

st.write("📘 **Datos ingresados:**")
st.dataframe(input_data, use_container_width=True)

# ==========================
# 🧼 LIMPIEZA Y ESCALADO
# ==========================
try:
    X_clean = cleaner.transform(input_data)
except Exception as e:
    st.error(f"Error al aplicar el limpiador: {e}")
    st.stop()

# ==========================
# 🔮 PREDICCIONES DE LOS MODELOS
# ==========================
pred_rf = model_rf.predict_proba(X_clean)[0][1]
pred_dt = model_dt.predict_proba(X_clean)[0]()
