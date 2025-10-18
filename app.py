# ==========================================
# 🎓 APP DE PREDICCIÓN DE EMPLEABILIDAD
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys
import cloudpickle

# ==========================
# ⚙️ CONFIGURACIÓN INICIAL
# ==========================
st.set_page_config(page_title="Predicción de Empleabilidad", page_icon="🎓", layout="wide")

st.title("🎓 Predicción de Empleabilidad Post-Graduación")
st.markdown("""
Este panel interactivo compara **tres modelos supervisados** — *Random Forest, Decision Tree* y *Logistic Regression* —
entrenados sobre factores académicos y personales que influyen en la **empleabilidad de un estudiante**.
""")
st.caption("Desarrollado por **Samantha Quintanchala** — Proyecto Final de Machine Learning")
st.info(f"🧠 Versión de Python: `{sys.version.split()[0]}`")

# ==========================
# ⚡ CARGA DE MODELOS CON CACHE
# ==========================
@st.cache_resource
def load_models():
    try:
        with open("cleaner.pkl", "rb") as f:
            cleaner = cloudpickle.load(f)
        with open("modelo_Random_Forest.pkl", "rb") as f:
            model_rf = cloudpickle.load(f)
        with open("modelo_Decision_Tree.pkl", "rb") as f:
            model_dt = cloudpickle.load(f)
        with open("modelo_Logistic_Regression.pkl", "rb") as f:
            model_lr = cloudpickle.load(f)
        return cleaner, model_rf, model_dt, model_lr
    except ModuleNotFoundError as e:
        st.sidebar.error(f"⚠️ Error al cargar modelos: {e}")
        st.stop()
    except Exception as e:
        st.sidebar.error(f"❌ No se pudieron cargar los modelos.\n\n**Detalles:** {e}")
        st.stop()

cleaner, model_rf, model_dt, model_lr = load_models()
st.sidebar.success("✅ Modelos cargados correctamente.")

# ==========================
# 🧩 ENTRADAS DEL USUARIO
# ==========================
st.sidebar.header("🧮 Factores de entrada")

IQ = st.sidebar.slider("IQ", 80, 160, 110)
Prev_Sem_Result = st.sidebar.slider("Promedio del semestre previo", 0.0, 10.0, 7.5)
CGPA = st.sidebar.slider("CGPA final", 0.0, 10.0, 7.0)
Academic_Performance = st.sidebar.slider("Desempeño académico (1–10)", 1, 10, 8)
Internship_Experience = st.sidebar.selectbox("¿Tuvo experiencia en prácticas?", ["No", "Sí"])
Extra_Curricular_Score = st.sidebar.slider("Actividades extracurriculares (1–10)", 1, 10, 5)
Communication_Skills = st.sidebar.slider("Habilidades de comunicación (1–10)", 1, 10, 7)
Projects_Completed = st.sidebar.slider("Proyectos completados", 0, 10, 3)

# ==========================
# 📊 CREAR DATAFRAME DE ENTRADA
# ==========================
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

st.markdown("### 📘 Datos ingresados por el usuario")
st.dataframe(input_data, width="stretch")

# ==========================
# 🧼 LIMPIEZA Y ESCALADO
# ==========================
try:
    X_new_clean = cleaner.transform(input_data)
except Exception as e:
    st.error(f"❌ Error al aplicar el preprocesamiento: {e}")
    st.stop()

# ==========================
# 🔮 PREDICCIONES
# ==========================
try:
    pred_rf = model_rf.predict_proba(X_new_clean)[0][1]
    pred_dt = model_dt.predict_proba(X_new_clean)[0][1]
    pred_lr = model_lr.predict_proba(X_new_clean)[0][1]
except Exception as e:
    st.error(f"⚠️ Error durante la predicción: {e}")
    st.stop()

# ==========================
# 📈 RESULTADOS
# ==========================
st.markdown("## 🔮 Resultados de Predicción")

models = ["Random Forest", "Decision Tree", "Logistic Regression"]
values = [pred_rf, pred_dt, pred_lr]

results_df = pd.DataFrame({
    "Modelo": models,
    "Probabilidad de empleo (%)": [v * 100 for v in values]
}).sort_values(by="Probabilidad de empleo (%)", ascending=False)

st.dataframe(results_df, width="stretch")

# ==========================
# 📊 GRÁFICO COMPARATIVO
# ==========================
fig, ax = plt.subplots(figsize=(6, 3))
ax.barh(models, values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
ax.set_xlim(0, 1)
ax.set_xlabel("Probabilidad de ser contratado", fontsize=10)
ax.set_title("Comparación entre modelos", fontsize=11, pad=10)
for i, v in enumerate(values):
    ax.text(v + 0.01, i, f"{v*100:.1f}%", va="center", fontsize=9)
st.pyplot(fig)

# ==========================
# 🧠 INTERPRETACIÓN FINAL
# ==========================
best_model = models[values.index(max(values))]
best_prob = max(values) * 100

st.markdown("---")
if best_prob >= 70:
    st.success(f"✅ Alta probabilidad de empleo ({best_prob:.1f}%) según **{best_model}**.")
elif best_prob >= 40:
    st.warning(f"⚠️ Probabilidad media de empleo ({best_prob:.1f}%) según **{best_model}**.")
else:
    st.error(f"🚫 Baja probabilidad de empleo ({best_prob:.1f}%) según **{best_model}**.")

st.caption("🎓 Proyecto de Machine Learning — Predicción de Empleabilidad con Modelos Supervisados")
