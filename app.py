import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sys
import streamlit as st
st.write("Python version:", sys.version)

# -----------------------------
# CARGAR MODELOS
# -----------------------------
rf = joblib.load('modelo_Random_Forest.pkl')
dt = joblib.load('modelo_Decision_Tree.pkl')
lr = joblib.load('modelo_Logistic_Regression.pkl')

st.set_page_config(page_title="Predicción de Empleabilidad", page_icon="🎓", layout="wide")

st.title("🎓 Predicción de Empleabilidad Post-Graduación")
st.markdown("""
Este panel compara tres modelos (Random Forest, Decision Tree y Logistic Regression) 
entrenados sobre factores académicos y personales que influyen en la empleabilidad de un estudiante.
""")

# -----------------------------
# ENTRADAS DE USUARIO
# -----------------------------
st.sidebar.header("🧮 Factores de entrada")

IQ = st.sidebar.slider("IQ", 80, 160, 110)
Prev_Sem_Result = st.sidebar.slider("Promedio del semestre previo", 0.0, 10.0, 7.5)
CGPA = st.sidebar.slider("CGPA final", 0.0, 10.0, 7.0)
Academic_Performance = st.sidebar.slider("Desempeño académico (1-10)", 1, 10, 8)
Internship_Experience = st.sidebar.selectbox("¿Tuvo experiencia en prácticas?", ["No", "Sí"])
Extra_Curricular_Score = st.sidebar.slider("Actividades extracurriculares (1-10)", 1, 10, 5)
Communication_Skills = st.sidebar.slider("Habilidades de comunicación (1-10)", 1, 10, 7)
Projects_Completed = st.sidebar.slider("Proyectos completados", 0, 10, 3)

# -----------------------------
# CONSTRUCCIÓN DEL DATAFRAME
# -----------------------------
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

# -----------------------------
# PREDICCIONES
# -----------------------------
pred_rf = rf.predict_proba(input_data)[0][1]
pred_dt = dt.predict_proba(input_data)[0][1]
pred_lr = lr.predict_proba(input_data)[0][1]

# -----------------------------
# RESULTADOS
# -----------------------------
st.subheader("🔮 Resultados de Predicción")
st.write(f"**Random Forest:** {pred_rf:.2f}")
st.write(f"**Decision Tree:** {pred_dt:.2f}")
st.write(f"**Logistic Regression:** {pred_lr:.2f}")

# -----------------------------
# VISUALIZACIÓN
# -----------------------------
models = ['Random Forest', 'Decision Tree', 'Logistic Regression']
values = [pred_rf, pred_dt, pred_lr]

fig, ax = plt.subplots()
ax.barh(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_xlim(0, 1)
ax.set_xlabel('Probabilidad de ser contratado')
ax.set_title('Comparación entre modelos')
st.pyplot(fig)

# -----------------------------
# INTERPRETACIÓN
# -----------------------------
best_model = models[values.index(max(values))]
st.success(f"✅ El modelo con mayor probabilidad predice que estás **{max(values)*100:.1f}% empleable** según **{best_model}**.")
st.markdown("---")
st.caption("Desarrollado por Samantha Quintanchala — Proyecto Final de Machine Learning")


