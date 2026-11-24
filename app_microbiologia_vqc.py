import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import zz_feature_map, real_amplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Clasificador Microbiológico Cuántico (VQC)", page_icon="🔬", layout="wide")
st.title("🔬 Clasificador Cuántico de Riesgo Microbiológico (VQC)")
st.subheader("Triaje Inteligente en Recepción de Muestras de Laboratorio (Aguas, Alimentos y Pharma)")
st.write("Herramienta de **Quantum Machine Learning (IBM Qiskit)** que analiza parámetros físico-químicos inmediatos para priorizar cultivos urgentes antes de incubación.")

# --- FICHA EJECUTIVA DE NEGOCIO Y ROI ---
with st.expander("📊 Impacto en Laboratorio, Perfiles Objetivo y Ventaja Tecnológica", expanded=True):
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        * **Perfiles y Sectores Objetivo:** Directores de calidad, microbiólogos, laboratorios de inspección y certificación (aguas, industria agroalimentaria, farmacéutica) y responsables de logística en frío.
        * **Cuello de botella que resuelve:** Un cultivo microbiológico tradicional tarda entre **24 y 72 horas**. Cuando entran cientos de lotes al día, el laboratorio necesita saber al instante cuáles traen mayor riesgo de proliferación bacteriana para meterlos primero en estufa.
        """)
    with col_info2:
        st.markdown("""
        * **Ventaja del Aprendizaje Automático Cuántico (VQC):** Mediante entrelazamiento cuántico (`ZZFeatureMap`), el circuito captura interacciones no lineales entre temperatura, pH, turbidez y tiempo de tránsito con pocos datos de calibración.
        * **Validación en Hardware Real (QPU):** Este entorno ejecuta simulación matemática exacta de Qiskit. Para ejecutar pruebas piloto sobre procesadores físicos de **IBM Quantum**, contactar al equipo de desarrollo al pie de página.
        """)

# --- ENTRENAMIENTO CACHEADO DEL MODELO CUÁNTICO VQC ---
@st.cache_resource
def entrenar_modelo_vqc():
    datos_base = pd.DataFrame({
        'Temp_C': list((3.5, 4.0, 2.8, 4.2, 14.5, 18.0, 12.0, 21.5)),
        'pH': list((4.2, 4.5, 4.0, 4.8, 6.9, 7.2, 6.7, 7.4)),
        'Turbidez_NTU': list((1.2, 1.5, 0.9, 1.8, 8.5, 11.0, 7.2, 14.5)),
        'Horas_Transito': list((4.0, 6.0, 3.5, 5.0, 18.0, 24.0, 16.0, 28.0)),
        'Riesgo_Real': list((0, 0, 0, 0, 1, 1, 1, 1))
    })
    cols = ['Temp_C', 'pH', 'Turbidez_NTU', 'Horas_Transito']
    escalador = MinMaxScaler(feature_range=(0, np.pi))
    angulos_train = escalador.fit_transform(datos_base[cols])
    
    f_map = zz_feature_map(feature_dimension=4, reps=1)
    ansatz = real_amplitudes(num_qubits=4, reps=1)
    circuito = f_map.compose(ansatz)
    observable = SparsePauliOp.from_list([("ZZZZ", 1.0)])
    estimador = StatevectorEstimator()
    objetivos = np.where(datos_base['Riesgo_Real'] == 0, -1.0, 1.0)
    
    def perdida(pesos):
        tareas = [(circuito, observable, np.concatenate([angulos_train[i], pesos])) for i in range(len(angulos_train))]
        res = estimador.run(tareas).result()
        return sum((float(r.data.evs) - objetivos[i]) ** 2 for i, r in enumerate(res)) / len(angulos_train)
        
    opt = minimize(perdida, np.zeros(ansatz.num_parameters), method="COBYLA", options={'maxiter': 80})
    return escalador, circuito, observable, opt.x

escalador_vqc, circuito_vqc, observable_vqc, pesos_entrenados = entrenar_modelo_vqc()

def predecir_riesgo_lote(temp, ph, turbidez, horas):
    df_in = pd.DataFrame({'Temp_C': [temp], 'pH': [ph], 'Turbidez_NTU': [turbidez], 'Horas_Transito': [horas]})
    angulos = escalador_vqc.transform(df_in)
    params = np.concatenate([angulos.flatten(), pesos_entrenados])
    estimador = StatevectorEstimator()
    val = float(next(iter(estimador.run([(circuito_vqc, observable_vqc, params)]).result())).data.evs)
    prob = round(((val + 1) / 2) * 100, 1)
    return max(0.0, min(100.0, prob))

# --- MENÚ LATERAL ---
st.sidebar.header("🧪 Modo de Análisis de Muestras")
modo = st.sidebar.radio(
    "Selecciona cómo introducir los datos:",
    (
        "1. Muestras de Referencia (Recepción)",
        "2. Control Numérico Manual",
        "3. Subir Lotes en Lote (CSV / Excel)"
    )
)

# =========================================================
# MODO 1 Y MODO 2: ANÁLISIS INDIVIDUAL DE MUESTRA
# =========================================================
if modo != "3. Subir Lotes en Lote (CSV / Excel)":
    if modo == "1. Muestras de Referencia (Recepción)":
        muestra_sel = st.sidebar.selectbox(
            "Selecciona una muestra recibida:",
            (
                "Lote A: Agua Potable Red (Fría y Limpia)",
                "Lote B: Extracto Marino (Rotura Cadena de Frío)",
                "Lote C: Muestra Agroalimentaria (Tránsito Prolongado)"
            )
        )
        if "Lote A" in muestra_sel:
            nombre_lote, temp, ph, turb, horas = "Lote_A_Agua_Red", 3.8, 4.4, 1.1, 4.5
        elif "Lote B" in muestra_sel:
            nombre_lote, temp, ph, turb, horas = "Lote_B_Extracto_Marino", 17.5, 7.1, 10.2, 22.0
        else:
            nombre_lote, temp, ph, turb, horas = "Lote_C_Agroalimentario", 13.5, 6.8, 7.8, 19.0
    else:
        st.sidebar.subheader("🎛️ Parámetros Físico-Químicos")
        nombre_lote = st.sidebar.text_input("Identificador del lote:", "Muestra_Lab_2026")
        temp = st.sidebar.slider("Temperatura de transporte (°C):", 0.0, 30.0, 15.0, 0.5)
        ph = st.sidebar.slider("Nivel de pH:", 3.0, 9.0, 7.0, 0.1)
        turb = st.sidebar.slider("Turbidez (NTU):", 0.1, 20.0, 8.5, 0.2)
        horas = st.sidebar.slider("Tiempo en tránsito (horas):", 1.0, 36.0, 18.0, 0.5)

    st.subheader(f"📋 Ficha de Recepción: {nombre_lote}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperatura", f"{temp} °C")
    c2.metric("Nivel pH", f"{ph}")
    c3.metric("Turbidez", f"{turb} NTU")
    c4.metric("Tiempo Tránsito", f"{horas} h")

    if st.button("🚀 Evaluar Riesgo Microbiológico con VQC"):
        with st.spinner("Ejecutando inferencia en circuito cuántico variacional..."):
            riesgo = predecir_riesgo_lote(temp, ph, turb, horas)
            
            if riesgo >= 50.0:
                st.error(f"🚨 **ALTO RIESGO MICROBIOLÓGICO ({riesgo}%):** Priorizar cultivo urgente en estufa y cuarentena preventiva.")
                motivo = "La combinación de temperatura elevada, pH cercano a la neutralidad o exceso de horas en tránsito activa la zona de alerta cuántica."
            else:
                st.success(f"✅ **LOTE APTO / BAJO RIESGO ({riesgo}%):** Cumple parámetros de estabilidad; continuar protocolo estándar.")
                motivo = "Los parámetros físico-químicos mantienen inhibido el crecimiento bacteriano esperado."
            st.info(f"💡 **Interpretación técnica:** {motivo}")

            fig, ax = plt.subplots(figsize=(8, 2.2))
            color_barra = '#d62728' if riesgo >= 50.0 else '#2ca02c'
            ax.barh([nombre_lote], [riesgo], color=color_barra, height=0.4)
            ax.axvline(50, color='black', linestyle='--', linewidth=1.2, label='Umbral de Alerta (50%)')
            ax.set_xlim(0, 100)
            ax.set_xlabel("Índice de Riesgo Microbiológico Cuántico (%)")
            ax.set_title("Clasificación VQC frente al Umbral Crítico de Laboratorio")
            ax.legend(loc='lower right')
            st.pyplot(fig)

# =========================================================
# MODO 3: PROCESAMIENTO EN LOTE (CSV O EXCEL)
# =========================================================
else:
    st.subheader("📂 Triaje Masivo de Muestras en Recepción (CSV o Excel)")
    st.write("Sube tu hoja de recepción de laboratorio con las columnas: `Lote`, `Temp_C`, `pH`, `Turbidez_NTU`, `Horas_Transito`.")
    
    plantilla = pd.DataFrame({
        "Lote": ["Muestra_01_Agua", "Muestra_02_Lacteo", "Muestra_03_Marisco", "Muestra_04_Zumo"],
        "Temp_C": list((3.2, 16.5, 19.0, 4.1)),
        "pH": list((4.3, 6.9, 7.3, 3.8)),
        "Turbidez_NTU": list((1.0, 9.2, 12.5, 2.1)),
        "Horas_Transito": list((4.0, 20.0, 26.0, 5.5))
    })
    st.download_button(
        label="📥 Descargar plantilla CSV de laboratorio",
        data=plantilla.to_csv(index=False).encode('utf-8'),
        file_name="plantilla_recepcion_microbiologia.csv",
        mime="text/csv"
    )
    
    archivo = st.file_uploader("Arrastra aquí tu archivo CSV o Excel (.xlsx)", type=["csv", "xlsx"])
    if archivo is not None:
        df_subido = pd.read_csv(archivo) if archivo.name.endswith('.csv') else pd.read_excel(archivo)
        st.write("Vista previa de lotes cargados:", df_subido)
        
        if st.button("🚀 Clasificar Todos los Lotes con VQC"):
            resultados = []
            barra = st.progress(0)
            total = len(df_subido)
            for idx, fila in df_subido.iterrows():
                r_pct = predecir_riesgo_lote(
                    float(fila.get("Temp_C")),
                    float(fila.get("pH")),
                    float(fila.get("Turbidez_NTU")),
                    float(fila.get("Horas_Transito"))
                )
                prioridad = "🚨 URGENTE (Cultivo Inmediato)" if r_pct >= 50.0 else "✅ ESTÁNDAR (Apto)"
                resultados.append({
                    "Lote": fila.get("Lote"),
                    "Riesgo_Cuantico_Pct": r_pct,
                    "Prioridad_Laboratorio": prioridad
                })
                barra.progress((idx + 1) / total)
                
            df_res = pd.DataFrame(resultados).sort_values(by="Riesgo_Cuantico_Pct", ascending=False)
            st.success("✅ Triaje cuántico completado. Lotes ordenados por urgencia microbiológica:")
            st.dataframe(df_res, use_container_width=True)
st.divider()
st.subheader("💬 Validación en Chip Real IBM y Colaboraciones")
col_c1, col_c2 = st.columns(2)
with col_c1:
if st.button("👍 ¡Esta herramienta me ha resultado útil!"):
st.toast("¡Gracias por tu valoración! Seguimos innovando en calidad y biotecnología.")
with col_c2:
st.markdown("¿Deseas validar este clasificador en hardware cuántico real de IBM Quantum o adaptarlo a tu laboratorio?")
st.markdown("📩 **Correo directo:** [nhemesysgonzalez@gmail.com](mailto:nhemesysgonzalez@gmail.com) | 💬 **[Comentarios en GitHub](https://github.com/nhemesysgonzalez-stack/clasificador-microbiologico-vqc/issues)**")


