import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

# Configuración de la interfaz profesional
st.set_page_config(page_title="Calculadora de Viscosidad - Fenómenos de Transporte", layout="wide")

# Inicialización de la memoria de cálculos
if 'tabla_resultados' not in st.session_state:
    st.session_state.tabla_resultados = pd.DataFrame(columns=[
        "Componente", "Fórmula", "Experimental (μPa·s)", "Calculada (μPa·s)", "Error (%)"
    ])

# Diccionario de fórmulas químicas para la interfaz
formulas_quimicas = {
    "Hidrógeno": "H2", "Metano": "CH4", "Etileno": "C2H4", "Etano": "C2H6",
    "Acetileno": "C2H2", "Propileno": "C3H6", "Propano": "C3H8", "n-Butano": "C4H10"
}

# --- FUNCIONES MATEMÁTICAS Y CONVERSIONES ---
def calcular_omega_v(t_star):
    # Integral de colisión (Ecuación de Neufeld et al.)
    return 1.16145*(t_star**-0.14874) + 0.52487*np.exp(-0.77320*t_star) + 2.16178*np.exp(-2.43787*t_star)

# --- INTERFAZ PRINCIPAL ---
st.title("Determinación de Viscosidad de Mezclas Gaseosas")
st.sidebar.header("Módulos del Sistema")
modulo = st.sidebar.radio("Seleccione módulo:", ["Sustancia Pura", "Regla de Mezclado"])

if modulo == "Sustancia Pura":
    st.header("Cálculo para Sustancias Puras")
    modelo = st.selectbox("Seleccione el Modelo Matemático:", 
                          ["Chapman-Enskog", "Chung et al.", "Correlación DIPPR", "Stiel y Thodos"])
    
    # Explicación técnica del modelo
    st.subheader("Ecuación y Definiciones")
    if modelo == "Chapman-Enskog":
        st.latex(r"\mu = 26.69 \frac{\sqrt{M T}}{\sigma^2 \Omega_v}")
        st.markdown("""
        * **M:** Peso molecular (g/mol)
        * **T:** Temperatura absoluta (K)
        * **σ:** Diámetro de colisión (Å)
        * **Ωv:** Integral de colisión (adimensional)
        * **Factor 26.69:** Constante resultante de la teoría cinética de los gases.
        """)
    elif modelo == "Chung et al.":
        st.latex(r"\mu = \frac{40.785 F_c \sqrt{M T}}{V_c^{2/3} \Omega_v}")
        st.markdown("* **Fc:** Factor de corrección, **Vc:** Volumen crítico (cm³/mol), **Ωv:** Integral de colisión basada en Tr.")
    elif modelo == "Correlación DIPPR":
        st.latex(r"\mu = \frac{A T^B}{1 + C/T + D/T^2}")
        st.markdown("* **A, B, C, D:** Coeficientes empíricos específicos para cada sustancia.")
    elif modelo == "Stiel y Thodos":
        st.latex(r"\mu = \frac{N \cdot \sqrt{M} \cdot P_c^{2/3}}{T_c^{1/6}} \times 10^{-8}")
        st.markdown("* **N:** Función de la temperatura reducida, **Pc:** Presión crítica, **Tc:** Temperatura crítica.")

    st.divider()

    # Entrada de datos
    st.subheader("Entrada de Parámetros")
    col1, col2 = st.columns(2)
    
    with col1:
        comp_nombre = st.selectbox("Componente:", list(formulas_quimicas.keys()))
        t_entrada = st.number_input("Temperatura de Operación (K):", value=298.15, format="%.2f")
        v_exp_entrada = st.number_input("Viscosidad Experimental (μPa·s):", format="%.4f")
        
    with col2:
        if modelo == "Chapman-Enskog":
            m_in = st.number_input("Peso Molecular (M):", format="%.4f")
            sigma_in = st.number_input("Diámetro sigma (Å):", format="%.4f")
            eps_k_in = st.number_input("Parámetro epsilon/k (K):", format="%.2f")
        elif modelo == "Chung et al.":
            m_in = st.number_input("Peso Molecular (M):", format="%.4f")
            tc_in = st.number_input("Temperatura Crítica (Tc):", format="%.2f")
            vc_in = st.number_input("Volumen Crítico (Vc):", format="%.2f")
            w_in = st.number_input("Factor Acéntrico (w):", format="%.4f")
        elif modelo == "Correlación DIPPR":
            a_in = st.number_input("Coeficiente A:", format="%.4e")
            b_in = st.number_input("Coeficiente B:", format="%.4f")
            c_in = st.number_input("Coeficiente C:", format="%.4f")
            d_in = st.number_input("Coeficiente D:", format="%.4f")
        elif modelo == "Stiel y Thodos":
            m_in = st.number_input("Peso Molecular (M):", format="%.4f")
            tc_in = st.number_input("Temperatura Crítica (Tc):", format="%.2f")
            pc_in = st.number_input("Presión Crítica (Pc):", format="%.2f")

    if st.button("Ejecutar Cálculo"):
        try:
            # Lógica con conversiones integradas
            if modelo == "Chapman-Enskog":
                t_star = t_entrada / eps_k_in
                # El resultado original es en micropoise (uP), se divide por 10 para μPa·s
                mu_final = (26.69 * np.sqrt(m_in * t_entrada)) / (sigma_in**2 * calcular_omega_v(t_star)) / 10
            elif modelo == "Chung et al.":
                tr = t_entrada / tc_in
                fc = 1 - 0.2756 * w_in
                mu_final = (40.785 * fc * np.sqrt(m_in * t_entrada)) / (vc_in**(2/3) * calcular_omega_v(tr)) / 10
            elif modelo == "Correlación DIPPR":
                # DIPPR suele dar resultados en Pa·s, multiplicamos por 1e6 para μPa·s
                mu_final = ((a_in * t_entrada**b_in) / (1 + c_in/t_entrada + d_in/t_entrada**2)) * 1e6
            elif modelo == "Stiel y Thodos":
                tr = t_entrada / tc_in
                nv = 1.778*(4.58*tr - 1.67)**0.625 if tr > 1.5 else 3.4*(tr**0.94)
                # Conversión de unidades de Pc y Tc a μPa·s
                mu_final = (9.91e-8 * nv * np.sqrt(m_in) * (pc_in**(2/3)) / (tc_in**(1/6))) * 1e6

            # Cálculo de error individual
            err_ind = abs(v_exp_entrada - mu_final) / v_exp_entrada * 100 if v_exp_entrada > 0 else 0
            
            # Actualización de tabla
            nuevo_registro = pd.DataFrame([{
                "Componente": comp_nombre, 
                "Fórmula": formulas_quimicas[comp_nombre], 
                "Experimental (μPa·s)": v_exp_entrada, 
                "Calculada (μPa·s)": round(mu_final, 5), 
                "Error (%)": round(err_ind, 3)
            }])
            st.session_state.tabla_resultados = pd.concat([st.session_state.tabla_resultados, nuevo_registro], ignore_index=True)
            
            # Mostrar resultado inmediato
            st.success(f"Resultado para {comp_nombre} ({formulas_quimicas[comp_nombre]}): {mu_final:.5f} μPa·s | Error: {err_ind:.3f}%")
        except Exception as e:
            st.error(f"Error en el procesamiento de datos: {e}")

    # Visualización de Resultados Acumulados
    if not st.session_state.tabla_resultados.empty:
        st.divider()
        st.subheader("Resultados Acumulados")
        st.table(st.session_state.tabla_resultados)
        
        if st.button("Reiniciar Sistema"):
            st.session_state.tabla_resultados = pd.DataFrame(columns=["Componente", "Fórmula", "Experimental (μPa·s)", "Calculada (μPa·s)", "Error (%)"])
            st.rerun()

        # Análisis Estadístico y Gráfico
        df_validos = st.session_state.tabla_resultados[st.session_state.tabla_resultados["Experimental (μPa·s)"] > 0]
        if len(df_validos) > 0:
            y_exp = df_validos["Experimental (μPa·s)"]
            y_calc = df_validos["Calculada (μPa·s)"]
            
            mape_global = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            
            c_met_1, c_met_2 = st.columns(2)
            c_met_1.metric("Error Global (MAPE)", f"{mape_global:.3f} %")
            
            if len(df_validos) > 1:
                r_cuadrado = r2_score(y_exp, y_calc)
                c_met_2.metric("Coeficiente R²", f"{r_cuadrado:.5f}")

            # Gráfica de Dispersión
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_exp, y=y_calc, mode='markers+text', 
                                     text=df_validos["Fórmula"], textposition="top center",
                                     marker=dict(size=10, color='black'), name="Datos"))
            
            eje_lim = [min(min(y_exp), min(y_calc))*0.95, max(max(y_exp), max(y_calc))*1.05]
            fig.add_trace(go.Scatter(x=eje_lim, y=eje_lim, mode='lines', name='Línea de Identidad', line=dict(color='gray', dash='dot')))
            
            fig.update_layout(xaxis_title="Viscosidad Experimental (μPa·s)", yaxis_title="Viscosidad Calculada (μPa·s)", height=500)
            st.plotly_chart(fig)

elif modulo == "Regla de Mezclado":
    st.header("Reglas de Mezclado para Gases")
    metodo_mezcla = st.selectbox("Seleccione Método:", ["Wilke", "Davidson"])
    
    st.subheader("Cálculo de Interacciones Binarias")
    if metodo_mezcla == "Wilke":
        st.latex(r"\phi_{ij} = \frac{[1 + (\mu_i/\mu_j)^{1/2}(M_j/M_i)^{1/4}]^2}{\sqrt{8}(1 + M_i/M_j)^{1/2}}")
    else:
        st.latex(r"\Psi_{ij} = \sqrt{\frac{M_j}{M_i}}")
        
    st.info("Ingrese las fracciones molares y viscosidades de cada componente para calcular la mezcla.")
    # (Aquí se puede expandir una tabla dinámica para los 8 componentes)
