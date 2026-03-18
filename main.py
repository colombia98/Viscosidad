import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Simulador de Viscosidad FTIQ", layout="wide")

# --- BASE DE DATOS DE CONSTANTES (Referencia interna para validación) ---
# El usuario puede ingresar estos datos, pero los tenemos aquí para el cálculo de error global
DB_GASEOSA = {
    "Hidrógeno": {"Formula": "H₂", "M": 2.016, "sigma": 2.827, "eps_k": 59.7, "Tc": 32.98, "Pc": 12.76, "Vc": 64.2, "w": -0.217, "A": 1.797e-7, "B": 0.6852, "C": -0.59, "D": 140.0, "mu_exp": 8.97, "xi": 0.3961},
    "Metano": {"Formula": "CH₄", "M": 16.043, "sigma": 3.758, "eps_k": 148.6, "Tc": 190.56, "Pc": 45.99, "Vc": 98.6, "w": 0.011, "A": 5.254e-7, "B": 0.5901, "C": 105.67, "D": 0.0, "mu_exp": 11.072, "xi": 0.1384},
    "Etileno": {"Formula": "C₂H₄", "M": 28.054, "sigma": 4.163, "eps_k": 224.7, "Tc": 282.34, "Pc": 50.41, "Vc": 131.1, "w": 0.087, "A": 2.064e-7, "B": 0.7202, "C": 270.0, "D": 0.0, "mu_exp": 10.23, "xi": 0.2593},
    "Etano": {"Formula": "C₂H₆", "M": 30.07, "sigma": 4.443, "eps_k": 215.7, "Tc": 305.32, "Pc": 48.72, "Vc": 145.5, "w": 0.099, "A": 2.592e-7, "B": 0.6901, "C": 252.6, "D": 0.0, "mu_exp": 9.248, "xi": 0.1359},
    "Acetileno": {"Formula": "C₂H₂", "M": 26.038, "sigma": 4.033, "eps_k": 231.8, "Tc": 308.3, "Pc": 61.38, "Vc": 113.0, "w": 0.187, "A": 4.432e-7, "B": 0.6105, "C": 185.0, "D": 0.0, "mu_exp": 11.62, "xi": 0.0016},
    "Propileno": {"Formula": "C₃H₆", "M": 42.081, "sigma": 4.678, "eps_k": 298.9, "Tc": 364.9, "Pc": 46.0, "Vc": 184.6, "w": 0.142, "A": 1.612e-7, "B": 0.7505, "C": 310.2, "D": 0.0, "mu_exp": 8.583, "xi": 0.0019},
    "Propano": {"Formula": "C₃H₈", "M": 44.096, "sigma": 5.118, "eps_k": 237.1, "Tc": 369.8, "Pc": 42.5, "Vc": 200.0, "w": 0.152, "A": 1.303e-7, "B": 0.7803, "C": 350.0, "D": 0.0, "mu_exp": 8.115, "xi": 0.0001},
    "n-Butano": {"Formula": "C₄H₁₀", "M": 58.122, "sigma": 4.687, "eps_k": 531.4, "Tc": 425.1, "Pc": 37.96, "Vc": 255.0, "w": 0.2, "A": 0.891e-7, "B": 0.8304, "C": 450.0, "D": 0.0, "mu_exp": 8.0, "xi": 0.0667}
}

# --- FUNCIONES DE CÁLCULO ---
def omega_v(T_star):
    return 1.16145*(T_star**-0.14874) + 0.52487*np.exp(-0.77320*T_star) + 2.16178*np.exp(-2.43787*T_star)

# --- INTERFAZ ---
st.title("🖥️ Software de Predicción de Viscosidad - Nivel Tesis")
st.sidebar.header("Menú de Navegación")
fase = st.sidebar.selectbox("Seleccione Fase:", ["Mezcla Gaseosa (Cracking Etano)", "Mezcla Líquida (Próximamente)"])

if fase == "Mezcla Gaseosa (Cracking Etano)":
    modulo = st.sidebar.radio("Módulo:", ["Sustancias Puras", "Reglas de Mezclado"])
    
    if modulo == "Sustancias Puras":
        st.header("1. Modelos de Viscosidad para Sustancias Puras")
        modelo = st.selectbox("Seleccione el Modelo:", ["Chapman-Enskog", "Chung et al.", "Stiel-Thodos", "Correlación DIPPR"])
        
        # --- EXPLICACIÓN DEL MODELO SELECCIONADO ---
        if modelo == "Chapman-Enskog":
            st.latex(r"\eta = \frac{26.69 \sqrt{MT}}{\sigma^2 \Omega_v}")
            st.write("**Donde:** η: Viscosidad (μP), M: Masa molar (g/mol), T: Temperatura (K), σ: Diámetro de colisión (Å), Ωv: Integral de colisión.")
        elif modelo == "Chung et al.":
            st.latex(r"\mu = \frac{40.785 F_c \sqrt{MT}}{V_c^{2/3} \Omega_\mu}")
            st.write("**Donde:** Fc: Factor de corrección molecular, Vc: Volumen crítico (cm³/mol), Ωμ: Integral de colisión.")
        elif modelo == "Stiel-Thodos":
            st.latex(r"\mu = 9.91 \times 10^{-8} \frac{N M^{1/2} P_c^{2/3}}{T_c^{1/6}}")
            st.write("**Donde:** N: Función de Tr, Pc: Presión crítica (atm), Tc: Temperatura crítica (K).")
        elif modelo == "Correlación DIPPR":
            st.latex(r"\eta = \frac{A \cdot T^B}{1 + C/T + D/T^2}")
            st.write("**Donde:** A, B, C, D: Coeficientes específicos del compuesto.")

        # --- ENTRADA DE DATOS DINÁMICA ---
        st.subheader("Entrada de Parámetros de Operación")
        col1, col2 = st.columns(2)
        with col1:
            comp_sel = st.selectbox("Seleccione el Compuesto a calcular:", list(DB_GASEOSA.keys()))
            comp_data = DB_GASEOSA[comp_sel]
            st.info(f"Fórmula Química: {comp_data['Formula']}")
        with col2:
            T_user = st.number_input("Temperatura del sistema (K):", value=298.15)
        
        st.write("---")
        st.subheader(f"Datos requeridos para {modelo}")
        
        # Pedir datos específicos según el modelo
        input_data = {}
        c1, c2, c3 = st.columns(3)
        if modelo == "Chapman-Enskog":
            input_data['M'] = c1.number_input("Masa Molar (M):", value=comp_data['M'])
            input_data['sigma'] = c2.number_input("Diámetro σ (Å):", value=comp_data['sigma'])
            input_data['eps_k'] = c3.number_input("Parámetro ε/κ (K):", value=comp_data['eps_k'])
        elif modelo == "Chung et al.":
            input_data['M'] = c1.number_input("Masa Molar (M):", value=comp_data['M'])
            input_data['Vc'] = c2.number_input("Volumen Crítico (Vc):", value=comp_data['Vc'])
            input_data['Tc'] = c3.number_input("Temperatura Crítica (Tc):", value=comp_data['Tc'])
            input_data['w'] = st.number_input("Factor Acéntrico (ω):", value=comp_data['w'])
        elif modelo == "Stiel-Thodos":
            input_data['M'] = c1.number_input("Masa Molar (M):", value=comp_data['M'])
            input_data['Pc'] = c2.number_input("Presión Crítica (Pc) [atm]:", value=comp_data['Pc'])
            input_data['Tc'] = c3.number_input("Temperatura Crítica (Tc):", value=comp_data['Tc'])
        elif modelo == "Correlación DIPPR":
            input_data['A'] = c1.number_input("Coeficiente A:", value=comp_data['A'], format="%.4e")
            input_data['B'] = c2.number_input("Coeficiente B:", value=comp_data['B'])
            input_data['C'] = c3.number_input("Coeficiente C:", value=comp_data['C'])
            input_data['D'] = st.number_input("Coeficiente D:", value=comp_data['D'])

        if st.button("Ejecutar Cálculo y Comparación"):
            # Lógica de cálculo
            if modelo == "Chapman-Enskog":
                T_star = T_user / input_data['eps_k']
                res = (26.69 * np.sqrt(input_data['M'] * T_user)) / (input_data['sigma']**2 * omega_v(T_star)) / 10
            elif modelo == "Chung et al.":
                Tr = T_user / input_data['Tc']
                Fc = 1 - 0.2756 * input_data['w']
                res = (40.785 * Fc * np.sqrt(input_data['M'] * T_user)) / (input_data['Vc']**(2/3) * omega_v(Tr)) / 10
            elif modelo == "Stiel-Thodos":
                Tr = T_user / input_data['Tc']
                N = 1.778*(4.58*Tr - 1.67)**0.625 if Tr > 1.5 else 3.4*(Tr**0.94)
                res = (9.91e-8 * N * np.sqrt(input_data['M']) * (input_data['Pc']**(2/3)) / (input_data['Tc']**(1/6))) * 1e6
            else:
                res = ((input_data['A'] * T_user**input_data['B']) / (1 + input_data['C']/T_user + input_data['D']/T_user**2)) * 1e6
            
            st.success(f"Viscosidad Calculada: {res:.4f} μPa·s")
            st.info(f"Viscosidad Experimental: {comp_data['mu_exp']} μPa·s")
            
            # --- CÁLCULO DE ERROR GLOBAL (PARA TODOS LOS COMPUESTOS) ---
            st.divider()
            st.subheader("📊 Validación Estadística del Modelo")
            
            y_exp, y_calc = [], []
            for k, v in DB_GASEOSA.items():
                y_exp.append(v['mu_exp'])
                # Repetir cálculo interno para estadística
                if modelo == "Chapman-Enskog":
                    val = (26.69 * np.sqrt(v['M'] * T_user)) / (v['sigma']**2 * omega_v(T_user/v['eps_k'])) / 10
                elif modelo == "Chung et al.":
                    val = (40.785 * (1-0.2756*v['w']) * np.sqrt(v['M'] * T_user)) / (v['Vc']**(2/3) * omega_v(T_user/v['Tc'])) / 10
                elif modelo == "Stiel-Thodos":
                    Tr_i = T_user/v['Tc']
                    N_i = 1.778*(4.58*Tr_i - 1.67)**0.625 if Tr_i > 1.5 else 3.4*(Tr_i**0.94)
                    val = (9.91e-8 * N_i * np.sqrt(v['M']) * (v['Pc']**(2/3)) / (v['Tc']**(1/6))) * 1e6
                else:
                    val = ((v['A'] * T_user**v['B']) / (1 + v['C']/T_user + v['D']/T_user**2)) * 1e6
                y_calc.append(val)
            
            mape = np.mean(np.abs((np.array(y_exp) - np.array(y_calc)) / np.array(y_exp))) * 100
            r2 = r2_score(y_exp, y_calc)
            
            col_a, col_b = st.columns(2)
            col_a.metric("Error Global (MAPE)", f"{mape:.2f} %")
            col_b.metric("Coeficiente R²", f"{r2:.4f}")
            
            # Gráfica
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_exp, y=y_calc, mode='markers', marker=dict(size=12, color='blue'), name="Compuestos"))
            fig.add_trace(go.Scatter(x=[min(y_exp), max(y_exp)], y=[min(y_exp), max(y_exp)], mode='lines', name='Línea Ideal', line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title="Viscosidad Experimental (μPa·s)", yaxis_title="Viscosidad Calculada (μPa·s)")
            st.plotly_chart(fig)

    elif modulo == "Reglas de Mezclado":
        st.header("2. Reglas de Mezclado para Gases Multicomponentes")
        regla = st.selectbox("Seleccione Regla:", ["Método de Wilke", "Método de Davidson"])
        
        if regla == "Método de Wilke":
            st.latex(r"\mu_{mix} = \sum_{i=1}^n \frac{x_i \mu_i}{\sum_{j=1}^n x_j \phi_{ij}}")
        else:
            st.latex(r"\mu_{mix} = \sum_{i=1}^n \frac{x_i \mu_i}{\sum_{j=1}^n x_j \Psi_{ij}}")
            
        st.write("Ajuste las fracciones molares (xi) de la mezcla:")
        
        # Tabla editable
        df_xi = pd.DataFrame([{"Compuesto": k, "xi": v["xi"], "M": v["M"], "mu": v["mu_exp"]} for k, v in DB_GASEOSA.items()])
        edited_df = st.data_editor(df_xi, num_rows="fixed")
        
        if st.button("Calcular Viscosidad de la Mezcla"):
            sum_xi = edited_df["xi"].sum()
            if abs(sum_xi - 1.0) > 0.02:
                st.error(f"La suma de fracciones molares debe ser 1.0 (Actual: {sum_xi:.3f})")
            else:
                # Cálculo de Wilke
                n = len(edited_df)
                mu_vals = edited_df["mu"].values
                M_vals = edited_df["M"].values
                xi_vals = edited_df["xi"].values
                
                if regla == "Método de Wilke":
                    phi = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            num = (1 + (mu_vals[i]/mu_vals[j])**0.5 * (M_vals[j]/M_vals[i])**0.25)**2
                            den = 8**0.5 * (1 + M_vals[i]/M_vals[j])**0.5
                            phi[i, j] = num / den
                    
                    denom = np.dot(phi, xi_vals)
                    mu_mix = np.sum(xi_vals * mu_vals / denom)
                else: # Davidson
                    psi = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            psi[i,j] = (M_vals[j]/M_vals[i])**0.5
                    denom = np.dot(psi, xi_vals)
                    mu_mix = np.sum(xi_vals * mu_vals / denom)
                
                st.success(f"Viscosidad de la Mezcla ({regla}): {mu_mix:.4f} μPa·s")
