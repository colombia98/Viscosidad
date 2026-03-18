import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="Software de Viscosidad - FTIQ", layout="wide")

st.markdown("""
    <style>
    .intermedio { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0056b3; font-family: monospace; font-size: 14px;}
    .intermedio-liq { background-color: #fcf8e3; padding: 15px; border-radius: 5px; border-left: 4px solid #e0a800; font-family: monospace; font-size: 14px;}
    </style>
    """, unsafe_allow_html=True)

# --- BASES DE DATOS DE REFERENCIA ---
COMPUESTOS_INFO = {
    "Hidrógeno": {"Formula": "H2", "M": 2.016},
    "Metano": {"Formula": "CH4", "M": 16.043},
    "Etileno": {"Formula": "C2H4", "M": 28.054},
    "Etano": {"Formula": "C2H6", "M": 30.070},
    "Acetileno": {"Formula": "C2H2", "M": 26.038},
    "Propileno": {"Formula": "C3H6", "M": 42.081},
    "Propano": {"Formula": "C3H8", "M": 44.096},
    "n-Butano": {"Formula": "C4H10", "M": 58.122}
}

COMPUESTOS_LIQ_INFO = {
    "n-Heptano": {"Formula": "C7H16", "M": 100.20},
    "n-Octano": {"Formula": "C8H18", "M": 114.23},
    "n-Nonano": {"Formula": "C9H20", "M": 128.26},
    "Ciclohexano": {"Formula": "C6H12", "M": 84.16},
    "Metilciclohexano": {"Formula": "C7H14", "M": 98.19},
    "Tolueno": {"Formula": "C7H8", "M": 92.14},
    "Etilbenceno": {"Formula": "C8H10", "M": 106.17},
    "p-Xileno": {"Formula": "C8H10", "M": 106.17},
    "m-Xileno": {"Formula": "C8H10", "M": 106.17},
    "o-Xileno": {"Formula": "C8H10", "M": 106.17}
}

if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])

if 'tabla_puras_liq' not in st.session_state:
    st.session_state.tabla_puras_liq = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (cP)", "Calc (cP)", "Error (%)"])

# --- FUNCIONES MATEMÁTICAS (GASES) ---
def omega_v(T_star):
    return 1.16145*(T_star**-0.14874) + 0.52487*np.exp(-0.77320*T_star) + 2.16178*np.exp(-2.43787*T_star)

# --- INTERFAZ PRINCIPAL ---
st.title("Determinación de Viscosidad - FTIQ")
st.info("💡 **Nota sobre decimales:** Dependiendo de tu sistema operativo y navegador, el programa puede requerir coma (,) o punto (.) para los decimales.")

st.sidebar.header("Configuración del Sistema")
fase = st.sidebar.radio("Fase de Operación:", ["Gas (Cracking de Etano)", "Líquido (Nafta Pesada)"])

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

# =====================================================================
# ======================== MÓDULO DE GASES ============================
# =====================================================================
if fase == "Gas (Cracking de Etano)":
    if menu == "Sustancias Puras":
        st.header("Análisis de Componentes Puros (Gas)")
        modelo = st.selectbox("Modelo Matemático:", ["Chung et al.", "Chapman-Enskog", "Stiel y Thodos", "DIPPR"])

        with st.expander("📖 Ver Ecuación y Variables del Modelo", expanded=True):
            if modelo == "Chung et al.":
                st.latex(r"\mu = \frac{40.785 F_c \sqrt{M T}}{V_c^{2/3} \Omega_v}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica calculada (en μP, luego convertida a μPa·s).
                * **Fc**: Factor empírico que depende del factor acéntrico (ω) y el momento dipolar (μ_r).
                * **M**: Masa molar del compuesto (g/mol).
                * **T**: Temperatura del sistema (K).
                * **Vc**: Volumen crítico (cm³/mol).
                * **Ωv**: Integral de colisión (dependiente de la temperatura reducida).
                """)
            elif modelo == "Chapman-Enskog":
                st.latex(r"\mu = \frac{26.69 \sqrt{M T}}{\sigma^2 \Omega_v}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica calculada.
                * **M**: Masa molar del compuesto (g/mol).
                * **T**: Temperatura del sistema (K).
                * **σ**: Diámetro de colisión de Lennard-Jones (Å).
                * **Ωv**: Integral de colisión (dependiente de T y ε/κ).
                """)
            elif modelo == "Stiel y Thodos":
                st.latex(r"\mu \xi = 34 \times 10^{-5} T_r^{0.94} \quad \text{para } T_r \leq 1.5")
                st.latex(r"\mu \xi = 17.78 \times 10^{-5} (4.58 T_r - 1.67)^{0.625} \quad \text{para } T_r > 1.5")
                st.latex(r"\xi = \frac{T_c^{1/6}}{M^{1/2} P_c^{2/3}}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica del gas a baja presión.
                * **Tr**: Temperatura reducida ($T / T_c$).
                * **ξ**: Parámetro de correlación.
                * **Tc**: Temperatura crítica (K).
                * **Pc**: Presión crítica (atm).
                * **M**: Masa molar (g/mol).
                """)
            elif modelo == "DIPPR":
                st.latex(r"\mu = \frac{A \cdot T^B}{1 + \frac{C}{T} + \frac{D}{T^2}}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica empírica.
                * **T**: Temperatura de operación (K).
                * **A, B, C, D**: Coeficientes de regresión de DIPPR.
                """)

        st.subheader("Entrada de Datos")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
            t_abs = st.number_input("Temperatura de Operación (K):", value=298.15)
            v_exp = st.number_input("Viscosidad Experimental (μPa·s):", value=0.0000, format="%.4f")
        
        with c2:
            m_in = st.number_input("Masa Molar (M):", value=COMPUESTOS_INFO[comp]["M"], format="%.4f")
            if modelo == "Chapman-Enskog":
                sigma = st.number_input("Sigma (σ en Å):", value=0.0000, format="%.4f")
                eps_k = st.number_input("Epsilon/k (ε/κ en K):", value=0.00, format="%.2f")
            elif modelo == "Chung et al.":
                tc = st.number_input("Temp. Crítica (Tc en K):", value=0.00, format="%.2f")
                vc = st.number_input("Vol. Crítico (Vc en cm³/mol):", value=0.00, format="%.2f")
            elif modelo == "Stiel y Thodos":
                tc = st.number_input("Temp. Crítica (Tc en K):", value=0.00, format="%.2f")
                pc = st.number_input("Pres. Crítica (Pc en atm):", value=0.00, format="%.2f")
            elif modelo == "DIPPR":
                a_dip = st.number_input("Coeficiente A:", format="%.4e")
                b_dip = st.number_input("Coeficiente B:", format="%.4f")
                
        with c3:
            if modelo == "Chung et al.":
                w = st.number_input("Factor Acéntrico (ω):", value=0.0000, format="%.4f")
                dipolo = st.number_input("Momento Dipolar (μ en Debyes):", value=0.0000, format="%.4f")
            elif modelo == "DIPPR":
                c_dip = st.number_input("Coeficiente C:", format="%.4f")
                d_dip = st.number_input("Coeficiente D:", format="%.4f")

        if st.button("Ejecutar Cálculo"):
            try:
                mu_c = 0.0
                if modelo == "Chung et al.":
                    tr = t_abs / tc
                    t_star_chung = 1.2593 * tr
                    omega = omega_v(t_star_chung)
                    mu_r = 131.3 * dipolo / np.sqrt(vc * tc)
                    fc = 1 - 0.2756 * w + 0.059035 * (mu_r**4)
                    mu_uP = (40.785 * fc * np.sqrt(m_in * t_abs)) / (vc**(2/3) * omega)
                    mu_c = mu_uP / 10 # uPa.s
                    
                    st.markdown(f"""<div class='intermedio'>
                    <b>🔢 Auditoría de Ecuaciones (Chung et al.):</b><br><br>
                    T* = 1.2593 * ({t_abs} / {tc}) = <b>{t_star_chung:.4f}</b><br>
                    Ωv = <b>{omega:.4f}</b><br>
                    μ_r = 131.3 * {dipolo} / √({vc} * {tc}) = <b>{mu_r:.5f}</b><br>
                    Fc = 1 - 0.2756({w}) + 0.059035({mu_r:.4f})^4 = <b>{fc:.5f}</b><br>
                    μ = [40.785 * {fc:.4f} * √({m_in} * {t_abs})] / [{vc}^(2/3) * {omega:.4f}] = <b>{mu_uP:.2f} μP</b>
                    </div>""", unsafe_allow_html=True)

                elif modelo == "Chapman-Enskog":
                    t_star = t_abs / eps_k
                    omega = omega_v(t_star)
                    mu_uP = (26.69 * np.sqrt(m_in * t_abs)) / (sigma**2 * omega)
                    mu_c = mu_uP / 10
                    st.markdown(f"<div class='intermedio'><b>T*</b> = {t_star:.4f} | <b>Ωv</b> = {omega:.4f}</div>", unsafe_allow_html=True)

                elif modelo == "Stiel y Thodos":
                    tr = t_abs / tc
                    if tr > 1.5:
                        N = 1.778e-4 * (4.58 * tr - 1.67)**0.625
                    else:
                        N = 3.4e-4 * (tr**0.94)
                    xi = (tc**(1/6)) / (np.sqrt(m_in) * (pc**(2/3)))
                    mu_cP = N / xi
                    mu_c = mu_cP * 1000
                    st.markdown(f"<div class='intermedio'><b>Tr</b> = {tr:.4f} | <b>N (μ·ξ)</b> = {N:.6e} | <b>ξ</b> = {xi:.4f}</div>", unsafe_allow_html=True)

                elif modelo == "DIPPR":
                    mu_Pa_s = (a_dip * t_abs**b_dip) / (1 + c_dip/t_abs + d_dip/t_abs**2)
                    mu_c = mu_Pa_s * 1e6

                # Cálculo de Error Porcentual Individual
                error = abs(v_exp - mu_c) / v_exp * 100 if v_exp > 0 else 0.0
                
                nuevo = pd.DataFrame([{"Componente": comp, "Modelo": modelo, "T (K)": t_abs, "Exp (uPa.s)": v_exp, "Calc (uPa.s)": round(mu_c, 4), "Error (%)": round(error, 3)}])
                st.session_state.tabla_puras = pd.concat([st.session_state.tabla_puras, nuevo], ignore_index=True)
                st.success(f"✅ Cálculo finalizado: {round(mu_c, 4)} μPa·s (Error: {round(error, 2)}%)")
                
            except ZeroDivisionError:
                st.error("Error: División por cero. Verifica que parámetros críticos como Tc, Vc, o Masa Molar no sean 0.")
            except Exception as e:
                st.error(f"Error en los datos: {e}")

        # --- TABLA Y DIAGRAMA DE DISPERSIÓN ---
        if not st.session_state.tabla_puras.empty:
            st.write("---")
            st.subheader("Memoria de Resultados y Validación")
            st.dataframe(st.session_state.tabla_puras, use_container_width=True)
            
            if st.button("Limpiar Memoria"):
                st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])
                st.rerun()

            df_valid = st.session_state.tabla_puras[st.session_state.tabla_puras["Exp (uPa.s)"] > 0]
            
            if len(df_valid) > 0:
                y_exp = df_valid["Exp (uPa.s)"].astype(float).values
                y_calc = df_valid["Calc (uPa.s)"].astype(float).values
                nombres = df_valid["Componente"].values
                
                mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Error Global MAPE", f"{mape:.3f} %")
                
                if len(y_exp) > 1:
                    r2 = r2_score(y_exp, y_calc)
                    col_m2.metric("Coeficiente de Determinación (R²)", f"{r2:.5f}")
                else:
                    col_m2.info("Añade otro componente para calcular R²")

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=y_exp, y=y_calc, 
                    mode='markers+text', 
                    text=nombres, 
                    textposition="top center", 
                    marker=dict(size=10, color='#0056b3'),
                    name="Viscosidad Calculada"
                ))
                val_min = min(min(y_exp), min(y_calc)) * 0.95
                val_max = max(max(y_exp), max(y_calc)) * 1.05
                fig.add_trace(go.Scatter(
                    x=[val_min, val_max], y=[val_min, val_max], 
                    mode='lines', 
                    name='Línea Ideal (Exp = Calc)', 
                    line=dict(color='red', dash='dash')
                ))
                fig.update_layout(
                    title="Diagrama de Dispersión: Viscosidad Experimental vs Calculada",
                    xaxis_title="Viscosidad Experimental (μPa·s)",
                    yaxis_title="Viscosidad Calculada (μPa·s)",
                    template="plotly_white"
                )
                st.plotly_chart(fig, use_container_width=True)

    elif menu == "Reglas de Mezclado":
        st.header("Estimación de Mezclas Multicomponentes (Gas)")
        metodo = st.selectbox("Seleccione el Método:", ["Wilke", "Davidson"])
        
        with st.expander("📖 Ver Ecuación y Variables de la Mezcla", expanded=True):
            if metodo == "Wilke":
                st.latex(r"\mu_m = \sum_{i=1}^n \frac{x_i \mu_i}{\sum_{j=1}^n x_j \Phi_{ij}} \quad \text{donde} \quad \Phi_{ij} = \frac{\left[1 + (\mu_i/\mu_j)^{1/2} (M_j/M_i)^{1/4}\right]^2}{\sqrt{8(1 + M_i/M_j)}}")
                st.markdown("""
                **Variables:**
                * **μ_m**: Viscosidad total de la mezcla gaseosa.
                * **xi, xj**: Fracciones molares de los componentes $i$ y $j$.
                * **μi, μj**: Viscosidades individuales de los componentes $i$ y $j$.
                * **Mi, Mj**: Masas molares de los componentes $i$ y $j$.
                * **Φij**: Parámetro de interacción binaria de Wilke.
                """)
            elif metodo == "Davidson":
                st.latex(r"\mu_m = \sum_{i=1}^n \frac{x_i \mu_i}{\sum_{j=1}^n x_j \Psi_{ij}} \quad \text{donde} \quad \Psi_{ij} = \sqrt{\frac{M_j}{M_i}}")
                st.markdown("""
                **Variables:**
                * **μ_m**: Viscosidad total de la mezcla gaseosa.
                * **xi, xj**: Fracciones molares de los componentes $i$ y $j$.
                * **μi**: Viscosidad individual del componente $i$.
                * **Mi, Mj**: Masas molares de los componentes $i$ y $j$.
                * **Ψij**: Factor de interacción simplificado de Davidson.
                """)
        
        st.write("Ingresa la fracción molar ($x_i$) y la viscosidad individual calculada ($\mu_i$) para cada gas.")
        
        data_init = []
        for c in COMPUESTOS_INFO:
            data_init.append({"Componente": c, "x_i": 0.0, "mu_i (uPa.s)": 0.0, "M_i": COMPUESTOS_INFO[c]["M"]})
        
        df_mezcla = st.data_editor(pd.DataFrame(data_init), num_rows="fixed", use_container_width=True)
        
        if st.button("Calcular Viscosidad de la Mezcla"):
            xi = df_mezcla["x_i"].astype(float).values
            mui = df_mezcla["mu_i (uPa.s)"].astype(float).values
            mi = df_mezcla["M_i"].astype(float).values
            mu_m = 0.0
            n = len(xi)
            
            suma_xi = sum(xi)
            if not np.isclose(suma_xi, 1.0, atol=0.01) and suma_xi > 0:
                st.warning(f"⚠️ Nota: La suma de las fracciones molares es {suma_xi:.4f}. Debería ser 1.0")

            try:
                if metodo == "Wilke":
                    phi = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            if mui[j] > 0 and mi[i] > 0:
                                num = (1 + (mui[i]/mui[j])**0.5 * (mi[j]/mi[i])**0.25)**2
                                den = np.sqrt(8 * (1 + mi[i]/mi[j]))
                                phi[i,j] = num / den
                    
                    for i in range(n):
                        den_sum = sum(xi[j] * phi[i,j] for j in range(n))
                        if den_sum > 0: 
                            mu_m += (xi[i] * mui[i]) / den_sum
                            
                elif metodo == "Davidson":
                    psi = np.zeros((n, n))
                    for i in range(n):
                        for j in range(n):
                            if mi[i] > 0:
                                psi[i,j] = np.sqrt(mi[j] / mi[i])
                    
                    for i in range(n):
                        den_sum = sum(xi[j] * psi[i,j] for j in range(n))
                        if den_sum > 0: 
                            mu_m += (xi[i] * mui[i]) / den_sum

                st.success(f"Viscosidad Total de la Mezcla ({metodo}): {mu_m:.4f} μPa·s")
            except Exception as e:
                st.error(f"Error matemático en las matrices de mezcla. Detalle: {e}")

# =====================================================================
# ======================= MÓDULO DE LÍQUIDOS ==========================
# =====================================================================
elif fase == "Líquido (Nafta Pesada)":
    if menu == "Sustancias Puras":
        st.header("Análisis de Componentes Puros (Líquido)")
        modelo_liq = st.selectbox("Modelo Matemático (Líquidos):", ["Andrade", "Van Velzen et al.", "Orrick y Erbar", "Letsou y Stiel"])

        # --- BLOQUE INFORMATIVO LÍQUIDOS ---
        with st.expander("📖 Ver Ecuación y Variables del Modelo Líquido", expanded=True):
            if modelo_liq == "Andrade":
                st.latex(r"\ln \mu = A + \frac{B}{T}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica del líquido (generalmente en cP o mPa·s).
                * **T**: Temperatura absoluta (K).
                * **A, B**: Constantes empíricas específicas del compuesto.
                """)
            elif modelo_liq == "Van Velzen et al.":
                st.latex(r"\log_{10} \mu = B \left( \frac{1}{T} - \frac{1}{T_0} \right)")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica del líquido (cP).
                * **T**: Temperatura absoluta (K).
                * **B, T_0**: Parámetros correlacionales del componente (o derivados de contribución de grupos).
                """)
            elif modelo_liq == "Orrick y Erbar":
                st.latex(r"\ln \left( \frac{\mu}{\rho M} \right) = A + \frac{B}{T}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica del líquido (cP).
                * **ρ**: Densidad del líquido a la temperatura dada (g/cm³).
                * **M**: Masa molar del componente (g/mol).
                * **T**: Temperatura del sistema (K).
                * **A, B**: Constantes correlacionales del modelo.
                """)
            elif modelo_liq == "Letsou y Stiel":
                st.latex(r"\mu \xi = (\mu \xi)^{(0)} + \omega (\mu \xi)^{(1)}")
                st.latex(r"(\mu \xi)^{(0)} \times 10^3 = 2.648 - 3.725 T_r + 1.309 T_r^2")
                st.latex(r"(\mu \xi)^{(1)} \times 10^3 = 7.425 - 13.39 T_r + 5.933 T_r^2")
                st.latex(r"\xi = \frac{T_c^{1/6}}{M^{1/2} P_c^{2/3}}")
                st.markdown("""
                **Variables:**
                * **μ**: Viscosidad dinámica del líquido (cP). *(Usado para líquidos saturados)*
                * **Tr**: Temperatura reducida ($T / T_c$).
                * **ω**: Factor acéntrico.
                * **Tc, Pc, M**: Propiedades críticas y masa molar.
                """)

        st.subheader("Entrada de Datos")
        cl1, cl2, cl3 = st.columns(3)
        
        with cl1:
            comp_liq = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_LIQ_INFO.keys()))
            t_abs_liq = st.number_input("Temperatura de Operación (K):", value=298.15)
            v_exp_liq = st.number_input("Viscosidad Experimental (cP / mPa·s):", value=0.0000, format="%.4f")
        
        with cl2:
            m_in_liq = st.number_input("Masa Molar (M):", value=COMPUESTOS_LIQ_INFO[comp_liq]["M"], format="%.4f")
            
            if modelo_liq == "Andrade" or modelo_liq == "Orrick y Erbar":
                a_cte = st.number_input("Constante A:", value=0.0000, format="%.4f")
                b_cte = st.number_input("Constante B:", value=0.0000, format="%.4f")
            elif modelo_liq == "Van Velzen et al.":
                b_vv = st.number_input("Constante B:", value=0.0000, format="%.4f")
                t0_vv = st.number_input("Constante T0 (K):", value=0.00, format="%.2f")
            elif modelo_liq == "Letsou y Stiel":
                tc_liq = st.number_input("Temp. Crítica (Tc en K):", value=0.00, format="%.2f")
                pc_liq = st.number_input("Pres. Crítica (Pc en atm):", value=0.00, format="%.2f")
                
        with cl3:
            if modelo_liq == "Orrick y Erbar":
                rho_liq = st.number_input("Densidad del Líquido (ρ en g/cm³):", value=0.0000, format="%.4f")
            elif modelo_liq == "Letsou y Stiel":
                w_liq = st.number_input("Factor Acéntrico (ω):", value=0.0000, format="%.4f")

        if st.button("Ejecutar Cálculo Líquido"):
            try:
                mu_c_liq = 0.0
                
                if modelo_liq == "Andrade":
                    mu_c_liq = np.exp(a_cte + (b_cte / t_abs_liq))
                    st.markdown(f"<div class='intermedio-liq'><b>Cálculo Andrade:</b> μ = exp({a_cte} + {b_cte}/{t_abs_liq}) = <b>{mu_c_liq:.4f} cP</b></div>", unsafe_allow_html=True)
                    
                elif modelo_liq == "Van Velzen et al.":
                    if t0_vv == 0: st.error("T0 no puede ser cero.")
                    else:
                        exponente = b_vv * ((1/t_abs_liq) - (1/t0_vv))
                        mu_c_liq = 10**(exponente)
                        st.markdown(f"<div class='intermedio-liq'><b>Cálculo Van Velzen:</b> log10(μ) = {exponente:.4f} → μ = <b>{mu_c_liq:.4f} cP</b></div>", unsafe_allow_html=True)
                
                elif modelo_liq == "Orrick y Erbar":
                    if rho_liq == 0: st.error("La densidad no puede ser cero.")
                    else:
                        termino = np.exp(a_cte + (b_cte / t_abs_liq))
                        mu_c_liq = rho_liq * m_in_liq * termino
                        st.markdown(f"<div class='intermedio-liq'><b>Cálculo Orrick-Erbar:</b> μ = ({rho_liq} * {m_in_liq}) * exp({a_cte} + {b_cte}/{t_abs_liq}) = <b>{mu_c_liq:.4f} cP</b></div>", unsafe_allow_html=True)
                
                elif modelo_liq == "Letsou y Stiel":
                    if tc_liq == 0 or pc_liq == 0: st.error("Tc y Pc deben ser mayores a cero.")
                    else:
                        tr_liq = t_abs_liq / tc_liq
                        xi_liq = (tc_liq**(1/6)) / (np.sqrt(m_in_liq) * (pc_liq**(2/3)))
                        
                        mu_xi_0 = (2.648 - 3.725*tr_liq + 1.309*(tr_liq**2)) / 1000
                        mu_xi_1 = (7.425 - 13.39*tr_liq + 5.933*(tr_liq**2)) / 1000
                        
                        mu_xi_total = mu_xi_0 + w_liq * mu_xi_1
                        mu_c_liq = mu_xi_total / xi_liq
                        
                        st.markdown(f"""<div class='intermedio-liq'>
                        <b>🔢 Auditoría de Ecuaciones (Letsou y Stiel):</b><br><br>
                        Tr = {t_abs_liq} / {tc_liq} = <b>{tr_liq:.4f}</b><br>
                        ξ = <b>{xi_liq:.5f}</b><br>
                        (μ·ξ)^(0) = <b>{mu_xi_0:.6f}</b> | (μ·ξ)^(1) = <b>{mu_xi_1:.6f}</b><br>
                        μ = ({mu_xi_0:.6f} + {w_liq}*{mu_xi_1:.6f}) / {xi_liq:.5f} = <b>{mu_c_liq:.4f} cP</b>
                        </div>""", unsafe_allow_html=True)

                # Cálculo de Error Porcentual Individual
                error_liq = abs(v_exp_liq - mu_c_liq) / v_exp_liq * 100 if v_exp_liq > 0 else 0.0
                
                nuevo_liq = pd.DataFrame([{"Componente": comp_liq, "Modelo": modelo_liq, "T (K)": t_abs_liq, "Exp (cP)": v_exp_liq, "Calc (cP)": round(mu_c_liq, 4), "Error (%)": round(error_liq, 3)}])
                st.session_state.tabla_puras_liq = pd.concat([st.session_state.tabla_puras_liq, nuevo_liq], ignore_index=True)
                st.success(f"✅ Cálculo finalizado: {round(mu_c_liq, 4)} cP (Error: {round(error_liq, 2)}%)")
                
            except Exception as e:
                st.error(f"Error matemático en el cálculo: {e}")

        # --- TABLA Y DIAGRAMA DE DISPERSIÓN (LÍQUIDOS) ---
        if not st.session_state.tabla_puras_liq.empty:
            st.write("---")
            st.subheader("Memoria de Resultados y Validación (Líquidos)")
            st.dataframe(st.session_state.tabla_puras_liq, use_container_width=True)
            
            if st.button("Limpiar Memoria Líquida"):
                st.session_state.tabla_puras_liq = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (cP)", "Calc (cP)", "Error (%)"])
                st.rerun()

            df_valid_liq = st.session_state.tabla_puras_liq[st.session_state.tabla_puras_liq["Exp (cP)"] > 0]
            
            if len(df_valid_liq) > 0:
                y_exp_l = df_valid_liq["Exp (cP)"].astype(float).values
                y_calc_l = df_valid_liq["Calc (cP)"].astype(float).values
                nombres_l = df_valid_liq["Componente"].values
                
                mape_l = np.mean(np.abs((y_exp_l - y_calc_l) / y_exp_l)) * 100
                
                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Error Global MAPE", f"{mape_l:.3f} %")
                
                if len(y_exp_l) > 1:
                    r2_l = r2_score(y_exp_l, y_calc_l)
                    col_m2.metric("Coeficiente de Determinación (R²)", f"{r2_l:.5f}")
                else:
                    col_m2.info("Añade otro componente líquido para calcular R²")

                fig_l = go.Figure()
                fig_l.add_trace(go.Scatter(
                    x=y_exp_l, y=y_calc_l, 
                    mode='markers+text', 
                    text=nombres_l, 
                    textposition="top center", 
                    marker=dict(size=10, color='#e0a800'),
                    name="Viscosidad Calculada"
                ))
                val_min_l = min(min(y_exp_l), min(y_calc_l)) * 0.95
                val_max_l = max(max(y_exp_l), max(y_calc_l)) * 1.05
                fig_l.add_trace(go.Scatter(
                    x=[val_min_l, val_max_l], y=[val_min_l, val_max_l], 
                    mode='lines', 
                    name='Línea Ideal (Exp = Calc)', 
                    line=dict(color='red', dash='dash')
                ))
                fig_l.update_layout(
                    title="Diagrama de Dispersión: Viscosidad Líquida (Nafta)",
                    xaxis_title="Viscosidad Experimental (cP)",
                    yaxis_title="Viscosidad Calculada (cP)",
                    template="plotly_white"
                )
                st.plotly_chart(fig_l, use_container_width=True)

    elif menu == "Reglas de Mezclado":
        st.header("Estimación de Mezclas Multicomponentes (Líquido)")
        metodo_liq = st.selectbox("Seleccione el Método:", ["Kendal y Monroe", "Grunberg y Nissan"])
        
        with st.expander("📖 Ver Ecuación de la Mezcla Líquida", expanded=True):
            if metodo_liq == "Kendal y Monroe":
                st.latex(r"\mu_m = \left( \sum_{i=1}^n x_i \mu_i^{1/3} \right)^3")
                st.markdown("""
                **Variables:**
                * **μ_m**: Viscosidad total de la mezcla líquida.
                * **x_i**: Fracción molar del componente *i*.
                * **μ_i**: Viscosidad del componente puro *i*.
                """)
            elif metodo_liq == "Grunberg y Nissan":
                st.latex(r"\ln \mu_m = \sum_{i=1}^n x_i \ln \mu_i + \sum_{i \neq j} x_i x_j G_{ij}")
                st.markdown("""
                **Variables:**
                * **μ_m**: Viscosidad total de la mezcla líquida.
                * **x_i**: Fracción molar del componente *i*.
                * **μ_i**: Viscosidad del componente puro *i*.
                * **G_ij**: Parámetro de interacción (para fines prácticos e ideales, se asume $G_{ij} = 0$, reduciéndose a la regla de Arrhenius).
                """)
        
        st.write("Ingresa la fracción molar ($x_i$) y la viscosidad individual calculada ($\mu_i$) para cada componente de la Nafta.")
        
        data_init_liq = []
        for c in COMPUESTOS_LIQ_INFO:
            data_init_liq.append({"Componente": c, "x_i": 0.0, "mu_i (cP)": 0.0})
        
        df_mezcla_liq = st.data_editor(pd.DataFrame(data_init_liq), num_rows="fixed", use_container_width=True)
        
        if st.button("Calcular Viscosidad de la Mezcla Líquida"):
            xi_l = df_mezcla_liq["x_i"].astype(float).values
            mui_l = df_mezcla_liq["mu_i (cP)"].astype(float).values
            mu_m_l = 0.0
            
            suma_xi_l = sum(xi_l)
            if not np.isclose(suma_xi_l, 1.0, atol=0.01) and suma_xi_l > 0:
                st.warning(f"⚠️ Nota: La suma de las fracciones molares es {suma_xi_l:.4f}. Debería ser 1.0")

            try:
                if metodo_liq == "Kendal y Monroe":
                    suma_kendal = 0.0
                    for i in range(len(xi_l)):
                        if mui_l[i] > 0 and xi_l[i] > 0:
                            suma_kendal += xi_l[i] * (mui_l[i]**(1/3))
                    mu_m_l = suma_kendal**3
                            
                elif metodo_liq == "Grunberg y Nissan":
                    suma_ln_mu = 0.0
                    for i in range(len(xi_l)):
                        if mui_l[i] > 0 and xi_l[i] > 0:
                            suma_ln_mu += xi_l[i] * np.log(mui_l[i])
                    # Asumiendo mezcla ideal (Gij = 0)
                    mu_m_l = np.exp(suma_ln_mu)

                st.success(f"Viscosidad Total de la Mezcla Líquida ({metodo_liq}): {mu_m_l:.4f} cP")
            except Exception as e:
                st.error(f"Error matemático al evaluar la regla de mezclado. Detalle: {e}")
