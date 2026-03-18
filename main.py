import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

# Configuración de página profesional
st.set_page_config(page_title="Software de Viscosidad - FTIQ", layout="wide")

# Estilo CSS para mejorar la apariencia de ingeniería
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #d1d8e0; }
    </style>
    """, unsafe_allow_html=True)

# --- BASE DE DATOS DE REFERENCIA (Fracciones y Masas del Cracking de Etano) ---
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

# Inicialización de estados para no perder datos al recargar
if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])

# --- FUNCIONES MATEMÁTICAS ---
def omega_v(T_star):
    return 1.16145*(T_star**-0.14874) + 0.52487*np.exp(-0.77320*T_star) + 2.16178*np.exp(-2.43787*T_star)

# --- INTERFAZ ---
st.title("Determinación de Viscosidad - Métodos Moleculares")
st.sidebar.header("Configuración del Sistema")
fase = st.sidebar.selectbox("Fase de la Mezcla:", ["Mezcla Gaseosa (Cracking)", "Mezcla Líquida (Nafta)"])

menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

if menu == "Sustancias Puras":
    st.header("Análisis de Componentes Puros")
    modelo = st.selectbox("Modelo Matemático:", ["Chapman-Enskog", "Chung et al.", "Stiel y Thodos", "DIPPR"])

    # Explicación de variables según el modelo
    with st.expander("Ver Ecuación y Significado de Variables"):
        if modelo == "Chapman-Enskog":
            st.latex(r"\mu = 26.69 \frac{\sqrt{M T}}{\sigma^2 \Omega_v}")
            st.write("M: g/mol, T: K, σ: Å, Ωv: Integral de colisión.")
        elif modelo == "Chung et al.":
            st.latex(r"\mu = \frac{40.785 F_c \sqrt{MT}}{V_c^{2/3} \Omega_v}")
            st.write("Fc: Factor de corrección (incluye ω y μ dipolar).")
        elif modelo == "DIPPR":
            st.latex(r"\mu = \frac{A T^B}{1 + C/T + D/T^2}")
            st.write("A, B, C, D: Parámetros de ajuste específicos.")

    st.subheader("Entrada de Datos")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
        t_abs = st.number_input("Temperatura (K):", value=298.15)
        v_exp = st.number_input("Viscosidad Exp (uPa.s):", format="%.4f")
    
    with c2:
        m_in = st.number_input("Masa Molar (M):", value=COMPUESTOS_INFO[comp]["M"])
        if modelo == "Chapman-Enskog":
            sigma = st.number_input("Sigma (σ):", format="%.4f")
            eps_k = st.number_input("Epsilon/k (ε/κ):", format="%.2f")
        elif modelo == "Chung et al.":
            tc = st.number_input("Temp. Crítica (Tc):", format="%.2f")
            vc = st.number_input("Vol. Crítico (Vc):", format="%.2f")
            w = st.number_input("Factor Acéntrico (ω):", format="%.4f")
        elif modelo == "DIPPR":
            a_dip = st.number_input("Coeficiente A:", format="%.4e")
            b_dip = st.number_input("Coeficiente B:", format="%.4f")
    
    with c3:
        if modelo == "Chung et al.":
            dipolo = st.number_input("Momento Dipolar (D):", value=0.0)
        elif modelo == "DIPPR":
            c_dip = st.number_input("Coeficiente C:", format="%.4f")
            d_dip = st.number_input("Coeficiente D:", format="%.4f")
        elif modelo == "Stiel y Thodos":
            tc = st.number_input("Temp. Crítica (Tc):", format="%.2f")
            pc = st.number_input("Pres. Crítica (Pc):", format="%.2f")

    if st.button("Ejecutar Cálculo"):
        # Lógica de cálculo con conversiones
        try:
            if modelo == "Chapman-Enskog":
                t_star = t_abs / eps_k
                mu_c = (26.69 * np.sqrt(m_in * t_abs)) / (sigma**2 * omega_v(t_star)) / 10
            elif modelo == "Chung et al.":
                tr = t_abs / tc
                mu_r = 131.3 * dipolo / (vc * tc)**0.5
                fc = 1 - 0.2756 * w + 0.059035 * mu_r**4
                mu_c = (40.785 * fc * np.sqrt(m_in * t_abs)) / (vc**(2/3) * omega_v(tr)) / 10
            elif modelo == "DIPPR":
                mu_c = ((a_dip * t_abs**b_dip) / (1 + c_dip/t_abs + d_dip/t_abs**2)) * 1e6
            elif modelo == "Stiel y Thodos":
                tr = t_abs / tc
                nv = 1.778*(4.58*tr - 1.67)**0.625 if tr > 1.5 else 3.4*(tr**0.94)
                mu_c = (9.91e-8 * nv * np.sqrt(m_in) * (pc**(2/3)) / (tc**(1/6))) * 1e6

            error = abs(v_exp - mu_c) / v_exp * 100 if v_exp > 0 else 0
            
            # Guardar en sesión
            nuevo = pd.DataFrame([{"Componente": comp, "Modelo": modelo, "T (K)": t_abs, "Exp (uPa.s)": v_exp, "Calc (uPa.s)": round(mu_c, 4), "Error (%)": round(error, 3)}])
            st.session_state.tabla_puras = pd.concat([st.session_state.tabla_puras, nuevo], ignore_index=True)
            st.success(f"Cálculo para {comp} exitoso.")
        except Exception as e:
            st.error(f"Error técnico: {e}")

    # Visualización de Resultados
    if not st.session_state.tabla_puras.empty:
        st.subheader("Resultados y Validación")
        st.table(st.session_state.tabla_puras)
        
        # Estadísticas Globales
        y_exp = st.session_state.tabla_puras["Exp (uPa.s)"].values
        y_calc = st.session_state.tabla_puras["Calc (uPa.s)"].values
        
        if len(y_exp) > 0:
            mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            st.metric("Error Global MAPE", f"{mape:.3f} %")
            
            if len(y_exp) > 1:
                r2 = r2_score(y_exp, y_calc)
                st.metric("Coeficiente de Determinación R²", f"{r2:.5f}")

            # Gráfica de Dispersión
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_exp, y=y_calc, mode='markers+text', text=st.session_state.tabla_puras["Componente"], textposition="top center", name="Datos"))
            lims = [min(y_exp)*0.9, max(y_exp)*1.1]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='Ideal (x=y)', line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title="Experimental", yaxis_title="Calculado")
            st.plotly_chart(fig)

elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Multicomponentes")
    metodo = st.selectbox("Seleccione el Método:", ["Wilke", "Davidson"])
    
    st.info("Ingrese los datos de cada componente (fracción molar xi y viscosidad pura calculada).")
    
    # Crear tabla dinámica para los 8 componentes
    data_init = []
    for c in COMPUESTOS_INFO:
        data_init.append({"Componente": c, "xi": 0.0, "mu_i (uPa.s)": 0.0, "M_i": COMPUESTOS_INFO[c]["M"]})
    
    df_mezcla = st.data_editor(pd.DataFrame(data_init), num_rows="fixed")
    
    if st.button("Calcular Viscosidad de Mezcla"):
        xi = df_mezcla["xi"].values
        mui = df_mezcla["mu_i (uPa.s)"].values
        mi = df_mezcla["M_i"].values
        
        if not np.isclose(sum(xi), 1.0, atol=0.01):
            st.warning(f"Atención: La suma de xi es {sum(xi):.3f}")

        mu_m = 0
        n = len(xi)
        
        if metodo == "Wilke":
            phi = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    num = (1 + (mui[i]/mui[j])**0.5 * (mi[j]/mi[i])**0.25)**2
                    den = (8 * (1 + mi[i]/mi[j]))**0.5
                    phi[i,j] = num / den
            
            for i in range(n):
                den_sum = sum(xi[j] * phi[i,j] for j in range(n))
                mu_m += (xi[i] * mui[i]) / den_sum
                
        elif metodo == "Davidson":
            # Psi_ij = sqrt(Mj / Mi) según Tabla 10 del Word
            psi = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    psi[i,j] = np.sqrt(mi[j]/mi[i])
            
            for i in range(n):
                den_sum = sum(xi[j] * psi[i,j] for j in range(n))
                mu_m += (xi[i] * mui[i]) / den_sum

        st.success(f"Viscosidad de la Mezcla ({metodo}): {mu_m:.4f} μPa·s")
