import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="Software de Viscosidad - FTIQ", layout="wide")

st.markdown("""
    <style>
    .stAlert { border-radius: 10px; }
    .intermedio { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0056b3; font-family: monospace; font-size: 14px;}
    </style>
    """, unsafe_allow_html=True)

# --- BASE DE DATOS ---
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

if 'tabla_puras' not in st.session_state:
    st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])

# --- FUNCIONES MATEMÁTICAS ---
def omega_v(T_star):
    return 1.16145*(T_star**-0.14874) + 0.52487*np.exp(-0.77320*T_star) + 2.16178*np.exp(-2.43787*T_star)

# --- INTERFAZ ---
st.title("Determinación de Viscosidad - Métodos Moleculares")
st.error("🔴 **RECORDATORIO:** Usa siempre el PUNTO (.) para los decimales. Ejemplo: 190.56")

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

if menu == "Sustancias Puras":
    st.header("Análisis de Componentes Puros")
    modelo = st.selectbox("Modelo Matemático:", ["Chung et al.", "Chapman-Enskog", "Stiel y Thodos", "DIPPR"])

    st.subheader("Entrada de Datos")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
        t_abs = st.number_input("Temperatura de Operación (K):", value=298.15, format="%.2f")
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
            dipolo = st.number_input("Momento Dipolar (μ en Debyes):", value=0.0000, format="%.4f", help="Para hidrocarburos suele ser 0.")
            # Eliminamos el factor de asociación Kappa como solicitaste.
        elif modelo == "DIPPR":
            c_dip = st.number_input("Coeficiente C:", format="%.4f")
            d_dip = st.number_input("Coeficiente D:", format="%.4f")

    if st.button("Ejecutar Cálculo"):
        try:
            if modelo == "Chung et al.":
                # 1. Temperatura Reducida y T*
                tr = t_abs / tc
                t_star_chung = 1.2593 * tr
                
                # 2. Integral de Colisión
                omega = omega_v(t_star_chung)
                
                # 3. Momento Dipolar Reducido y Factor Fc (SIN KAPPA)
                mu_r = 131.3 * dipolo / np.sqrt(vc * tc)
                fc = 1 - 0.2756 * w + 0.059035 * (mu_r**4)
                
                # 4. Viscosidad (Fórmula de la imagen)
                mu_uP = (40.785 * fc * np.sqrt(m_in * t_abs)) / (vc**(2/3) * omega)
                mu_c = mu_uP / 10 # De Micropoise a uPa.s
                
                # MOSTRAR SUSTITUCIÓN EXACTA PARA AUDITORÍA
                st.markdown(f"""<div class='intermedio'>
                <b>🔢 Auditoría de Ecuaciones (Chung et al.):</b><br><br>
                1. <b>T*</b> = 1.2593 * ({t_abs} / {tc}) = <b>{t_star_chung:.4f}</b><br>
                2. <b>Ωv</b> (Integral de colisión) = <b>{omega:.4f}</b><br>
                3. <b>μ_r</b> = 131.3 * {dipolo} / √({vc} * {tc}) = <b>{mu_r:.5f}</b><br>
                4. <b>Fc</b> = 1 - 0.2756({w}) + 0.059035({mu_r:.4f})^4 = <b>{fc:.5f}</b><br>
                5. <b>μ</b> = [40.785 * {fc:.4f} * √({m_in} * {t_abs})] / [{vc}^(2/3) * {omega:.4f}] = <b>{mu_uP:.2f} μP</b><br>
                6. <b>Conversión</b>: {mu_uP:.2f} / 10 = <b>{mu_c:.4f} μPa·s</b>
                </div>""", unsafe_allow_html=True)

            elif modelo == "Chapman-Enskog":
                t_star = t_abs / eps_k
                omega = omega_v(t_star)
                mu_uP = (26.69 * np.sqrt(m_in * t_abs)) / (sigma**2 * omega)
                mu_c = mu_uP / 10
                st.markdown(f"<div class='intermedio'>T* = {t_star:.4f} | Ωv = {omega:.4f}</div>", unsafe_allow_html=True)

            elif modelo == "Stiel y Thodos":
                tr = t_abs / tc
                if tr > 1.5:
                    N = 1.778e-4 * (4.58 * tr - 1.67)**0.625
                else:
                    N = 3.4e-4 * (tr**0.94)
                xi = (tc**(1/6)) / (np.sqrt(m_in) * (pc**(2/3)))
                mu_cP = N / xi
                mu_c = mu_cP * 1000
                st.markdown(f"<div class='intermedio'>Tr = {tr:.4f} | N = {N:.6e} | ξ = {xi:.4f}</div>", unsafe_allow_html=True)

            elif modelo == "DIPPR":
                mu_Pa_s = (a_dip * t_abs**b_dip) / (1 + c_dip/t_abs + d_dip/t_abs**2)
                mu_c = mu_Pa_s * 1e6

            error = abs(v_exp - mu_c) / v_exp * 100 if v_exp > 0 else 0
            nuevo = pd.DataFrame([{"Componente": comp, "Modelo": modelo, "T (K)": t_abs, "Exp (uPa.s)": v_exp, "Calc (uPa.s)": round(mu_c, 4), "Error (%)": round(error, 3)}])
            st.session_state.tabla_puras = pd.concat([st.session_state.tabla_puras, nuevo], ignore_index=True)
            
            st.success(f"✅ Cálculo finalizado: {round(mu_c, 4)} μPa·s")
            
        except ZeroDivisionError:
            st.error("Error: Hay un valor en 0 dividiendo la ecuación. Verifica Tc, Vc o Masa Molar.")
        except Exception as e:
            st.error(f"Error técnico: {e}")

    # Visualización de la Tabla
    if not st.session_state.tabla_puras.empty:
        st.subheader("Memoria de Resultados")
        st.table(st.session_state.tabla_puras)
        if st.button("Limpiar Memoria"):
            st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])
            st.rerun()

elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Multicomponentes")
    metodo = st.selectbox("Seleccione el Método:", ["Wilke", "Davidson"])
    
    data_init = []
    for c in COMPUESTOS_INFO:
        data_init.append({"Componente": c, "xi": 0.0, "mu_i (uPa.s)": 0.0, "M_i": COMPUESTOS_INFO[c]["M"]})
    
    df_mezcla = st.data_editor(pd.DataFrame(data_init), num_rows="fixed")
    
    if st.button("Calcular Mezcla"):
        xi = df_mezcla["xi"].values
        mui = df_mezcla["mu_i (uPa.s)"].values
        mi = df_mezcla["M_i"].values
        mu_m = 0
        n = len(xi)
        
        try:
            if metodo == "Wilke":
                phi = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if mui[j] > 0 and mi[i] > 0:
                            num = (1 + (mui[i]/mui[j])**0.5 * (mi[j]/mi[i])**0.25)**2
                            den = (8 * (1 + mi[i]/mi[j]))**0.5
                            phi[i,j] = num / den
                for i in range(n):
                    den_sum = sum(xi[j] * phi[i,j] for j in range(n))
                    if den_sum > 0: mu_m += (xi[i] * mui[i]) / den_sum
                    
            elif metodo == "Davidson":
                psi = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if mi[i] > 0:
                            psi[i,j] = np.sqrt(mi[j]/mi[i])
                for i in range(n):
                    den_sum = sum(xi[j] * psi[i,j] for j in range(n))
                    if den_sum > 0: mu_m += (xi[i] * mui[i]) / den_sum

            st.success(f"Viscosidad de la Mezcla ({metodo}): {mu_m:.4f} μPa·s")
        except Exception as e:
            st.error(f"Error en matriz: {e}")
