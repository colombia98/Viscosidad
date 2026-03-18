import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="Software de Viscosidad - FTIQ", layout="wide")

st.markdown("""
    <style>
    .stAlert { border-radius: 10px; }
    .intermedio { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0056b3; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)

# --- BASE DE DATOS DE REFERENCIA ---
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
st.info("💡 **Nota sobre decimales:** Dependiendo del idioma de tu navegador, Streamlit puede pedirte usar coma (,) o punto (.) para los decimales. Si uno te da error, intenta con el otro en las cajas de texto.")

st.sidebar.header("Configuración")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

if menu == "Sustancias Puras":
    st.header("Análisis de Componentes Puros")
    modelo = st.selectbox("Modelo Matemático:", ["Chapman-Enskog", "Chung et al.", "Stiel y Thodos", "DIPPR"])

    with st.expander("Ver Ecuación y Significado de Variables"):
        if modelo == "Chapman-Enskog":
            st.latex(r"\mu = 26.69 \frac{\sqrt{M T}}{\sigma^2 \Omega_v}")
        elif modelo == "Chung et al.":
            st.latex(r"\mu = \frac{40.785 F_c \sqrt{MT}}{V_c^{2/3} \Omega_v}")
            st.latex(r"\mu_r = 131.3 \frac{\mu_D}{\sqrt{V_c T_c}} \quad ; \quad F_c = 1 - 0.2756\omega + 0.059035\mu_r^4 + \kappa")
            st.markdown("*(Nota: Para los hidrocarburos del cracking, el momento dipolar $\mu_D$ y $\kappa$ suelen ser 0)*")
        elif modelo == "DIPPR":
            st.latex(r"\mu = \frac{A T^B}{1 + C/T + D/T^2}")
        elif modelo == "Stiel y Thodos":
            st.latex(r"\xi = \frac{T_c^{1/6}}{M^{1/2} P_c^{2/3}} \quad ; \quad \mu = \frac{N}{\xi}")

    st.subheader("Entrada de Datos")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        comp = st.selectbox("Seleccione Compuesto:", list(COMPUESTOS_INFO.keys()))
        t_abs = st.number_input("Temperatura (K):", value=298.15, format="%.2f")
        v_exp = st.number_input("Viscosidad Exp (uPa.s):", value=0.0000, format="%.4f")
    
    with c2:
        m_in = st.number_input("Masa Molar (M):", value=COMPUESTOS_INFO[comp]["M"], format="%.4f")
        if modelo == "Chapman-Enskog":
            sigma = st.number_input("Sigma (σ en Å):", value=0.0000, format="%.4f")
            eps_k = st.number_input("Epsilon/k (ε/κ en K):", value=0.00, format="%.2f")
        elif modelo == "Chung et al.":
            tc = st.number_input("Temp. Crítica (Tc en K):", value=0.00, format="%.2f")
            vc = st.number_input("Vol. Crítico (Vc en cm³/mol):", value=0.00, format="%.2f")
            w = st.number_input("Factor Acéntrico (ω):", value=0.0000, format="%.4f")
        elif modelo == "DIPPR":
            a_dip = st.number_input("Coeficiente A:", format="%.4e")
            b_dip = st.number_input("Coeficiente B:", format="%.4f")
        elif modelo == "Stiel y Thodos":
            tc = st.number_input("Temp. Crítica (Tc en K):", value=0.00, format="%.2f")
            pc = st.number_input("Pres. Crítica (Pc en atm):", value=0.00, format="%.2f")
    
    with c3:
        if modelo == "Chung et al.":
            dipolo = st.number_input("Momento Dipolar (D):", value=0.0000, format="%.4f")
            kappa = st.number_input("Factor Asociación (κ):", value=0.0000, format="%.4f")
        elif modelo == "DIPPR":
            c_dip = st.number_input("Coeficiente C:", format="%.4f")
            d_dip = st.number_input("Coeficiente D:", format="%.4f")

    if st.button("Ejecutar Cálculo"):
        try:
            # -----------------------------------------------------
            # BLOQUE DE CÁLCULOS RIGUROSOS Y TRANSPARENTES
            # -----------------------------------------------------
            if modelo == "Chapman-Enskog":
                t_star = t_abs / eps_k
                omega = omega_v(t_star)
                mu_uP = (26.69 * np.sqrt(m_in * t_abs)) / (sigma**2 * omega)
                mu_c = mu_uP / 10
                
                # Desglose de variables
                st.markdown(f"""<div class='intermedio'>
                <b>Valores Intermedios Calculados:</b><br>
                T* = {t_star:.4f}<br>
                Ωv = {omega:.4f}
                </div>""", unsafe_allow_html=True)

            elif modelo == "Chung et al.":
                tr = t_abs / tc
                t_star_chung = 1.2593 * tr
                omega = omega_v(t_star_chung)
                
                # Fórmulas idénticas a las imágenes que enviaste
                mu_r = 131.3 * dipolo / np.sqrt(vc * tc)
                fc = 1 - 0.2756 * w + 0.059035 * (mu_r**4) + kappa
                
                mu_uP = (40.785 * fc * np.sqrt(m_in * t_abs)) / (vc**(2/3) * omega)
                mu_c = mu_uP / 10
                
                # Desglose de variables para auditoría
                st.markdown(f"""<div class='intermedio'>
                <b>Valores Intermedios Calculados (Chung et al.):</b><br>
                Tr = {tr:.4f}<br>
                T* (1.2593 * Tr) = {t_star_chung:.4f}<br>
                Ωv = {omega:.4f}<br>
                μ_r (Momento dipolar reducido) = {mu_r:.6f}<br>
                Fc (Factor de corrección) = {fc:.6f}
                </div>""", unsafe_allow_html=True)

            elif modelo == "DIPPR":
                mu_Pa_s = (a_dip * t_abs**b_dip) / (1 + c_dip/t_abs + d_dip/t_abs**2)
                mu_c = mu_Pa_s * 1e6
                
            elif modelo == "Stiel y Thodos":
                tr = t_abs / tc
                if tr > 1.5:
                    N = 1.778e-4 * (4.58 * tr - 1.67)**0.625
                else:
                    N = 3.4e-4 * (tr**0.94)
                    
                xi = (tc**(1/6)) / (np.sqrt(m_in) * (pc**(2/3)))
                mu_cP = N / xi
                mu_c = mu_cP * 1000
                
                # Desglose de variables
                st.markdown(f"""<div class='intermedio'>
                <b>Valores Intermedios Calculados:</b><br>
                Tr = {tr:.4f}<br>
                N = {N:.6e}<br>
                ξ = {xi:.4f}
                </div>""", unsafe_allow_html=True)

            # Cálculo del Error
            error = abs(v_exp - mu_c) / v_exp * 100 if v_exp > 0 else 0
            
            # Actualización de la Tabla
            nuevo = pd.DataFrame([{"Componente": comp, "Modelo": modelo, "T (K)": t_abs, "Exp (uPa.s)": v_exp, "Calc (uPa.s)": round(mu_c, 4), "Error (%)": round(error, 3)}])
            st.session_state.tabla_puras = pd.concat([st.session_state.tabla_puras, nuevo], ignore_index=True)
            st.success(f"Cálculo completado: {round(mu_c, 4)} μPa·s")
            
        except Exception as e:
            st.error(f"Error en los datos ingresados. Asegúrate de no dejar variables críticas en 0. Detalle: {e}")

    # Visualización de Resultados y Gráficas
    if not st.session_state.tabla_puras.empty:
        st.subheader("Resultados y Validación")
        st.table(st.session_state.tabla_puras)
        
        if st.button("Limpiar Tabla"):
            st.session_state.tabla_puras = pd.DataFrame(columns=["Componente", "Modelo", "T (K)", "Exp (uPa.s)", "Calc (uPa.s)", "Error (%)"])
            st.rerun()
            
        y_exp = st.session_state.tabla_puras["Exp (uPa.s)"].values
        y_calc = st.session_state.tabla_puras["Calc (uPa.s)"].values
        
        # Solo graficar si los valores experimentales son válidos (>0)
        y_exp_validos = [y for y in y_exp if y > 0]
        
        if len(y_exp_validos) > 0:
            mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            st.metric("Error Global MAPE", f"{mape:.3f} %")
            
            if len(y_exp_validos) > 1:
                r2 = r2_score(y_exp, y_calc)
                st.metric("Coeficiente R²", f"{r2:.5f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_exp, y=y_calc, mode='markers+text', text=st.session_state.tabla_puras["Componente"], textposition="top center", name="Datos"))
            lims = [min(min(y_exp), min(y_calc))*0.9, max(max(y_exp), max(y_calc))*1.1]
            fig.add_trace(go.Scatter(x=lims, y=lims, mode='lines', name='Línea de Identidad', line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title="Viscosidad Experimental (μPa·s)", yaxis_title="Viscosidad Calculada (μPa·s)")
            st.plotly_chart(fig)

elif menu == "Reglas de Mezclado":
    st.header("Estimación de Mezclas Multicomponentes")
    metodo = st.selectbox("Seleccione el Método:", ["Wilke", "Davidson"])
    
    st.info("Ingresa la fracción molar (xi) y la viscosidad calculada (μi) de cada componente.")
    
    data_init = []
    for c in COMPUESTOS_INFO:
        data_init.append({"Componente": c, "xi": 0.0, "mu_i (uPa.s)": 0.0, "M_i": COMPUESTOS_INFO[c]["M"]})
    
    df_mezcla = st.data_editor(pd.DataFrame(data_init), num_rows="fixed")
    
    if st.button("Calcular Viscosidad de Mezcla"):
        xi = df_mezcla["xi"].values
        mui = df_mezcla["mu_i (uPa.s)"].values
        mi = df_mezcla["M_i"].values
        
        suma_xi = sum(xi)
        if not np.isclose(suma_xi, 1.0, atol=0.01):
            st.warning(f"Atención: La suma de xi es {suma_xi:.4f}. Debería ser 1.0.")

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
                    if den_sum > 0:
                        mu_m += (xi[i] * mui[i]) / den_sum
                    
            elif metodo == "Davidson":
                psi = np.zeros((n, n))
                for i in range(n):
                    for j in range(n):
                        if mi[i] > 0:
                            psi[i,j] = np.sqrt(mi[j]/mi[i])
                
                for i in range(n):
                    den_sum = sum(xi[j] * psi[i,j] for j in range(n))
                    if den_sum > 0:
                        mu_m += (xi[i] * mui[i]) / den_sum

            st.success(f"Viscosidad Calculada de la Mezcla ({metodo}): {mu_m:.4f} μPa·s")
        except Exception as e:
            st.error(f"Error en el cálculo matricial. Detalle: {e}")
