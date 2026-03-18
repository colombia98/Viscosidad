import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import r2_score

st.set_page_config(page_title="Software de Viscosidad - FTIQ", layout="wide")

st.markdown("""
    <style>
    .intermedio { background-color: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #0056b3; font-family: monospace; font-size: 14px;}
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
st.title("Determinación de Viscosidad - FTIQ")
st.info("💡 **Nota sobre decimales:** Dependiendo de tu sistema operativo y navegador, el programa puede requerir coma (,) o punto (.) para los decimales. Te sugerimos ensayar con ambas y utilizar la que no te marque error.")

st.sidebar.header("Navegación")
menu = st.sidebar.radio("Módulo de Trabajo:", ["Sustancias Puras", "Reglas de Mezclado"])

if menu == "Sustancias Puras":
    st.header("Análisis de Componentes Puros")
    modelo = st.selectbox("Modelo Matemático:", ["Chung et al.", "Chapman-Enskog", "Stiel y Thodos", "DIPPR"])

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
                st.markdown(f"<div class='intermedio'><b>Tr</b> = {tr:.4f} | <b>N</b> = {N:.6e} | <b>ξ</b> = {xi:.4f}</div>", unsafe_allow_html=True)

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

        # Filtrar solo los datos que tengan viscosidad experimental mayor a 0 para las métricas
        df_valid = st.session_state.tabla_puras[st.session_state.tabla_puras["Exp (uPa.s)"] > 0]
        
        if len(df_valid) > 0:
            y_exp = df_valid["Exp (uPa.s)"].astype(float).values
            y_calc = df_valid["Calc (uPa.s)"].astype(float).values
            nombres = df_valid["Componente"].values
            
            # Cálculo de Error Global MAPE
            mape = np.mean(np.abs((y_exp - y_calc) / y_exp)) * 100
            
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Error Global MAPE", f"{mape:.3f} %")
            
            # Cálculo de R^2 (requiere al menos 2 datos)
            if len(y_exp) > 1:
                r2 = r2_score(y_exp, y_calc)
                col_m2.metric("Coeficiente de Determinación (R²)", f"{r2:.5f}")
            else:
                col_m2.info("Añade otro componente para calcular R²")

            # Diagrama de Dispersión (Plotly)
            fig = go.Figure()
            # Puntos de dispersión
            fig.add_trace(go.Scatter(
                x=y_exp, y=y_calc, 
                mode='markers+text', 
                text=nombres, 
                textposition="top center", 
                marker=dict(size=10, color='#0056b3'),
                name="Viscosidad Calculada"
            ))
            
            # Línea de Identidad (X = Y)
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
    st.header("Estimación de Mezclas Multicomponentes")
    metodo = st.selectbox("Seleccione el Método:", ["Wilke", "Davidson"])
    
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
                
                # Aportaciones sum( xi * mu_i / sum(xj * phi_ij) )
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
                
                # Aportaciones sum( xi * mu_i / sum(xj * psi_ij) )
                for i in range(n):
                    den_sum = sum(xi[j] * psi[i,j] for j in range(n))
                    if den_sum > 0: 
                        mu_m += (xi[i] * mui[i]) / den_sum

            st.success(f"Viscosidad Total de la Mezcla ({metodo}): {mu_m:.4f} μPa·s")
        except Exception as e:
            st.error(f"Error matemático en las matrices de mezcla. Detalle: {e}")
