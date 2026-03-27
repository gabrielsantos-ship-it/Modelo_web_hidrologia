import math
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(page_title="Modelo Hidrologico Urbano", layout="wide")

GLOSSARIO_ROWS = [
    {"Termo": "Delta_t_seg", "Unidade": "s", "Descricao": "Passo de tempo da simulacao."},
    {"Termo": "Duracao_chuva_min", "Unidade": "min", "Descricao": "Duracao total do evento; define o numero de passos ativos."},
    {"Termo": "N_passos_ativos", "Unidade": "-", "Descricao": "Numero de intervalos (inclui passo inicial t=0 quando aplicavel)."},
    {"Termo": "Area_m2", "Unidade": "m2", "Descricao": "Area de contribuicao da AC."},
    {"Termo": "L_m", "Unidade": "m", "Descricao": "Comprimento caracteristico de escoamento na AC."},
    {"Termo": "S_m_m", "Unidade": "m/m", "Descricao": "Declividade da superficie de escoamento na AC."},
    {"Termo": "ACs_montante_ids", "Unidade": "-", "Descricao": "IDs das AC imediatamente a montante, separados por ponto e virgula (ex.: 1;2)."},
    {"Termo": "h0_m", "Unidade": "m", "Descricao": "Lamina inicial de agua superficial na AC."},
    {"Termo": "Infiltracao_eq_mm_h", "Unidade": "mm/h", "Descricao": "Taxa de infiltracao equivalente ponderada por area de US."},
    {"Termo": "n_eq", "Unidade": "-", "Descricao": "Coeficiente de Manning equivalente ponderado por area de US."},
    {"Termo": "P_mm_h", "Unidade": "mm/h", "Descricao": "Intensidade de precipitacao no intervalo."},
    {"Termo": "P_m_s", "Unidade": "m/s", "Descricao": "Precipitacao convertida para unidades SI."},
    {"Termo": "i_excesso_m_s", "Unidade": "m/s", "Descricao": "Chuva efetiva = max(P - infiltracao, 0)."},
    {"Termo": "Q_in_m3_s", "Unidade": "m3/s", "Descricao": "Vazao recebida das AC(s) a montante imediata."},
    {"Termo": "Q_out_m3_s", "Unidade": "m3/s", "Descricao": "Vazao de saida da AC (propagacao por Manning)."},
    {"Termo": "Q_ex_m3_s", "Unidade": "m3/s", "Descricao": "Vazao no exutorio selecionado."},
    {"Termo": "Nivel_m", "Unidade": "m", "Descricao": "Profundidade no canal obtida por Manning (secao retangular)."},
    {"Termo": "Cota_abs_m", "Unidade": "m", "Descricao": "Nivel absoluto = Nivel_m + nivel_base."},
    {"Termo": "B_canal", "Unidade": "m", "Descricao": "Largura do canal na secao retangular."},
    {"Termo": "S_canal", "Unidade": "m/m", "Descricao": "Declividade do canal."},
    {"Termo": "n_canal", "Unidade": "-", "Descricao": "Manning do canal."},
    {"Termo": "nivel_base", "Unidade": "m", "Descricao": "Cota de referencia somada ao nivel para cota absoluta."},
    {"Termo": "NSE / Nash-Sutcliffe", "Unidade": "-", "Descricao": "Eficiencia de Nash-Sutcliffe: 1 - sum((O-S)^2)/sum((O-mean(O))^2). Tambem chamado de Nash."},
    {"Termo": "R2 (Pearson)", "Unidade": "-", "Descricao": "Coeficiente de determinacao linear: quadrado da correlacao de Pearson entre observado e simulado."},
    {"Termo": "RMSE / MAE / Vies", "Unidade": "m", "Descricao": "Erro quadratico medio, erro absoluto medio e vies medio (S - O) para nivel."},
]


def parse_upstream_ids(text: str) -> list[int]:
    if text is None:
        return []
    raw = str(text).strip()
    if raw in {"", "0", "nan", "None"}:
        return []
    ids = []
    for part in raw.split(";"):
        p = part.strip()
        if p.isdigit():
            ids.append(int(p))
    return ids


def solve_channel_depth(
    q_ex: np.ndarray, b_canal: float, s_canal: float, n_canal: float, n_iter: int = 5
) -> np.ndarray:
    y = np.zeros_like(q_ex, dtype=float)
    if b_canal <= 0 or s_canal <= 0 or n_canal <= 0:
        return y

    for i, q in enumerate(q_ex):
        if q <= 0:
            y[i] = 0.0
            continue
        yk = max((q * n_canal / (b_canal * math.sqrt(s_canal))) ** (3.0 / 5.0), 1e-6)
        for _ in range(n_iter):
            rh = (b_canal * yk) / max((b_canal + 2.0 * yk), 1e-9)
            yk = q * n_canal / (b_canal * math.sqrt(s_canal) * max(rh, 1e-9) ** (2.0 / 3.0))
        y[i] = max(0.0, yk)
    return y


def run_simulation(
    dt: float,
    chuva_mmh: np.ndarray,
    ac_df: pd.DataFrame,
    exutorio_id: int,
    b_canal: float,
    s_canal: float,
    n_canal: float,
    nivel_base: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_steps = len(chuva_mmh)
    tempo_s = np.arange(n_steps) * dt
    chuva_ms = chuva_mmh / 3600000.0

    ac_df = ac_df.copy()
    ac_df["ID_AC"] = ac_df["ID_AC"].astype(int)
    ac_df = ac_df.sort_values("ID_AC").reset_index(drop=True)
    n_ac = len(ac_df)
    ac_ids = ac_df["ID_AC"].tolist()
    id_to_idx = {ac_id: i for i, ac_id in enumerate(ac_ids)}

    h = np.zeros((n_steps, n_ac), dtype=float)
    q_out = np.zeros((n_steps, n_ac), dtype=float)
    q_in = np.zeros((n_steps, n_ac), dtype=float)
    i_excesso = np.zeros((n_steps, n_ac), dtype=float)

    for j, row in ac_df.iterrows():
        h[0, j] = max(float(row["h0_m"]), 0.0)

    for t in range(n_steps):
        for j, row in ac_df.iterrows():
            area = max(float(row["Area_m2"]), 1e-9)
            comp = max(float(row["L_m"]), 1e-9)
            n_eq = max(float(row["n_eq"]), 1e-9)
            infil_mmh = max(float(row["Infiltracao_eq_mm_h"]), 0.0)
            decliv = max(float(row["S_m_m"]), 1e-9)
            w_eq = area / comp

            h_prev = h[t, j] if t == 0 else h[t - 1, j]
            i_eff = max(chuva_ms[t] - infil_mmh / 3600000.0, 0.0)
            i_excesso[t, j] = i_eff

            upstream_ids = parse_upstream_ids(row["ACs_montante_ids"])
            qin_t = 0.0
            for up_id in upstream_ids:
                if up_id in id_to_idx:
                    qin_t += q_out[t, id_to_idx[up_id]]
            q_in[t, j] = qin_t

            q_unit = (1.0 / n_eq) * (max(h_prev, 0.0) ** (5.0 / 3.0)) * math.sqrt(decliv)
            qout_t = q_unit * w_eq
            q_out[t, j] = qout_t

            h_new = max(0.0, h_prev + dt * (i_eff + qin_t / area - qout_t / area))
            h[t, j] = h_new

    if exutorio_id not in id_to_idx:
        exutorio_id = ac_ids[-1]

    q_ex = q_out[:, id_to_idx[exutorio_id]]
    nivel = solve_channel_depth(q_ex, b_canal, s_canal, n_canal, n_iter=5)
    cota = nivel + nivel_base

    df_saida = pd.DataFrame(
        {
            "Passo": np.arange(n_steps),
            "Tempo_s": tempo_s,
            "P_mm_h": chuva_mmh,
            "P_m_s": chuva_ms,
            "Q_ex_m3_s": q_ex,
            "Nivel_m": nivel,
            "Cota_abs_m": cota,
        }
    )

    registros = []
    for j, ac_id in enumerate(ac_ids):
        df_ac = pd.DataFrame(
            {
                "Passo": np.arange(n_steps),
                "Tempo_s": tempo_s,
                "ID_AC": ac_id,
                "i_excesso_m_s": i_excesso[:, j],
                "Q_in_m3_s": q_in[:, j],
                "Q_out_m3_s": q_out[:, j],
                "h_m": h[:, j],
            }
        )
        registros.append(df_ac)
    df_ac_detalhe = pd.concat(registros, ignore_index=True)
    return df_saida, df_ac_detalhe


def calcular_parametros_equivalentes_por_us(ac_df: pd.DataFrame, us_df: pd.DataFrame) -> pd.DataFrame:
    ac_out = ac_df.copy()
    us_work = us_df.copy()
    us_work["ID_AC"] = pd.to_numeric(us_work["ID_AC"], errors="coerce")
    us_work["Area_US_m2"] = pd.to_numeric(us_work["Area_US_m2"], errors="coerce")
    us_work["Infiltracao_US_mm_h"] = pd.to_numeric(us_work["Infiltracao_US_mm_h"], errors="coerce")
    us_work["n_US"] = pd.to_numeric(us_work["n_US"], errors="coerce")
    us_work = us_work.dropna(subset=["ID_AC", "Area_US_m2", "Infiltracao_US_mm_h", "n_US"])

    for i, row in ac_out.iterrows():
        ac_id = int(row["ID_AC"])
        area_ac = float(row["Area_m2"])
        us_ac = us_work[us_work["ID_AC"].astype(int) == ac_id]
        if us_ac.empty or area_ac <= 0:
            continue
        infil_eq = float((us_ac["Area_US_m2"] * us_ac["Infiltracao_US_mm_h"]).sum() / area_ac)
        n_eq = float((us_ac["Area_US_m2"] * us_ac["n_US"]).sum() / area_ac)
        ac_out.loc[i, "Infiltracao_eq_mm_h"] = infil_eq
        ac_out.loc[i, "n_eq"] = n_eq
    return ac_out


def nash_sutcliffe_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return float("nan")
    den = float(np.sum((obs - np.mean(obs)) ** 2))
    if den < 1e-30:
        return float("nan")
    return 1.0 - float(np.sum((obs - sim) ** 2)) / den


def r2_pearson(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 2:
        return float("nan")
    r = np.corrcoef(obs, sim)[0, 1]
    return float(r**2) if np.isfinite(r) else float("nan")


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 1:
        return float("nan")
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 1:
        return float("nan")
    return float(np.mean(np.abs(obs - sim)))


def vies_medio(obs: np.ndarray, sim: np.ndarray) -> float:
    """Vies medio = mean(sim - obs)."""
    obs = np.asarray(obs, dtype=float)
    sim = np.asarray(sim, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs, sim = obs[mask], sim[mask]
    if len(obs) < 1:
        return float("nan")
    return float(np.mean(sim - obs))


def montar_df_validacao(df_saida: pd.DataFrame, df_obs_antigo: pd.DataFrame | None) -> pd.DataFrame:
    out = df_saida[["Passo", "Tempo_s"]].copy()
    out["Nivel_sim_m"] = df_saida["Nivel_m"].values
    if df_obs_antigo is not None and "Tempo_s" in df_obs_antigo.columns and "Nivel_obs_m" in df_obs_antigo.columns:
        old = df_obs_antigo[["Tempo_s", "Nivel_obs_m"]].drop_duplicates(subset=["Tempo_s"])
        out = out.merge(old, on="Tempo_s", how="left")
    else:
        out["Nivel_obs_m"] = np.nan
    return out


with st.sidebar:
    st.header("Menu")
    pagina = st.radio(
        "Navegar",
        ["Simulacao", "Validacao", "Instrucoes", "Descricao das variaveis"],
        label_visibility="collapsed",
    )

if pagina == "Instrucoes":
    st.title("Instrucoes de uso")
    st.markdown(
        """
### Fluxo recomendado

1. **Configuracao** (barra lateral): defina `Delta t` e a duracao total da chuva em minutos.
2. **Parametros do canal**: largura, declividade, Manning e nivel de base (cota opcional).
3. **Areas de contribuicao (AC)**: informe geometria, declividade, montantes (`ACs_montante_ids` com IDs separados por `;`) e lamina inicial `h0_m`.
4. **Unidades de uso do solo (US)**: preencha areas e parametros por classe. Com a opcao automatica ativada, `Infiltracao_eq_mm_h` e `n_eq` sao calculados por media ponderada pela area da AC.
5. **Chuva**: edite `P_mm_h` por passo. O passo inicial (`t = 0`) pode ser zero; a chuva efetiva costuma comecar no passo seguinte, conforme seu evento.
6. Clique em **Rodar simulacao** e visualize os hidrogramas de nivel e vazao. Baixe os resultados em Excel se desejar.
7. Aba **Validacao**: informe niveis medidos em campo (grade ou CSV) e compare com a serie simulada (NSE, R2, RMSE, etc.).

### Observacoes

- O modelo usa passo de tempo discreto e evento unico de chuva.
- A propagacao superficial usa Manning em plano inclinado; a conversao para nivel no canal usa Manning em secao retangular.
- Se alterar `Numero de ACs`, recarregue a pagina se necessario para atualizar linhas padrao das US.
        """
    )

elif pagina == "Descricao das variaveis":
    st.title("Descricao das variaveis e termos")
    st.caption("Tabela de referencia alinhada a metodologia do modelo.")
    st.dataframe(
        pd.DataFrame(GLOSSARIO_ROWS),
        use_container_width=True,
        hide_index=True,
    )

elif pagina == "Validacao":
    st.title("Validacao: nivel observado vs simulado")
    st.caption(
        "Compare niveis medidos em campo com a serie simulada no exutorio. "
        "O Nash-Sutcliffe (NSE) e o mesmo indicador frequentemente chamado de Nash."
    )

    df_sim = st.session_state.get("df_saida_validacao")
    if df_sim is None:
        st.warning(
            "Nenhuma simulacao disponivel. Va em **Simulacao**, preencha os dados e clique em **Rodar simulacao**."
        )
    else:
        st.info(
            f"Serie simulada carregada: **{len(df_sim)}** passos. "
            "Unidade de nivel: metros (mesma referencia do modelo)."
        )

        tab_manual, tab_csv = st.tabs(["Entrada manual na grade", "Importar CSV"])

        with tab_manual:
            prev = st.session_state.get("df_validacao_obs")
            df_val = montar_df_validacao(df_sim, prev)
            df_edit = st.data_editor(
                df_val,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                disabled=["Passo", "Tempo_s", "Nivel_sim_m"],
                column_config={
                    "Nivel_obs_m": st.column_config.NumberColumn(
                        "Nivel observado (m)",
                        help="Medicao de campo no mesmo instante de Tempo_s (deixe vazio para ignorar o passo nas metricas).",
                        format="%.4f",
                    ),
                },
                key="editor_validacao_manual",
            )
            st.session_state["df_validacao_obs"] = df_edit.copy()

        with tab_csv:
            st.markdown(
                "Arquivo **CSV** com colunas `Tempo_s` e `Nivel_obs_m` (opcional: `Passo`). "
                "Valores separados por virgula ou ponto e virgula."
            )
            up = st.file_uploader("Carregar CSV de niveis observados", type=["csv"])
            if up is not None:
                try:
                    raw = up.read()
                    df_up = None
                    for sep in [";", ",", "\t"]:
                        try:
                            df_try = pd.read_csv(BytesIO(raw), sep=sep)
                            if len(df_try.columns) >= 2:
                                df_up = df_try
                                break
                        except Exception:
                            continue
                    if df_up is None:
                        df_up = pd.read_csv(BytesIO(raw))

                    df_up.columns = [c.strip() for c in df_up.columns]
                    if "Tempo_s" not in df_up.columns and "tempo_s" in df_up.columns:
                        df_up = df_up.rename(columns={"tempo_s": "Tempo_s"})
                    if "Nivel_obs_m" not in df_up.columns:
                        for cand in ["Nivel_obs_m", "nivel_obs", "Nivel_obs", "obs", "y"]:
                            if cand in df_up.columns:
                                df_up = df_up.rename(columns={cand: "Nivel_obs_m"})
                                break

                    if "Tempo_s" not in df_up.columns or "Nivel_obs_m" not in df_up.columns:
                        st.error("O CSV precisa ter colunas Tempo_s e Nivel_obs_m (ou nomes equivalentes).")
                    else:
                        df_up["Tempo_s"] = pd.to_numeric(df_up["Tempo_s"], errors="coerce")
                        df_up["Nivel_obs_m"] = pd.to_numeric(df_up["Nivel_obs_m"], errors="coerce")
                        df_up = df_up.dropna(subset=["Tempo_s", "Nivel_obs_m"])
                        base = df_sim[["Passo", "Tempo_s"]].copy()
                        base["Nivel_sim_m"] = df_sim["Nivel_m"].values
                        merged = base.merge(
                            df_up[["Tempo_s", "Nivel_obs_m"]].drop_duplicates(subset=["Tempo_s"]),
                            on="Tempo_s",
                            how="left",
                        )
                        st.session_state["df_validacao_obs"] = merged
                        st.success(f"Importados {len(df_up)} pontos; alinhados por Tempo_s a grade da simulacao.")
                except Exception as exc:
                    st.error(f"Erro ao ler CSV: {exc}")

        df_obs = st.session_state.get("df_validacao_obs")
        if df_obs is not None and "Nivel_obs_m" in df_obs.columns and "Nivel_sim_m" in df_obs.columns:
            mask = df_obs["Nivel_obs_m"].notna()
            n_valid = int(mask.sum())
            if n_valid < 2:
                st.warning("Informe pelo menos **2** valores de nivel observado para calcular metricas.")
            else:
                obs = df_obs.loc[mask, "Nivel_obs_m"].astype(float).values
                sim = df_obs.loc[mask, "Nivel_sim_m"].astype(float).values

                nse = nash_sutcliffe_efficiency(obs, sim)
                r2 = r2_pearson(obs, sim)
                err_rmse = rmse(obs, sim)
                err_mae = mae(obs, sim)
                vb = vies_medio(obs, sim)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("NSE (Nash-Sutcliffe)", f"{nse:.4f}")
                m2.metric("R2 (Pearson)", f"{r2:.4f}")
                m3.metric("RMSE (m)", f"{err_rmse:.4f}")
                m4.metric("MAE (m)", f"{err_mae:.4f}")
                st.metric("Vies medio (sim - obs) (m)", f"{vb:.4f}")
                st.caption("Vies medio: media de (simulado - observado) nos passos com observacao.")

                st.subheader("Grafico comparativo")
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=df_obs["Tempo_s"],
                        y=df_obs["Nivel_sim_m"],
                        name="Nivel simulado",
                        mode="lines",
                        line=dict(color="#1f77b4", width=2),
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_obs.loc[mask, "Tempo_s"],
                        y=obs,
                        name="Nivel observado",
                        mode="markers+lines",
                        line=dict(color="#d62728", dash="dot"),
                        marker=dict(size=8),
                    )
                )
                fig.update_layout(
                    title="Nivel no canal: observado vs simulado",
                    xaxis_title="Tempo (s)",
                    yaxis_title="Nivel (m)",
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Tabela de comparacao (passos com observacao)")
                st.dataframe(
                    df_obs.loc[mask].reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                )

else:
    st.title("Modelo Hidrologico Urbano - Escoamento e Nivel no Canal")
    st.caption("Versao web local para simulacao de evento unico de chuva-vazao.")

    with st.sidebar:
        st.header("Configuracao")
        dt_seg = st.number_input("Delta t (s)", min_value=60, value=600, step=60)
        duracao_min = st.number_input("Duracao da chuva (min)", min_value=10, value=40, step=10)
        n_passos = int(math.ceil((duracao_min * 60) / dt_seg) + 1)
        st.write(f"Passos ativos: {n_passos} (inclui t=0)")

    st.subheader("1) Parametros do canal")
    col1, col2, col3, col4 = st.columns(4)
    b_canal = col1.number_input("B_canal (m)", min_value=0.1, value=4.0)
    s_canal = col2.number_input("S_canal (m/m)", min_value=0.0001, value=0.004, format="%.4f")
    n_canal = col3.number_input("n_canal", min_value=0.001, value=0.018, format="%.3f")
    nivel_base = col4.number_input("Nivel base (m)", value=0.2, format="%.3f")

    st.subheader("2) Areas de contribuicao (AC)")
    n_ac = st.number_input("Numero de ACs", min_value=1, max_value=20, value=3, step=1)
    exutorio_id = st.number_input("ID da AC de exutorio", min_value=1, max_value=20, value=int(n_ac))

    default_ac = pd.DataFrame(
        [
            {
                "ID_AC": 1,
                "Area_m2": 85000.0,
                "L_m": 350.0,
                "S_m_m": 0.015,
                "ACs_montante_ids": "0",
                "h0_m": 0.0,
                "Infiltracao_eq_mm_h": 12.0,
                "n_eq": 0.12,
            },
            {
                "ID_AC": 2,
                "Area_m2": 92000.0,
                "L_m": 420.0,
                "S_m_m": 0.012,
                "ACs_montante_ids": "1",
                "h0_m": 0.0,
                "Infiltracao_eq_mm_h": 10.0,
                "n_eq": 0.10,
            },
            {
                "ID_AC": 3,
                "Area_m2": 130000.0,
                "L_m": 500.0,
                "S_m_m": 0.010,
                "ACs_montante_ids": "1;2",
                "h0_m": 0.0,
                "Infiltracao_eq_mm_h": 8.0,
                "n_eq": 0.09,
            },
        ]
    )
    if n_ac > len(default_ac):
        for idx in range(len(default_ac) + 1, n_ac + 1):
            default_ac.loc[len(default_ac)] = {
                "ID_AC": idx,
                "Area_m2": 100000.0,
                "L_m": 400.0,
                "S_m_m": 0.01,
                "ACs_montante_ids": "0",
                "h0_m": 0.0,
                "Infiltracao_eq_mm_h": 10.0,
                "n_eq": 0.1,
            }
    ac_df_edit = default_ac.head(n_ac).copy()

    ac_df = st.data_editor(
        ac_df_edit,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("3) Unidades de uso do solo (US) - editavel")
    usar_us_para_eq = st.checkbox(
        "Calcular Infiltracao_eq_mm_h e n_eq automaticamente a partir das US",
        value=True,
    )
    classes_padrao = ["Superficie impermeavel", "Gramado", "Vegetacao arborea"]
    rows_us = []
    for ac_id in range(1, int(n_ac) + 1):
        if ac_id == 1:
            areas = [34000.0, 28000.0, 23000.0]
        elif ac_id == 2:
            areas = [46000.0, 25000.0, 21000.0]
        elif ac_id == 3:
            areas = [78000.0, 31000.0, 21000.0]
        else:
            area_ac = float(ac_df.loc[ac_df["ID_AC"] == ac_id, "Area_m2"].iloc[0])
            areas = [0.6 * area_ac, 0.25 * area_ac, 0.15 * area_ac]
        for idx, classe in enumerate(classes_padrao, start=1):
            rows_us.append(
                {
                    "ID_AC": ac_id,
                    "ID_US": idx,
                    "Classe_US": classe,
                    "Area_US_m2": areas[idx - 1],
                    "Infiltracao_US_mm_h": [1.0, 15.0, 30.0][idx - 1],
                    "n_US": [0.015, 0.20, 0.35][idx - 1],
                }
            )
    us_default_df = pd.DataFrame(rows_us)
    us_df = st.data_editor(
        us_default_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )

    if usar_us_para_eq:
        ac_preview = calcular_parametros_equivalentes_por_us(ac_df, us_df)
        st.caption("Parametros equivalentes calculados por US (ponderados pela area da AC).")
        st.dataframe(
            ac_preview[["ID_AC", "Infiltracao_eq_mm_h", "n_eq"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        ac_preview = ac_df.copy()

    st.subheader("4) Chuva (mm/h)")
    tempo_vec = np.arange(n_passos) * dt_seg
    default_chuva = np.zeros(n_passos)
    if n_passos >= 5:
        default_chuva[1:5] = [18, 42, 25, 8]
    chuva_df = pd.DataFrame({"Passo": np.arange(n_passos), "Tempo_s": tempo_vec, "P_mm_h": default_chuva})
    chuva_df_edit = st.data_editor(
        chuva_df,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        disabled=["Passo", "Tempo_s"],
    )

    if st.button("Rodar simulacao", type="primary"):
        try:
            chuva_mmh = chuva_df_edit["P_mm_h"].astype(float).to_numpy()
            ac_df = ac_preview.copy()
            ac_df["ID_AC"] = ac_df["ID_AC"].astype(int)
            for col in ["Area_m2", "L_m", "S_m_m", "h0_m", "Infiltracao_eq_mm_h", "n_eq"]:
                ac_df[col] = ac_df[col].astype(float)

            df_saida, df_ac_detalhe = run_simulation(
                dt=float(dt_seg),
                chuva_mmh=chuva_mmh,
                ac_df=ac_df,
                exutorio_id=int(exutorio_id),
                b_canal=float(b_canal),
                s_canal=float(s_canal),
                n_canal=float(n_canal),
                nivel_base=float(nivel_base),
            )

            st.session_state["df_saida_validacao"] = df_saida.copy()
            st.session_state["df_validacao_obs"] = montar_df_validacao(
                df_saida, st.session_state.get("df_validacao_obs")
            )

            st.success("Simulacao concluida.")
            c1, c2, c3 = st.columns(3)
            c1.metric("Pico de vazao (m3/s)", f"{df_saida['Q_ex_m3_s'].max():.4f}")
            c2.metric("Pico de nivel (m)", f"{df_saida['Nivel_m'].max():.4f}")
            c3.metric("Tempo do pico (s)", f"{df_saida.loc[df_saida['Nivel_m'].idxmax(), 'Tempo_s']:.0f}")

            fig_nivel = px.line(
                df_saida,
                x="Tempo_s",
                y="Nivel_m",
                markers=False,
                title="Hidrograma de nivel no canal",
            )
            fig_nivel.update_layout(xaxis_title="Tempo (s)", yaxis_title="Nivel (m)")
            st.plotly_chart(fig_nivel, use_container_width=True)

            fig_q = px.line(df_saida, x="Tempo_s", y="Q_ex_m3_s", title="Hidrograma de vazao no exutorio")
            fig_q.update_layout(xaxis_title="Tempo (s)", yaxis_title="Vazao (m3/s)")
            st.plotly_chart(fig_q, use_container_width=True)

            st.subheader("Saida - exutorio")
            st.dataframe(df_saida, use_container_width=True, hide_index=True)

            st.subheader("Saida detalhada por AC")
            st.dataframe(df_ac_detalhe, use_container_width=True, hide_index=True)

            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                df_saida.to_excel(writer, index=False, sheet_name="Exutorio")
                df_ac_detalhe.to_excel(writer, index=False, sheet_name="Detalhe_AC")
            st.download_button(
                "Baixar resultados em Excel",
                data=output.getvalue(),
                file_name="resultado_modelo_hidrologico.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as exc:
            st.error(f"Erro na simulacao: {exc}")

