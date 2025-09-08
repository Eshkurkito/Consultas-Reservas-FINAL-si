
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, datetime, timedelta

st.set_page_config(page_title="Revenue Definitivo", layout="wide")

TITLE = "📊 App Definitiva de Revenue"
SHEET_NAME = "Estado de pagos de las reservas"
REQUIRED_COLS = ["Alojamiento", "Portal", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]

# =============================
# Utilidades de carga y limpieza
# =============================
@st.cache_data(show_spinner=False)
def load_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=SHEET_NAME)
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Falta la columna requerida: '{c}' en la hoja '{SHEET_NAME}'.")
    df["Alojamiento"] = df["Alojamiento"].astype(str)
    if "Portal" in df.columns:
        df["Portal"] = df["Portal"].astype(str).replace({"nan": np.nan})
    for c in ["Fecha alta", "Fecha entrada", "Fecha salida"]:
        df[c] = pd.to_datetime(df[c], errors="coerce").dt.normalize()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce")
    df = df[df["Precio"].fillna(0) > 0].copy()
    return df

def month_range_defaults():
    today = pd.Timestamp.today().tz_localize(None).normalize()
    start = today.to_period("M").start_time
    end   = today.to_period("M").end_time
    return start.date(), end.date()

def _ensure_dates(v):
    return pd.to_datetime(v, errors="coerce").tz_localize(None).normalize()

# =============================
# Expansión a noches por día
# =============================
def expand_nightly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["Fecha entrada", "Fecha salida", "Precio"]).copy()
    x["Fecha entrada"] = pd.to_datetime(x["Fecha entrada"]).dt.normalize()
    x["Fecha salida"]  = pd.to_datetime(x["Fecha salida"]).dt.normalize()
    x["Fecha alta"]    = pd.to_datetime(x["Fecha alta"]).dt.normalize()
    x["los"] = (x["Fecha salida"] - x["Fecha entrada"]).dt.days.clip(lower=1)
    x["adr_reserva"] = x["Precio"] / x["los"]

    rep = np.repeat(x.index.values, x["los"].values)
    base = x.loc[rep, ["Alojamiento", "Portal", "Fecha entrada", "Fecha salida", "Fecha alta", "adr_reserva"]].copy()
    offsets = x["los"].apply(lambda n: np.arange(n)).explode().values
    base["Fecha"] = (base["Fecha entrada"].values.astype("datetime64[D]") + offsets).astype("datetime64[ns]")
    base["DOW"] = pd.to_datetime(base["Fecha"]).dt.day_name(locale="es_ES") if hasattr(pd.Series([]).dt, "day_name") else pd.to_datetime(base["Fecha"]).dt.day_name()
    base.rename(columns={"adr_reserva": "ADR_dia"}, inplace=True)
    return base[["Alojamiento", "Portal", "Fecha", "Fecha alta", "ADR_dia", "DOW"]]

# =============================
# Series diarias por periodo y corte
# =============================
def daily_series(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props_sel=None):
    dfc = df.copy()
    for c in ["Fecha alta","Fecha entrada","Fecha salida"]:
        dfc[c] = pd.to_datetime(dfc[c]).dt.normalize()
    if props_sel:
        dfc = dfc[dfc["Alojamiento"].isin(props_sel)]
    dfc = dfc[dfc["Fecha alta"] <= cutoff]

    nights = expand_nightly(dfc)
    nights = nights[(nights["Fecha"] >= start) & (nights["Fecha"] <= end)]
    if nights.empty:
        idx = pd.date_range(start, end, freq="D")
        return pd.DataFrame({"Fecha": idx, "noches": 0, "ingresos": 0.0})
    agg = nights.groupby("Fecha").agg(noches=("Fecha","size"), ingresos=("ADR_dia","sum")).reset_index()
    idx = pd.date_range(start, end, freq="D")
    agg = agg.set_index("Fecha").reindex(idx).fillna(0.0).rename_axis("Fecha").reset_index()
    agg["noches"] = agg["noches"].astype(int)
    return agg

# =============================
# Pace (curva D)
# =============================
def build_pace(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, props_sel=None, years_back=3):
    base = df.copy()
    for c in ["Fecha alta","Fecha entrada","Fecha salida"]:
        base[c] = pd.to_datetime(base[c]).dt.normalize()
    if props_sel:
        base = base[base["Alojamiento"].isin(props_sel)]

    per_year = []
    for back in range(1, years_back + 1):
        ys = (start - pd.DateOffset(years=back)).normalize()
        ye = (end   - pd.DateOffset(years=back)).normalize()
        df_y = base[(base["Fecha entrada"] <= ye) & (base["Fecha salida"] >= ys)].copy()
        if df_y.empty:
            continue
        nights_y = expand_nightly(df_y)
        nights_y = nights_y[(nights_y["Fecha"] >= ys) & (nights_y["Fecha"] <= ye)]
        if nights_y.empty:
            continue
        nights_y["D"] = (nights_y["Fecha"] - nights_y["Fecha alta"]).dt.days
        nights_y = nights_y[nights_y["D"] >= 0]
        per_year.append((back, nights_y))

    hist_curves = []
    for back, ny in per_year:
        g = ny.groupby("D").size().rename("n").reset_index()
        tot_n = int(ny.shape[0]) if ny.shape[0] > 0 else 1
        g["F"] = g["n"].iloc[::-1].cumsum()[::-1] / tot_n
        g["year_back"] = back
        hist_curves.append(g)
    if not hist_curves:
        return None

    hist = pd.concat(hist_curves, ignore_index=True)
    D_max = int(hist["D"].max())
    idx = pd.DataFrame({"D": np.arange(0, D_max + 1, 1)})
    def pctl(x, q):
        x = x.dropna()
        return np.nanpercentile(x, q) if len(x) else np.nan
    med = idx.merge(hist.groupby("D")["F"].median().rename("F50"), on="D", how="left")
    p25 = idx.merge(hist.groupby("D")["F"].apply(lambda x: pctl(x, 25)).rename("F25"), on="D", how="left")
    p75 = idx.merge(hist.groupby("D")["F"].apply(lambda x: pctl(x, 75)).rename("F75"), on="D", how="left")
    out = med.merge(p25, on="D", how="left").merge(p75, on="D", how="left")

    today = pd.Timestamp.today().normalize()
    df_now = base[base["Fecha alta"] <= today].copy()
    nights_now = expand_nightly(df_now)
    nights_now = nights_now[(nights_now["Fecha"] >= start) & (nights_now["Fecha"] <= end)]
    if nights_now.empty:
        out["F_now"] = np.nan
        return out
    nights_now["D"] = (nights_now["Fecha"] - nights_now["Fecha alta"]).dt.days
    nights_now = nights_now[nights_now["D"] >= 0]
    g_now = nights_now.groupby("D").size().rename("n").reset_index()
    tot_n_now = int(nights_now.shape[0]) if nights_now.shape[0] > 0 else 1
    g_now["F_now"] = g_now["n"].iloc[::-1].cumsum()[::-1] / tot_n_now
    out = out.merge(g_now[["D", "F_now"]], on="D", how="left")
    return out

# -----------------------------
# Pace por canal (Portal)
# -----------------------------
def build_pace_by_channel(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, props_sel=None, years_back=3, min_points=20):
    base = df.copy()
    for c in ["Fecha entrada","Fecha salida","Fecha alta"]:
        base[c] = pd.to_datetime(base[c], errors="coerce").dt.normalize()
    if props_sel:
        base = base[base["Alojamiento"].isin(props_sel)]
    base["Portal"] = base["Portal"].fillna("—").astype(str)

    hist_list = []
    for back in range(1, years_back+1):
        ys = (start - pd.DateOffset(years=back)).normalize()
        ye = (end   - pd.DateOffset(years=back)).normalize()
        df_y = base[(base["Fecha entrada"] <= ye) & (base["Fecha salida"] >= ys)].copy()
        if df_y.empty: 
            continue
        nights_y = expand_nightly(df_y)
        nights_y = nights_y[(nights_y["Fecha"] >= ys) & (nights_y["Fecha"] <= ye)]
        if nights_y.empty:
            continue
        nights_y = nights_y.merge(df_y[["Alojamiento","Portal","Fecha alta","Fecha entrada","Fecha salida"]],
                                  left_on=["Alojamiento","Fecha alta"], right_on=["Alojamiento","Fecha alta"], how="left")
        nights_y["D"] = (nights_y["Fecha"] - nights_y["Fecha alta"]).dt.days
        nights_y = nights_y[nights_y["D"] >= 0]
        hist_list.append(nights_y[["Portal","D"]])
    if not hist_list:
        return None

    hist = pd.concat(hist_list, ignore_index=True)
    counts = hist.groupby("Portal").size()
    valid_canals = counts[counts >= min_points].index.tolist()
    hist = hist[hist["Portal"].isin(valid_canals)]
    if hist.empty:
        return None

    res_list = []
    for canal, g in hist.groupby("Portal"):
        gD = g.groupby("D").size().rename("n").reset_index()
        tot_n = int(g.shape[0]) if g.shape[0]>0 else 1
        gD["F50"] = gD["n"].iloc[::-1].cumsum()[::-1] / tot_n
        gD["Portal"] = canal
        res_list.append(gD[["D","Portal","F50"]])
    hist_curves = pd.concat(res_list, ignore_index=True)

    today = pd.Timestamp.today().normalize()
    now_df = base[base["Fecha alta"] <= today].copy()
    nights_now = expand_nightly(now_df)
    nights_now = nights_now[(nights_now["Fecha"] >= start) & (nights_now["Fecha"] <= end)]
    if nights_now.empty:
        return hist_curves.rename(columns={"Portal":"Canal"}).assign(F_now=np.nan, idx_pp=np.nan)

    nights_now = nights_now.merge(now_df[["Alojamiento","Portal","Fecha alta"]], on=["Alojamiento","Fecha alta"], how="left")
    nights_now["Portal"] = nights_now["Portal"].fillna("—").astype(str)
    nights_now["D"] = (nights_now["Fecha"] - nights_now["Fecha alta"]).dt.days
    nights_now = nights_now[nights_now["D"] >= 0]

    now_curves = []
    for canal, g in nights_now.groupby("Portal"):
        gD = g.groupby("D").size().rename("n").reset_index()
        tot_n = int(g.shape[0]) if g.shape[0]>0 else 1
        gD["F_now"] = gD["n"].iloc[::-1].cumsum()[::-1] / tot_n
        gD["Portal"] = canal
        now_curves.append(gD[["D","Portal","F_now"]])
    now_curves = pd.concat(now_curves, ignore_index=True) if now_curves else pd.DataFrame(columns=["D","Portal","F_now"])

    out = hist_curves.merge(now_curves, on=["D","Portal"], how="left")
    idx_tbl = out.dropna(subset=["F50","F_now"]).copy()
    if not idx_tbl.empty:
        idx_tbl["diff_pp"] = (idx_tbl["F_now"] - idx_tbl["F50"]) * 100.0
        idx = idx_tbl.groupby("Portal")["diff_pp"].mean().rename("idx_pp").reset_index()
        out = out.merge(idx, on="Portal", how="left")
    out.rename(columns={"Portal":"Canal"}, inplace=True)
    return out

# =============================
# Evolución por corte
# =============================
def evolucion_por_corte(df: pd.DataFrame, base_cut: pd.Timestamp, n_cortes: int, step_days: int,
                        start: pd.Timestamp, end: pd.Timestamp, props_sel=None):
    cortes = [base_cut - pd.to_timedelta(step_days*i, unit="D") for i in range(int(n_cortes))][::-1]
    tablas = []
    for c in cortes:
        ser = daily_series(df, c, start, end, props_sel)
        ser["Corte"] = c.normalize().date()
        tablas.append(ser)
    if not tablas:
        return None
    evo = pd.concat(tablas, ignore_index=True)
    return evo

# =============================
# KPIs y auxiliares
# =============================
def kpis_periodo(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp,
                 props_sel=None, inventario:int|None=None):
    ser = daily_series(df, cutoff, start, end, props_sel)
    if ser.empty:
        return {"ingresos":0.0, "noches":0, "occ":0.0, "adr":0.0, "revpar":0.0}, ser
    noches = int(ser["noches"].sum())
    ingresos = float(ser["ingresos"].sum())
    dias = (end - start).days + 1
    inv = inventario if inventario and inventario>0 else (len(props_sel) if props_sel else 1)
    occ = (noches/(inv*dias))*100.0 if inv>0 and dias>0 else 0.0
    adr = (ingresos/noches) if noches>0 else 0.0
    revpar = (ingresos/dias/inv) if inv>0 and dias>0 else 0.0
    return {"ingresos":ingresos,"noches":noches,"occ":occ,"adr":adr,"revpar":revpar}, ser

def yoy_compare(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp,
                props_sel=None, inventario:int|None=None):
    (k_now, ser_now) = kpis_periodo(df, cutoff, start, end, props_sel, inventario)
    last_start = (start - pd.DateOffset(years=1)).normalize()
    last_end   = (end   - pd.DateOffset(years=1)).normalize()
    last_cut   = (cutoff - pd.DateOffset(years=1)).normalize()
    (k_prev, ser_prev) = kpis_periodo(df, last_cut, last_start, last_end, props_sel, inventario)
    def pct(a, b): return ((a-b)/b*100.0) if b and b!=0 else np.nan
    deltas = {
        "Δ ingresos %": pct(k_now["ingresos"], k_prev["ingresos"]),
        "Δ noches %": pct(k_now["noches"], k_prev["noches"]),
        "Δ ADR %": pct(k_now["adr"], k_prev["adr"]),
        "Δ Occ pp": (k_now["occ"] - k_prev["occ"]),
        "Δ RevPAR %": pct(k_now["revpar"], k_prev["revpar"]),
    }
    return (k_now, ser_now), (k_prev, ser_prev), deltas, (last_start, last_end, last_cut)

# =============================
# Narrativa y alertas PRO
# =============================
def narrativa_pro(k_now, k_prev, pace_df=None):
    lines = []
    inc_diff = k_now["ingresos"] - k_prev["ingresos"]
    vol_diff = k_now["noches"] - k_prev["noches"]
    adr_diff = k_now["adr"] - k_prev["adr"]

    if inc_diff >= 0:
        if vol_diff>0 and adr_diff>0: driver = "mayor ocupación y mejora de ADR"
        elif vol_diff>0: driver = "mayor ocupación"
        elif adr_diff>0: driver = "mejor ADR"
        else: driver = "un mix de reservas más favorable"
        lines.append(f"Los ingresos del periodo mejoran, impulsados por {driver}.")
    else:
        if vol_diff<0 and adr_diff<0: driver = "descensos simultáneos en ocupación y ADR"
        elif vol_diff<0: driver = "menor ocupación"
        elif adr_diff<0: driver = "ADR inferior"
        else: driver = "un mix desfavorable"
        lines.append(f"Los ingresos del periodo retroceden; el principal factor es {driver}.")

    if pace_df is not None and not pace_df.empty and "F_now" in pace_df.columns:
        ref = pace_df["F50"].dropna()
        now = pace_df["F_now"].dropna()
        if len(ref)>5 and len(now)>5:
            merged = pace_df.dropna(subset=["F50","F_now"]).copy()
            diff_pp = (merged["F_now"] - merged["F50"]).mean() * 100.0
            if diff_pp > 3:
                lines.append(f"El ritmo de ventas es **más rápido** que el histórico (≈ +{diff_pp:.1f} pp sobre la mediana).")
            elif diff_pp < -3:
                lines.append(f"El ritmo de ventas es **más lento** que el histórico (≈ {diff_pp:.1f} pp por debajo de la mediana).")
            else:
                lines.append("El ritmo de ventas es **similar** a la mediana histórica.")
    if k_now["occ"] < 70:
        lines.append("Recomendación: activar campañas en días valle (Do–Lu), revisar visibilidad y aplicar promociones suaves (3–5%).")
    if k_now["occ"] > 85 and k_now["adr"] <= k_prev["adr"]:
        lines.append("Recomendación: hay margen para elevar tarifas en picos (Ju–Sa) y endurecer condiciones promocionales.")
    if k_now["adr"] < k_prev["adr"]:
        lines.append("Recomendación: optimizar mix de canales (priorizar directo con beneficios no tarifarios) para proteger ADR.")
    return lines

def compute_alerts_pro(k_now, k_prev, ser_now: pd.DataFrame, ser_prev: pd.DataFrame,
                       pace_df=None, target_occ_pct: float = 80.0,
                       start: pd.Timestamp=None, end: pd.Timestamp=None, inv:int|None=None):
    alerts = []
    # Pace index
    if pace_df is not None and not pace_df.dropna(subset=["F50","F_now"]).empty:
        merged = pace_df.dropna(subset=["F50","F_now"]).copy()
        diff_pp = float((merged["F_now"] - merged["F50"]).mean() * 100.0)
        if diff_pp <= -5: alerts.append(("🔴 Ritmo lento", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp <= -2: alerts.append(("🟠 Ritmo algo lento", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp >= +5: alerts.append(("🟢 Ritmo rápido", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp >= +2: alerts.append(("🟢 Ritmo algo rápido", f"{diff_pp:+.1f} pp vs mediana"))
        else: alerts.append(("🟡 Ritmo en línea", f"{diff_pp:+.1f} pp vs mediana"))
    # ADR outlier vs YoY
    if ser_prev is not None and not ser_prev.empty:
        sp = ser_prev.copy()
        sp["ADR_dia"] = np.where(sp["noches"]>0, sp["ingresos"]/sp["noches"], np.nan)
        if sp["ADR_dia"].notna().any():
            q1 = float(np.nanpercentile(sp["ADR_dia"], 25))
            q3 = float(np.nanpercentile(sp["ADR_dia"], 75))
            iqr = q3 - q1
            if iqr > 0:
                if k_now["adr"] > q3 + 1.5*iqr:
                    alerts.append(("🟢 ADR alto (fuera de rango)", f"ADR actual {k_now['adr']:.0f}€ > Q3+1.5·IQR ({q3+1.5*iqr:.0f}€)"))
                elif k_now["adr"] < q1 - 1.5*iqr:
                    alerts.append(("🔴 ADR bajo (fuera de rango)", f"ADR actual {k_now['adr']:.0f}€ < Q1−1.5·IQR ({q1-1.5*iqr:.0f}€)"))
    # Riesgo de objetivo de ocupación
    if start is not None and end is not None:
        inv_eff = inv if inv and inv>0 else 1
        today = pd.Timestamp.today().normalize()
        if start <= today <= end:
            days_elapsed = int((today - start).days) + 1
            target_by_today = target_occ_pct/100.0 * inv_eff * days_elapsed
            nights_by_today = int(ser_now.loc[ser_now["Fecha"] <= today, "noches"].sum()) if not ser_now.empty else 0
            gap = nights_by_today - target_by_today
            if gap >= 0:
                alerts.append(("🟢 Objetivo de ocupación: en línea", f"+{gap:.0f} noches vs objetivo al día de hoy"))
            elif gap >= -0.1*target_by_today:
                alerts.append(("🟠 Objetivo de ocupación: riesgo moderado", f"{gap:.0f} noches vs objetivo al día de hoy"))
            else:
                alerts.append(("🔴 Objetivo de ocupación: riesgo alto", f"{gap:.0f} noches vs objetivo al día de hoy"))
        else:
            if k_now["occ"] >= target_occ_pct:
                alerts.append(("🟢 Ocupación final ≥ objetivo", f"{k_now['occ']:.1f}% vs {target_occ_pct:.0f}%"))
            else:
                alerts.append(("🔴 Ocupación final < objetivo", f"{k_now['occ']:.1f}% vs {target_occ_pct:.0f}%"))
    return alerts

# =============================
# Componentes gráficos (Altair)
# =============================
def line_chart_df(df: pd.DataFrame, x: str, y: str, color: str|None=None, title:str=""):
    base = alt.Chart(df).encode(x=alt.X(x, type="temporal")).properties(height=280)
    if color:
        return base.mark_line().encode(y=y, color=color).properties(title=title)
    return base.mark_line().encode(y=y).properties(title=title)

def bar_chart_df(df: pd.DataFrame, x: str, y: str, title:str=""):
    return alt.Chart(df).mark_bar().encode(x=x, y=y).properties(height=280, title=title)

# =============================
# Sidebar (carga y filtros)
# =============================
st.title(TITLE)
st.sidebar.header("Carga de datos")
uploaded = st.sidebar.file_uploader("Sube el Excel de reservas", type=["xlsx", "xls"])
if uploaded:
    try:
        raw = load_excel(uploaded)
        st.sidebar.success("Archivo cargado correctamente.")
    except Exception as e:
        st.sidebar.error(str(e))
        raw = None
else:
    raw = None
    st.info("Sube tu Excel con la hoja 'Estado de pagos de las reservas' para comenzar.")

start_def, end_def = month_range_defaults()
st.sidebar.header("Parámetros generales")
cutoff = st.sidebar.date_input("Fecha de corte (OTB)", value=pd.Timestamp.today().date(), key="cutoff")
p_start = st.sidebar.date_input("Inicio del periodo", value=start_def, key="p_start")
p_end   = st.sidebar.date_input("Fin del periodo", value=end_def, key="p_end")

props_all = sorted(raw["Alojamiento"].dropna().unique()) if raw is not None else []
props_sel = st.sidebar.multiselect("Alojamientos (opcional)", options=props_all, default=[])
inv_total = st.sidebar.number_input("Inventario total (si lo conoces)", min_value=0, value=0, step=1,
                                    help="Si queda a 0, se usa nº de alojamientos seleccionados; si no hay selección, 1.", key="inv")

mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "Evolución por fecha de corte",
        "Pace",
        "Cuadro de mando PRO",
    ],
    key="mode_radio"
)

# =============================
# VISTA: Consulta normal
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()
    start = _ensure_dates(p_start); end = _ensure_dates(p_end); cut = _ensure_dates(cutoff)
    props = props_sel if props_sel else None

    k, ser = kpis_periodo(raw, cut, start, end, props, inv_total if inv_total>0 else None)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Ingresos", f"{k['ingresos']:,.2f} €")
    c2.metric("Noches", f"{k['noches']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{k['occ']:.1f} %")
    c4.metric("ADR", f"{k['adr']:,.2f} €")
    c5.metric("RevPAR", f"{k['revpar']:,.2f} €")

    colA, colB = st.columns(2)
    with colA:
        ch1 = line_chart_df(ser, "Fecha:T", "ingresos:Q", title="Ingresos diarios")
        st.altair_chart(ch1, use_container_width=True)
    with colB:
        ser2 = ser.copy()
        inv = (inv_total if inv_total>0 else (len(props_sel) if props_sel else 1))
        ser2["Occ_%"] = np.where(inv>0, ser2["noches"]/inv*100.0, 0.0)
        ch2 = line_chart_df(ser2, "Fecha:T", "Occ_%:Q", title="Ocupación diaria (%)")
        st.altair_chart(ch2, use_container_width=True)

    st.markdown("#### Detalle diario (últimos 60 días del periodo)")
    st.dataframe(ser.tail(60), use_container_width=True)

# =============================
# VISTA: Evolución por fecha de corte
# =============================
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()
    start = _ensure_dates(p_start); end = _ensure_dates(p_end); base_cut = _ensure_dates(cutoff)
    st.sidebar.markdown("---")
    n_cortes = st.sidebar.number_input("Nº de cortes a comparar", min_value=2, value=4, step=1)
    step_days = st.sidebar.number_input("Separación entre cortes (días)", min_value=1, value=7, step=1)

    props = props_sel if props_sel else None
    evo = evolucion_por_corte(raw, base_cut, int(n_cortes), int(step_days), start, end, props)
    if evo is None or evo.empty:
        st.warning("Sin datos para construir la evolución por corte con los parámetros indicados.")
        st.stop()

    chn = alt.Chart(evo).mark_line().encode(
        x=alt.X("Fecha:T"),
        y=alt.Y("noches:Q"),
        color=alt.Color("Corte:N")
    ).properties(height=280, title="Evolución por corte — Noches")
    st.altair_chart(chn, use_container_width=True)

    chr = alt.Chart(evo).mark_line().encode(
        x=alt.X("Fecha:T"),
        y=alt.Y("ingresos:Q"),
        color=alt.Color("Corte:N")
    ).properties(height=280, title="Evolución por corte — Ingresos")
    st.altair_chart(chr, use_container_width=True)

    st.markdown("#### Tabla (últimos 60 días del periodo)")
    st.dataframe(evo.tail(60), use_container_width=True)

# =============================
# VISTA: Pace
# =============================
elif mode == "Pace":
    if raw is None:
        st.stop()
    start = _ensure_dates(p_start); end = _ensure_dates(p_end)
    props = props_sel if props_sel else None

    pace = build_pace(raw, start, end, props, years_back=3)
    if pace is None or pace.empty:
        st.warning("No hay suficiente histórico para calcular la curva Pace.")
        st.stop()

    pdata = pace.melt(id_vars=["D"], value_vars=[c for c in ["F25","F50","F75","F_now"] if c in pace.columns],
                      var_name="Serie", value_name="F")
    pdata["F_pct"] = pdata["F"] * 100.0
    ch = alt.Chart(pdata).mark_line().encode(
        x=alt.X("D:Q", title="Días hasta la estancia (lead D)"),
        y=alt.Y("F_pct:Q", title="Fracción acumulada (%)"),
        color="Serie:N"
    ).properties(title="Curva Pace — Históricos vs Actual", height=320)
    st.altair_chart(ch, use_container_width=True)

    merged = pace.dropna(subset=["F50", "F_now"]).copy()
    if not merged.empty:
        diff_pp = (merged["F_now"] - merged["F50"]).mean() * 100.0
        st.metric("Índice de ritmo vs mediana histórica", f"{diff_pp:+.1f} pp")
    st.dataframe(pace.tail(30), use_container_width=True)

    st.markdown("---")
    by_channel = st.checkbox("Ver Pace por canal (Portal)", value=False, help="Curvas F(D) por canal con índice de ritmo por canal.")
    if by_channel:
        pace_ch = build_pace_by_channel(raw, start, end, props, years_back=3)
        if pace_ch is None or pace_ch.empty:
            st.info("No hay suficiente histórico por canal para construir Pace por canal.")
        else:
            idx_tbl = pace_ch.dropna(subset=["idx_pp"]).drop_duplicates(subset=["Canal","idx_pp"])[["Canal","idx_pp"]].sort_values("idx_pp", ascending=False)
            idx_tbl["idx_pp"] = idx_tbl["idx_pp"].map(lambda v: f"{v:+.1f} pp")
            st.markdown("#### Índice de ritmo por canal (media de F_now − F50)")
            st.dataframe(idx_tbl, use_container_width=True)

            plot_df = []
            for canal in pace_ch["Canal"].dropna().unique():
                sub = pace_ch[pace_ch["Canal"]==canal].copy()
                s1 = sub.dropna(subset=["F50"])[["D"]].copy(); s1["F"] = sub.dropna(subset=["F50"])["F50"].values; s1["Serie"]="Mediana histórica"; s1["Canal"]=canal
                s2 = sub.dropna(subset=["F_now"])[["D"]].copy(); s2["F"] = sub.dropna(subset=["F_now"])["F_now"].values; s2["Serie"]="Actual"; s2["Canal"]=canal
                plot_df.append(s1); plot_df.append(s2)
            if plot_df:
                plot_df = pd.concat(plot_df, ignore_index=True)
                plot_df["F_pct"] = plot_df["F"] * 100.0
                ch2 = alt.Chart(plot_df).mark_line().encode(
                    x=alt.X("D:Q", title="Lead D"),
                    y=alt.Y("F_pct:Q", title="Fracción acumulada (%)"),
                    color=alt.Color("Canal:N"),
                    strokeDash="Serie:N"
                ).properties(height=320, title="Pace por canal — Actual vs Mediana histórica")
                st.altair_chart(ch2, use_container_width=True)

# =============================
# VISTA: Cuadro de mando PRO
# =============================
elif mode == "Cuadro de mando PRO":
    if raw is None:
        st.stop()
    start = _ensure_dates(p_start); end = _ensure_dates(p_end); cut = _ensure_dates(cutoff)
    st.sidebar.markdown("---")
    props_all = sorted(raw["Alojamiento"].dropna().unique())
    prop_focus = st.sidebar.selectbox("Alojamiento (opcional)", options=["— Todos —"] + props_all, index=0)
    focus_sel = None if prop_focus == "— Todos —" else [prop_focus]

    (k_now, ser_now), (k_prev, ser_prev), deltas, (last_start, last_end, last_cut) = yoy_compare(
        raw, cut, start, end, focus_sel, inv_total if inv_total>0 else None
    )
    pace_focus = build_pace(raw, start, end, focus_sel, years_back=3)

    st.subheader("KPIs del periodo")
    cols = st.columns(5)
    cols[0].metric("Ingresos", f"{k_now['ingresos']:,.2f} €", f"{deltas['Δ ingresos %']:+.1f}% YoY")
    cols[1].metric("Noches", f"{k_now['noches']:,}".replace(",", "."), f"{deltas['Δ noches %']:+.1f}% YoY")
    cols[2].metric("Ocupación", f"{k_now['occ']:.1f} %", f"{deltas['Δ Occ pp']:+.1f} pp")
    cols[3].metric("ADR", f"{k_now['adr']:,.2f} €", f"{deltas['Δ ADR %']:+.1f}% YoY")
    cols[4].metric("RevPAR", f"{k_now['revpar']:,.2f} €", f"{deltas['Δ RevPAR %']:+.1f}% YoY")

    st.markdown("#### Tendencias comparadas (Actual vs Año anterior)")
    ser_prev_aligned = ser_prev.copy()
    ser_prev_aligned["Fecha_align"] = ser_prev_aligned["Fecha"] + pd.DateOffset(years=1)
    ser_now2 = ser_now.rename(columns={"Fecha":"Fecha_align"}).copy()
    g1 = alt.Chart(pd.concat([ser_now2.assign(serie="Actual"), ser_prev_aligned.assign(serie="Año anterior")])
                   ).mark_line().encode(
        x=alt.X("Fecha_align:T", title="Fecha"),
        y=alt.Y("ingresos:Q", title="Ingresos €"),
        color="serie:N"
    ).properties(height=260, title="Ingresos diarios (Actual vs Año anterior)")
    st.altair_chart(g1, use_container_width=True)

    st.markdown("#### Recomendaciones y análisis")
    for ln in narrativa_pro(k_now, k_prev, pace_focus):
        st.write("- " + ln)

    if pace_focus is not None and not pace_focus.dropna(subset=["F50","F_now"]).empty:
        merged = pace_focus.dropna(subset=["F50","F_now"]).copy()
        diff_pp = (merged["F_now"] - merged["F50"]).mean() * 100.0
        st.info(f"**Índice de ritmo vs mediana histórica:** {diff_pp:+.1f} pp")

    st.markdown("---")
    target_occ = st.slider("Objetivo de ocupación del periodo (%)", 50, 100, 80, step=5, help="Se evalúa el progreso frente al objetivo según el avance del periodo.")
    alerts = compute_alerts_pro(k_now, k_prev, ser_now, ser_prev, pace_focus, target_occ, start, end, (inv_total if inv_total>0 else (len(focus_sel) if focus_sel else 1)))
    st.markdown("#### Alertas")
    if alerts:
        ccols = st.columns(min(3, len(alerts)))
        for i, (title, detail) in enumerate(alerts):
            with ccols[i % len(ccols)]:
                st.write(f"**{title}**")
                st.caption(detail)
    else:
        st.info("Sin alertas destacables para el periodo y objetivo definidos.")

    st.markdown("#### Detalle diario (últimos 60 días)")
    st.dataframe(ser_now.tail(60), use_container_width=True)
