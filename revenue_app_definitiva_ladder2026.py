
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date, datetime, timedelta

st.set_page_config(page_title="Revenue Definitivo", layout="wide")

# ---- Utilidad de reinicio seguro (evita errores de nodos huérfanos en Streamlit)
def _reset_app():
    try:
        st.cache_data.clear()
    except Exception:
        pass
    try:
        st.cache_resource.clear()
    except Exception:
        pass
    try:
        # Limpiar estado de widgets
        for k in list(st.session_state.keys()):
            del st.session_state[k]
    except Exception:
        pass
    st.experimental_rerun()

st.sidebar.button("🔄 Reiniciar app", on_click=_reset_app, help="Limpia caché y estado de la sesión y vuelve a ejecutar.")

TITLE = "📊 App Definitiva de Revenue"
SHEET_NAME = "Estado de pagos de las reservas"
REQUIRED_COLS = ["Alojamiento", "Portal", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]

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

def expand_nightly(df: pd.DataFrame) -> pd.DataFrame:
    x = df.dropna(subset=["Fecha entrada", "Fecha salida", "Precio"]).copy()
    x["Fecha entrada"] = pd.to_datetime(x["Fecha entrada"]).dt.normalize()
    x["Fecha salida"]  = pd.to_datetime(x["Fecha salida"]).dt.normalize()
    x["Fecha alta"]    = pd.to_datetime(x["Fecha alta"]).dt.normalize()
    x["los"] = (x["Fecha salida"] - x["Fecha entrada"]).dt.days
    x = x[x["los"] > 0].copy()
    if x.empty:
        return pd.DataFrame(columns=["Alojamiento","Portal","Fecha","Fecha alta","ADR_dia","DOW"])
    x["adr_reserva"] = x["Precio"] / x["los"]
    x = x.reset_index(drop=True)
    los_vals = x["los"].astype(int).to_numpy()
    rep = np.repeat(np.arange(len(x)), los_vals)
    offsets = np.concatenate([np.arange(n, dtype="int64") for n in los_vals]) if len(los_vals) else np.array([], dtype="int64")
    base = x.loc[rep, ["Alojamiento", "Portal", "Fecha entrada", "Fecha salida", "Fecha alta", "adr_reserva"]].copy()
    base["Fecha"] = (base["Fecha entrada"] + pd.to_timedelta(offsets, unit="D")).values
    try:
        base["DOW"] = pd.to_datetime(base["Fecha"]).dt.day_name(locale="es_ES")
    except Exception:
        base["DOW"] = pd.to_datetime(base["Fecha"]).dt.day_name()
    base.rename(columns={"adr_reserva": "ADR_dia"}, inplace=True)
    return base[["Alojamiento", "Portal", "Fecha", "Fecha alta", "ADR_dia", "DOW"]]

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
    if pace_df is not None and not pace_df.dropna(subset=["F50","F_now"]).empty:
        merged = pace_df.dropna(subset=["F50","F_now"]).copy()
        diff_pp = float((merged["F_now"] - merged["F50"]).mean() * 100.0)
        if diff_pp <= -5: alerts.append(("🔴 Ritmo lento", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp <= -2: alerts.append(("🟠 Ritmo algo lento", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp >= +5: alerts.append(("🟢 Ritmo rápido", f"{diff_pp:+.1f} pp vs mediana"))
        elif diff_pp >= +2: alerts.append(("🟢 Ritmo algo rápido", f"{diff_pp:+.1f} pp vs mediana"))
        else: alerts.append(("🟡 Ritmo en línea", f"{diff_pp:+.1f} pp vs mediana"))
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

def line_chart_df(df: pd.DataFrame, x: str, y: str, color: str|None=None, title:str=""):
    base = alt.Chart(df).encode(x=alt.X(x, type="temporal")).properties(height=280)
    if color:
        return base.mark_line().encode(y=y, color=color).properties(title=title)
    return base.mark_line().encode(y=y).properties(title=title)

def bar_chart_df(df: pd.DataFrame, x: str, y: str, title:str=""):
    return alt.Chart(df).mark_bar().encode(x=x, y=y).properties(height=280, title=title)

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
    # =============================
    # 🔹 Ladder 2026 (tarifas sugeridas)
    # =============================
    with st.expander("💡 Ladder 2026 (tarifas sugeridas)", expanded=False):
        st.caption("Planificación de tarifas 2026 con histórico 2025. Define temporadas via calendario editable y genera precios sugeridos por día.")

        st.markdown("**Calendario de temporadas**")
        cal_file = st.file_uploader("Sube un CSV con columnas: start_date,end_date,season (Alta|Media|Baja)", type=["csv"], key="ladder_cal_csv")
        st.caption("Ejemplo: 2026-07-01,2026-08-31,Alta  |  2026-04-01,2026-06-15,Media  |  2026-01-01,2026-03-31,Baja")
        def load_calendar(csv_file):
            if csv_file is None:
                return pd.DataFrame(columns=["start_date","end_date","season"])
            cal = pd.read_csv(csv_file)
            for c in ["start_date","end_date","season"]:
                if c not in cal.columns:
                    st.error(f"Falta columna '{c}' en el calendario de temporadas.")
                    return pd.DataFrame(columns=["start_date","end_date","season"])
            cal["start_date"] = pd.to_datetime(cal["start_date"], errors="coerce").dt.normalize()
            cal["end_date"]   = pd.to_datetime(cal["end_date"], errors="coerce").dt.normalize()
            cal["season"]     = cal["season"].astype(str).str.strip().str.title()
            return cal

        cal_df = load_calendar(cal_file)
        if cal_df.empty:
            st.info("Puedes descargar una plantilla CSV desde el botón de abajo y ajustarla.")
            tpl = pd.DataFrame([
                {"start_date":"2026-01-01","end_date":"2026-02-29","season":"Baja"},
                {"start_date":"2026-03-01","end_date":"2026-06-15","season":"Media"},
                {"start_date":"2026-06-16","end_date":"2026-09-10","season":"Alta"},
                {"start_date":"2026-09-11","end_date":"2026-10-31","season":"Media"},
                {"start_date":"2026-11-01","end_date":"2026-12-20","season":"Baja"},
                {"start_date":"2026-12-21","end_date":"2026-12-31","season":"Alta"},
            ])
            buf = io.BytesIO(); tpl.to_csv(buf, index=False); st.download_button("📥 Descargar plantilla de calendario (CSV)", buf.getvalue(), "temporadas_2026_template.csv", "text/csv")

        st.markdown("**Parámetros de tarificación**")
        publish_date = st.date_input("Fecha de publicación (para cálculo de lead)", value=pd.Timestamp.today().date(), key="ladder_pub")
        pctl_low, pctl_high = st.slider("Percentiles ADR objetivo 2025 (por DOW)",
                                        min_value=30, max_value=80, value=(50, 65), step=1,
                                        help="El ADR objetivo 2026 se estima del rango de percentiles aplicado al 2025.")
        manual_adr = st.number_input("ADR objetivo manual (opcional, sobreescribe el calculado)", min_value=0.0, value=0.0, step=1.0, key="ladder_manual_adr",
                                     help="Déjalo a 0 para usar ADR objetivo automático por DOW.")

        def season_for_day(day, cal_table):
            if cal_table is None or cal_table.empty:
                return "Media"
            m = cal_table[(cal_table["start_date"] <= day) & (cal_table["end_date"] >= day)]
            if m.empty:
                return "Media"
            s = m.iloc[0]["season"]
            return s if s in ["Alta","Media","Baja"] else "Media"

        def dow_bucket(day):
            dow = day.weekday()
            return "Fin de semana (Vie-Sa)" if dow in (4,5) else "Entre semana (Do-Jue)"

        def lead_bin(lead):
            x = int(lead)
            if x >= 120: return "120+"
            if 90 <= x <= 119: return "90-119"
            if 60 <= x <= 89:  return "60-89"
            if 30 <= x <= 59:  return "30-59"
            if 21 <= x <= 29:  return "21-29"
            if 14 <= x <= 20:  return "14-20"
            if 7  <= x <= 13:  return "7-13"
            if 3  <= x <= 6:   return "3-6"
            return "0-2"

        LADDERS = {
            "Alta": {
                "Entre semana (Do-Jue)": {"base":1.00, "floor":0.95, "cap":1.60,
                    "bins":{"120+":1.05,"90-119":1.05,"60-89":1.06,"30-59":1.08,"21-29":1.10,"14-20":1.12,"7-13":1.15,"3-6":1.18,"0-2":1.22}},
                "Fin de semana (Vie-Sa)": {"base":1.08, "floor":0.95, "cap":1.60,
                    "bins":{"120+":1.05,"90-119":1.05,"60-89":1.06,"30-59":1.08,"21-29":1.10,"14-20":1.12,"7-13":1.15,"3-6":1.18,"0-2":1.22}},
            },
            "Media": {
                "Entre semana (Do-Jue)": {"base":0.97, "floor":0.90, "cap":1.45,
                    "bins":{"120+":1.00,"90-119":1.00,"60-89":1.02,"30-59":1.04,"21-29":1.06,"14-20":1.08,"7-13":1.10,"3-6":1.12,"0-2":1.15}},
                "Fin de semana (Vie-Sa)": {"base":1.03, "floor":0.90, "cap":1.45,
                    "bins":{"120+":1.00,"90-119":1.00,"60-89":1.02,"30-59":1.04,"21-29":1.06,"14-20":1.08,"7-13":1.10,"3-6":1.12,"0-2":1.15}},
            },
            "Baja": {
                "Entre semana (Do-Jue)": {"base":0.92, "floor":0.85, "cap":1.30,
                    "bins":{"120+":0.96,"90-119":0.96,"60-89":0.98,"30-59":1.00,"21-29":1.02,"14-20":1.03,"7-13":1.05,"3-6":1.07,"0-2":1.10}},
                "Fin de semana (Vie-Sa)": {"base":0.98, "floor":0.85, "cap":1.30,
                    "bins":{"120+":0.96,"90-119":0.96,"60-89":0.98,"30-59":1.00,"21-29":1.02,"14-20":1.03,"7-13":1.05,"3-6":1.07,"0-2":1.10}},
            },
        }

        def adr_objetivo_por_dow_2026(df, start_26, end_26, props_sel):
            start_25 = (pd.to_datetime(start_26) - pd.DateOffset(years=1)).normalize()
            end_25   = (pd.to_datetime(end_26)   - pd.DateOffset(years=1)).normalize()
            cut_25   = pd.Timestamp.today().normalize()
            k25, ser25 = kpis_periodo(df, cut_25, start_25, end_25, props_sel, None)
            if ser25.empty or (ser25["noches"].sum() == 0):
                return {}
            ser25 = ser25.copy()
            ser25["ADR_2025"] = np.where(ser25["noches"]>0, ser25["ingresos"]/ser25["noches"], np.nan)
            ser25["DOW"] = ser25["Fecha"].dt.weekday
            adr_map = {}
            for dow in range(7):
                vals = ser25.loc[ser25["DOW"]==dow, "ADR_2025"].dropna()
                if len(vals)==0:
                    continue
                lo = np.nanpercentile(vals, pctl_low)
                hi = np.nanpercentile(vals, pctl_high)
                adr_map[dow] = float((lo+hi)/2.0)
            return adr_map

        pace_for_period = build_pace(raw, start, end, focus_sel, years_back=3)
        pace_adj_pct = 0.0
        pace_note = "Sin ajuste pace"
        if pace_for_period is not None and not pace_for_period.dropna(subset=["F50","F_now"]).empty:
            merged = pace_for_period.dropna(subset=["F50","F_now"]).copy()
            diff_pp = float((merged["F_now"] - merged["F50"]).mean() * 100.0)
            if diff_pp <= -5: pace_adj_pct, pace_note = -0.05, f"Pace lento ({diff_pp:+.1f} pp)"
            elif diff_pp <= -2: pace_adj_pct, pace_note = -0.03, f"Pace algo lento ({diff_pp:+.1f} pp)"
            elif diff_pp >= +5: pace_adj_pct, pace_note = +0.07, f"Pace rápido ({diff_pp:+.1f} pp)"
            elif diff_pp >= +2: pace_adj_pct, pace_note = +0.04, f"Pace algo rápido ({diff_pp:+.1f} pp)"
            else: pace_adj_pct, pace_note = 0.0, f"Pace en línea ({diff_pp:+.1f} pp)"

        idx_days = pd.date_range(start, end, freq="D")
        adr_map = adr_objetivo_por_dow_2026(raw, start, end, focus_sel)

        rows = []
        for day in idx_days:
            dowb = dow_bucket(day)
            season = "Media"
            if cal_df is not None and not cal_df.empty:
                season = season_for_day(day, cal_df)
            lead = (pd.to_datetime(day).date() - publish_date).days
            lead = max(0, lead)
            lbin = lead_bin(lead)
            if manual_adr and manual_adr > 0:
                adr_obj = float(manual_adr)
            else:
                dow_idx = day.weekday()
                adr_obj = float(adr_map.get(dow_idx, np.nan))
            ladder = LADDERS.get(season, LADDERS["Media"]).get(dowb, LADDERS["Media"]["Entre semana (Do-Jue)"])
            base_mult = ladder["base"]
            bin_mult = ladder["bins"].get(lbin, 1.00)
            price_pre = adr_obj * base_mult * bin_mult if not np.isnan(adr_obj) else np.nan
            price_adj = price_pre * (1.0 + pace_adj_pct) if not np.isnan(price_pre) else np.nan
            floor = adr_obj * ladder["floor"] if not np.isnan(adr_obj) else np.nan
            cap   = adr_obj * ladder["cap"]   if not np.isnan(adr_obj) else np.nan
            price_final = price_adj
            if not np.isnan(price_final):
                if not np.isnan(floor): price_final = max(price_final, floor)
                if not np.isnan(cap):   price_final = min(price_final, cap)
            note = f"{season}, {dowb}, lead {lbin} · ADR obj={adr_obj:.0f} × base {base_mult:.2f} × bin {bin_mult:.2f}"
            if pace_adj_pct != 0.0: note += f" × pace {pace_adj_pct*100:+.0f}% ({pace_note})"
            note += f" → floor {ladder['floor']*100:.0f}% / cap {ladder['cap']*100:.0f}%"
            rows.append({
                "Fecha": day.date(),
                "Día": day.day_name() if hasattr(day, "day_name") else str(day.weekday()),
                "Lead": lead,
                "Temporada": season,
                "DOW bucket": dowb,
                "ADR objetivo": None if np.isnan(adr_obj) else round(adr_obj, 2),
                "Multiplicador ladder": round(base_mult * bin_mult, 3),
                "Ajuste pace %": round(pace_adj_pct*100, 1),
                "Floor": None if np.isnan(floor) else round(floor, 2),
                "Cap": None if np.isnan(cap) else round(cap, 2),
                "Precio sugerido": None if np.isnan(price_final) else round(price_final, 2),
                "Notas": note
            })
        ladder_df = pd.DataFrame(rows)
        st.dataframe(ladder_df, use_container_width=True, hide_index=True)

        start_25 = (pd.to_datetime(start) - pd.DateOffset(years=1)).normalize()
        end_25   = (pd.to_datetime(end)   - pd.DateOffset(years=1)).normalize()
        cut_25   = pd.Timestamp.today().normalize()
        k25, ser25 = kpis_periodo(raw, cut_25, start_25, end_25, focus_sel, None)
        ser25 = ser25.copy()
        if not ser25.empty and ser25["noches"].sum()>0:
            ser25["ADR_2025"] = np.where(ser25["noches"]>0, ser25["ingresos"]/ser25["noches"], np.nan)
            ser25["Fecha_align"] = ser25["Fecha"] + pd.DateOffset(years=1)
            g_hist = ser25[["Fecha_align","ADR_2025"]].rename(columns={"Fecha_align":"Fecha"})
        else:
            g_hist = pd.DataFrame(columns=["Fecha","ADR_2025"])
        g_obj = ladder_df[["Fecha","ADR objetivo"]].copy().rename(columns={"ADR objetivo":"ADR_obj_2026"})
        g_price = ladder_df[["Fecha","Precio sugerido"]].copy().rename(columns={"Precio sugerido":"Precio_sugerido_2026"})
        plot_df = g_price.merge(g_obj, on="Fecha", how="left").merge(g_hist, on="Fecha", how="left")
        plot_df["Fecha"] = pd.to_datetime(plot_df["Fecha"])
        lines = plot_df.melt(id_vars=["Fecha"], value_vars=[c for c in ["Precio_sugerido_2026","ADR_obj_2026","ADR_2025"] if c in plot_df.columns],
                             var_name="Serie", value_name="€")
        lines = lines.dropna(subset=["€"])
        if not lines.empty:
            ch_prices = alt.Chart(lines).mark_line().encode(
                x=alt.X("Fecha:T"),
                y=alt.Y("€:Q"),
                color="Serie:N"
            ).properties(height=280, title="Comparativa: ADR 2025 vs ADR objetivo 2026 vs Precio sugerido Ladder 2026")
            st.altair_chart(ch_prices, use_container_width=True)
        csv_bytes = ladder_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar tabla (CSV)", data=csv_bytes, file_name="ladder_2026_sugerencias.csv", mime="text/csv")
        try:
            import xlsxwriter
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="xlsxwriter") as writer:
                ladder_df.to_excel(writer, index=False, sheet_name="Ladder_2026")
            st.download_button("📥 Descargar tabla (Excel)", data=xbuf.getvalue(), file_name="ladder_2026_sugerencias.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            st.caption("Para exportar a Excel instala `xlsxwriter`. El CSV funciona en cualquier caso.")

    st.dataframe(ser_now.tail(60), use_container_width=True)
