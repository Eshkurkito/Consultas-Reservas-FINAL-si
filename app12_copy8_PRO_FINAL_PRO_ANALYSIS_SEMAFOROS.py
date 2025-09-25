# ===========================
# BLOQUE 1/5 — Núcleo & Utils
# ===========================
import io
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional, Dict
from pandas.io.formats.style import Styler


import os   
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# Utilidades comunes
# ---------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave y tipos."""
    required = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]
    for col in required:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"], errors="coerce")
    df["Fecha salida"] = pd.to_datetime(df["Fecha salida"], errors="coerce")
    df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    df["Alquiler con IVA (€)"] = pd.to_numeric(df["Alquiler con IVA (€)"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def load_excel_from_blobs(file_blobs: List[tuple[str, bytes]]) -> pd.DataFrame:
    """Carga y concatena varios Excel a partir de blobs (nombre, bytes)."""
    frames = []
    for name, data in file_blobs:
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            sheet = (
                "Estado de pagos de las reservas"
                if "Estado de pagos de las reservas" in xls.sheet_names
                else xls.sheet_names[0]
            )
            df = pd.read_excel(xls, sheet_name=sheet)
            df["__source_file__"] = name
            frames.append(df)
        except Exception as e:
            st.error(f"No se pudo leer {name}: {e}")
            st.stop()
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    return parse_dates(df_all)

def get_inventory(df: pd.DataFrame, override: Optional[int]) -> int:
    inv = df["Alojamiento"].nunique()
    if override and override > 0:
        inv = int(override)
    return int(inv)

def help_block(kind: str):
    """Bloque de ayuda contextual por sección."""
    texts = {
        "Consulta normal": """
**Qué es:** KPIs del periodo elegido **a la fecha de corte**.
- *Noches ocupadas*: noches del periodo dentro de reservas con **Fecha alta ≤ corte**.
- *Noches disponibles*: inventario × nº de días del periodo (puedes **sobrescribir inventario**).
- *Ocupación %* = Noches ocupadas / Noches disponibles.
- *Ingresos* = precio prorrateado por noche dentro del periodo.
- *ADR* = Ingresos / Noches ocupadas.
- *RevPAR* = Ingresos / Noches disponibles.
""",
        "KPIs por meses": """
**Qué es:** Serie por **meses** con KPIs a la **misma fecha de corte**.
""",
        "Evolución por corte": """
**Qué es:** Cómo **crecen** los KPIs del mismo periodo cuando **mueves la fecha de corte**.
""",
        "Pickup": """
**Qué es:** Diferencia entre dos cortes A y B (**B – A**) en el mismo periodo.
""",
        "Pace": """
**Qué es:** KPI confirmado a **D días antes de la estancia** (D=0 día de llegada).
""",
        "Predicción": """
**Qué es:** Forecast por Pace con banda **[P25–P75]** de noches finales y semáforo de pickup.
""",
        "Lead": "Lead time = días entre Alta y Entrada; LOS = noches por reserva.",
        "DOW": "Calor por Día de la Semana × Mes: Noches, %, ADR.",
        "ADR bands": "Percentiles P10/P25/P50/P75/P90 del ADR por reserva (por mes).",
        "Calendario": "Matriz Alojamiento × Día (ocupado/ADR por noche).",
        "Resumen": "Vista compacta + simulador.",
        "Estacionalidad": "Distribución por Mes, DOW o Día del mes.",
    }
    txt = texts.get(kind, None)
    if txt:
        with st.expander("ℹ️ Cómo leer esta sección", expanded=False):
            st.markdown(txt)

def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> tuple[date, date]:
    """Date inputs que pueden sincronizarse con un periodo global (si keep_period está activo)."""
    keep = st.session_state.get("keep_period", False)
    g_start = st.session_state.get("global_period_start")
    g_end = st.session_state.get("global_period_end")
    val_start = g_start if (keep and g_start) else default_start
    val_end = g_end if (keep and g_end) else default_end
    c1, c2 = st.columns(2)
    with c1:
        start_val = st.date_input(label_start, value=val_start, key=f"{key_prefix}_start")
    with c2:
        end_val = st.date_input(label_end, value=val_end, key=f"{key_prefix}_end")
    if keep:
        st.session_state["global_period_start"] = start_val
        st.session_state["global_period_end"] = end_val
    return start_val, end_val

def occurrences_of_dow_by_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    days = pd.date_range(start, end, freq='D')
    df = pd.DataFrame({"Fecha": days})
    df["Mes"] = df["Fecha"].dt.to_period('M').astype(str)
    df["DOW"] = df["Fecha"].dt.weekday.map({0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"})
    occ = df.groupby(["DOW","Mes"]).size().reset_index(name="occ")
    return occ


# ===== CDM PRO ANALYSIS (Kai upgrade with semaphores) =====
def _kai_pct(cur, ly):
    try:
        if ly in (0, None) or pd.isna(ly) or ly == 0:
            return np.nan
        return (cur - ly) / ly
    except Exception:
        return np.nan

def _kai_fmt_pct(x, decimals=1, pos_good=True):
    if x is None or pd.isna(x):
        return "-"
    val = float(x)
    if pos_good:
        icon = "🟢" if val > 0.01 else ("🔴" if val < -0.01 else "🟠")
    else:
        icon = "🔴" if val > 0.01 else ("🟢" if val < -0.01 else "🟠")
    return f"{icon} {val:.{decimals}%}"

def _kai_cdm_pro_analysis(tot_now: dict, tot_ly_cut: dict, tot_ly_final: dict, pace: dict, price_ref_p50: float|None) -> str:
    """
    Análisis PRO con:
      • 🚦 Semáforo global (estado del periodo).
      • 📈 Análisis técnico (KPIs vs LY + atribución RevPAR).
      • 🧠 Explicación ejecutiva narrada dentro de un Streamlit expander.
    Devuelve un bloque técnico (string) y pinta en pantalla la explicación ejecutiva.
    """
    import math

    # ---------- Utilidades locales ----------
    def _pct(a, b):
        try:
            if b in (0, None) or pd.isna(b) or b == 0:
                return np.nan
            return (a - b) / b
        except Exception:
            return np.nan

    def _fmt_pct(x, good_up=True, d=1):
        if x is None or pd.isna(x):
            return "—"
        v = float(x)
        icon = "🟢" if (v > 0.01 if good_up else v < -0.01) else ("🔴" if (v < -0.01 if good_up else v > 0.01) else "🟠")
        return f"{icon} {v:.{d}%}"

    def _fmt_money(x, decimals=0):
        try:
            f = f"{{:,.{decimals}f}} €".format(float(x)).replace(",", ".")
            return f
        except:
            return "—"

    def _dlog(a, b):
        try:
            if a <= 0 or b <= 0 or pd.isna(a) or pd.isna(b):
                return np.nan
            return math.log(a / b)
        except Exception:
            return np.nan

    def _abs_pct(x):
        if x is None or pd.isna(x):
            return "—"
        return f"{float(x)*100:+.1f}%"

    # ---------- KPIs actuales vs LY ----------
    occ_now    = float(tot_now.get("ocupacion_pct", np.nan))
    adr_now    = float(tot_now.get("adr", np.nan))
    revpar_now = float(tot_now.get("revpar", np.nan))
    ing_now    = float(tot_now.get("ingresos", np.nan))

    occ_ly    = float(tot_ly_cut.get("ocupacion_pct", np.nan))
    adr_ly    = float(tot_ly_cut.get("adr", np.nan))
    revpar_ly = float(tot_ly_cut.get("revpar", np.nan))
    ing_ly    = float(tot_ly_cut.get("ingresos", np.nan))

    ing_ly_final = float(tot_ly_final.get("ingresos", np.nan))

    d_occ    = _pct(occ_now,    occ_ly)
    d_adr    = _pct(adr_now,    adr_ly)
    d_revpar = _pct(revpar_now, revpar_ly)
    d_ing    = _pct(ing_now,    ing_ly)

    falta_ingresar_vs_LYfinal = (ing_ly_final - ing_now) if (np.isfinite(ing_ly_final) and np.isfinite(ing_now)) else np.nan
    gap_ing_final             = _pct(ing_now, ing_ly_final)

    # ---------- Atribución RevPAR (log) ----------
    dlog_occ = _dlog(occ_now, occ_ly)
    dlog_adr = _dlog(adr_now, adr_ly)
    contrib_occ_pp = (dlog_occ*100) if np.isfinite(dlog_occ) else np.nan
    contrib_adr_pp = (dlog_adr*100) if np.isfinite(dlog_adr) else np.nan

    # ---------- 🚦 Semáforo global ----------
    coverage_p50 = np.nan
    lt_state = None
    if isinstance(pace, dict):
        need_p50   = pace.get("pickup_typ_p50", np.nan)
        adr_tail   = pace.get("adr_tail_p50", np.nan)
        nights_otb = pace.get("nights_otb", np.nan)

        # Cobertura del gap (escenario P50) si existe gap>0
        if np.isfinite(need_p50) and np.isfinite(adr_tail) and np.isfinite(falta_ingresar_vs_LYfinal) and (falta_ingresar_vs_LYfinal > 0):
            extra_p50 = need_p50 * adr_tail
            coverage_p50 = extra_p50 / falta_ingresar_vs_LYfinal

        # Estado de lead time
        if np.isfinite(need_p50) and np.isfinite(nights_otb):
            if nights_otb >= need_p50 * 1.10:
                lt_state = "adelantado"
            elif nights_otb <= need_p50 * 0.90:
                lt_state = "retrasado"
            else:
                lt_state = "linea"

    score = 0
    if np.isfinite(d_revpar):
        score += 2 if d_revpar > 0.02 else (-2 if d_revpar < -0.02 else 0)
    if np.isfinite(coverage_p50):
        score += 2 if coverage_p50 >= 1.0 else (1 if coverage_p50 >= 0.7 else -2)
    if lt_state is not None:
        score += 1 if lt_state == "adelantado" else (-1 if lt_state == "retrasado" else 0)
    if np.isfinite(falta_ingresar_vs_LYfinal):
        score += 1 if falta_ingresar_vs_LYfinal <= 0 else 0

    if score >= 2:
        semaforo_global = "🟢 Bien encaminado"
    elif score <= -2:
        semaforo_global = "🔴 En riesgo"
    else:
        semaforo_global = "🟠 Neutro/mixto"

    # ---------- 📈 Bloque técnico (devuelto como string) ----------
    lines = []
    lines.append(f"### 🚦 Semáforo global: {semaforo_global}")
    lines.append("### 📈 Análisis técnico PRO")
    lines.append(f"- Ocupación vs LY: {_fmt_pct(d_occ)}")
    lines.append(f"- ADR vs LY: {_fmt_pct(d_adr)}")
    lines.append(f"- RevPAR vs LY: {_fmt_pct(d_revpar)}")
    lines.append(f"- Ingresos vs LY: {_fmt_pct(d_ing)}")
    if np.isfinite(contrib_occ_pp) and np.isfinite(contrib_adr_pp):
        lines.append(f"- Atribución RevPAR (p.p.): Ocupación {contrib_occ_pp:+.1f} · ADR {contrib_adr_pp:+.1f}")

    # ---------- 🧠 Explicación ejecutiva (narrada en expander) ----------
    explicacion = []

    # Veredicto general
    if np.isfinite(d_adr) and np.isfinite(d_occ):
        if d_adr < -0.05 and d_occ > 0.0:
            veredicto = "🟠 Estamos comprando volumen barato"
            razon = f"El ADR cayó { _abs_pct(d_adr) }, y aunque la ocupación subió { _abs_pct(d_occ) }, el resultado neto es negativo."
            conclusion = "Vendemos más noches, pero cada noche vale bastante menos: la mejora de volumen no compensa la pérdida de precio."
        elif d_adr > 0.05 and d_occ < 0.0:
            veredicto = "🟠 Precio alto con penalización"
            razon = f"Subimos ADR { _abs_pct(d_adr) }, pero la ocupación cayó { _abs_pct(d_occ) }."
            conclusion = "El precio está penalizando la demanda: más caro, pero menos volumen."
        elif d_adr < -0.05 and d_occ < -0.05:
            veredicto = "🔴 Problema de demanda"
            razon = "Baja el ADR y baja la ocupación."
            conclusion = "No basta con tocar precios: necesitamos activar más demanda."
        else:
            veredicto = "🟢 Balance razonable"
            razon = "Los movimientos de precio y ocupación están equilibrados."
            conclusion = "Estamos en línea con el mercado."
    else:
        veredicto, razon, conclusion = "🟠 Datos insuficientes", "", ""

    explicacion.append(f"**👉 Veredicto general:** {veredicto}")
    if razon:      explicacion.append(f"- {razon}")
    if conclusion: explicacion.append(f"- {conclusion}")

    # Evolución vs LY
    explicacion.append("")
    explicacion.append("**👉 Evolución frente al LY (a fecha de corte):**")
    explicacion.append(f"- Ocupación: {_fmt_pct(d_occ)}")
    explicacion.append(f"- ADR: {_fmt_pct(d_adr)}")
    explicacion.append(f"- RevPAR: {_fmt_pct(d_revpar)}")
    explicacion.append(f"- Ingresos: {_fmt_pct(d_ing)}")

    # Qué explica el resultado (atribución)
    if np.isfinite(contrib_occ_pp) and np.isfinite(contrib_adr_pp):
        explicacion.append("")
        explicacion.append("**👉 Qué explica el resultado (atribución RevPAR):**")
        explicacion.append(f"- Ocupación {contrib_occ_pp:+.1f} p.p.")
        explicacion.append(f"- ADR {contrib_adr_pp:+.1f} p.p.")
        explicacion.append("→ El peso del precio medio es el principal driver del resultado.")

    # Viabilidad de cierre del gap
    explicacion.append("")
    explicacion.append("**👉 Viabilidad de cierre del gap:**")
    if np.isfinite(falta_ingresar_vs_LYfinal) and falta_ingresar_vs_LYfinal > 0:
        explicacion.append(f"- Faltan { _fmt_money(falta_ingresar_vs_LYfinal) } para igualar el cierre LY.")
        if np.isfinite(coverage_p50):
            cov_txt = f"{coverage_p50*100:.0f}%"
            explicacion.append(f"- Con pickup P50 y ADR tail, cobertura estimada del gap ≈ {cov_txt}.")
        explicacion.append("→ Aunque toquemos más el precio, la elasticidad no cubriría sola el gap: hay que **activar demanda**.")
    elif np.isfinite(falta_ingresar_vs_LYfinal) and falta_ingresar_vs_LYfinal <= 0:
        explicacion.append("- 🟢 Ya superamos los ingresos del LY a esta fecha.")
    else:
        explicacion.append("- — No se pudo calcular el gap con fiabilidad.")

    # Plan de acción
    explicacion.append("")
    explicacion.append("**👉 Plan de acción (siguiente quincena):**")
    if veredicto.startswith("🟠 Estamos comprando volumen"):
        explicacion.append("- Revisar y retirar descuentos de baja conversión.")
        explicacion.append("- Micro-rebajas **quirúrgicas** en días valle (LT corto).")
        explicacion.append("- Mantener precios en fines de semana/eventos (picos).")
        explicacion.append("- Boost de demanda: visibilidad OTAs, campañas directas, partners.")
    elif veredicto.startswith("🟠 Precio alto"):
        explicacion.append("- Ajustes selectivos en días flojos; proteger picos.")
        explicacion.append("- Vigilar conversión y cancelaciones semanalmente.")
        explicacion.append("- Activar acciones de demanda en ventanas con pickup exigente.")
    elif veredicto.startswith("🔴 Problema de demanda"):
        explicacion.append("- Activar demanda: campañas flash, partners, mejorar visibilidad.")
        explicacion.append("- Relajar mínimos/restricciones que bloqueen reservas.")
    else:
        explicacion.append("- Mantener estrategia actual y monitorizar pickup semanal.")

    # Pintar narrativa en Streamlit (expander)
    exp_text = "\n".join(explicacion)
    st.markdown("### 🧠 Explicación ejecutiva (narrada)")
    with st.expander("Ver análisis detallado", expanded=True):
        st.markdown(exp_text)

    # Devolvemos el bloque técnico (lo que se muestra con st.markdown fuera)
    return "\n".join(lines)


# ===== END CDM PRO ANALYSIS (Kai upgrade with semaphores) =====
# ---------------------------
# Motor de KPIs & series
# ---------------------------

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    """KPIs vectorizados sin expandir noche a noche."""
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv * days

    if df_cut.empty:
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": noches_disponibles,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    arr_e = df_cut["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df_cut["Fecha salida"].values.astype('datetime64[ns]')

    total_nights = ((arr_s - arr_e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    ov_days = np.clip(ov_days, 0, None)

    price = df_cut["Alquiler con IVA (€)"].values.astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df_cut["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }
    return by_prop, tot

def compute_portal_share(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    filter_props: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """Distribución por portal sobre reservas que intersectan el periodo a la fecha de corte."""
    if "Agente/Intermediario" not in df_all.columns:
        return None

    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df = df[df["Alojamiento"].isin(filter_props)]
    df = df.dropna(subset=["Fecha entrada", "Fecha salida", "Agente/Intermediario"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Agente/Intermediario", "Reservas", "% Reservas"]) 

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    arr_e = df["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df["Fecha salida"].values.astype('datetime64[ns]')

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    mask = ov_days > 0
    if mask.sum() == 0:
        return pd.DataFrame(columns=["Agente/Intermediario", "Reservas", "% Reservas"]) 

    df_sel = df.loc[mask]
    counts = df_sel.groupby("Agente/Intermediario").size().reset_index(name="Reservas").sort_values("Reservas", ascending=False)
    total = counts["Reservas"].sum()
    counts["% Reservas"] = np.where(total > 0, counts["Reservas"] / total * 100.0, 0.0)
    return counts

def daily_series(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], inventory_override: Optional[int]) -> pd.DataFrame:
    """Serie diaria: noches, ingresos, ocupación %, ADR, RevPAR."""
    days = list(pd.date_range(start, end, freq='D'))
    rows = []
    for d in days:
        _bp, tot = compute_kpis(
            df_all=df_all,
            cutoff=cutoff,
            period_start=d,
            period_end=d,
            inventory_override=inventory_override,
            filter_props=props,
        )
        rows.append({"Fecha": d.normalize(), **tot})
    return pd.DataFrame(rows)

def build_calendar_matrix(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], mode: str = "Ocupado/Libre") -> pd.DataFrame:
    """Matriz (alojamientos × días) con '■' si ocupado o ADR por noche si mode='ADR'."""
    df_cut = df_all[(df_all["Fecha alta"] <= cutoff)].copy()
    if props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"])
    if df_cut.empty:
        return pd.DataFrame()

    rows = []
    for _, r in df_cut.iterrows():
        e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Alquiler con IVA (€)"])
        ov_start = max(e, start)
        ov_end = min(s, end + pd.Timedelta(days=1))
        n_nights = (s - e).days
        if ov_start >= ov_end or n_nights <= 0:
            continue
        adr_night = p / n_nights if n_nights > 0 else 0.0
        for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
            rows.append({"Alojamiento": r["Alojamiento"], "Fecha": d.normalize(), "Ocupado": 1, "ADR_noche": adr_night})
    if not rows:
        return pd.DataFrame()
    df_nightly = pd.DataFrame(rows)

    if mode == "Ocupado/Libre":
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="Ocupado", aggfunc='sum', fill_value=0)
        piv = piv.applymap(lambda x: '■' if x > 0 else '')
    else:
        piv = df_nightly.pivot_table(index="Alojamiento", columns="Fecha", values="ADR_noche", aggfunc='mean', fill_value='')
    piv = piv.reindex(sorted(piv.columns), axis=1)
    return piv

def pace_series(df_all: pd.DataFrame, period_start: pd.Timestamp, period_end: pd.Timestamp, d_max: int, props: Optional[List[str]], inv_override: Optional[int]) -> pd.DataFrame:
    """Curva Pace: para cada D (0..d_max), noches/ingresos confirmados a D días antes de la estancia."""
    df = df_all.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"]).copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    if df.empty:
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))

    e = df["Fecha entrada"].values.astype('datetime64[ns]')
    s = df["Fecha salida"].values.astype('datetime64[ns]')
    c = df["Fecha alta"].values.astype('datetime64[ns]')
    price = df["Alquiler con IVA (€)"].values.astype('float64')

    total_nights = ((s - e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)
    adr_night = np.where(total_nights > 0, price / total_nights, 0.0)

    ov_start = np.maximum(e, start_ns)
    ov_end = np.minimum(s, end_excl_ns)
    valid = (ov_end > ov_start) & (total_nights > 0)
    if not valid.any():
        inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
        if inv_override and inv_override > 0:
            inv = int(inv_override)
        days = (period_end - period_start).days + 1
        return pd.DataFrame({"D": list(range(d_max + 1)), "noches": 0, "ingresos": 0.0, "ocupacion_pct": 0.0, "adr": 0.0, "revpar": 0.0})

    e = e[valid]; s = s[valid]; c = c[valid]; ov_start = ov_start[valid]; ov_end = ov_end[valid]; adr_night = adr_night[valid]

    D_vals = np.arange(0, d_max + 1, dtype='int64')
    D_td = D_vals * one_day

    start_thr = c[:, None] + D_td[None, :]
    ov_start_b = np.maximum(ov_start[:, None], start_thr)
    nights_D = ((ov_end[:, None] - ov_start_b) / one_day).astype('int64')
    nights_D = np.clip(nights_D, 0, None)

    nights_series = nights_D.sum(axis=0).astype(float)
    ingresos_series = (nights_D * adr_night[:, None]).sum(axis=0)

    inv = len(set(props)) if props else df_all["Alojamiento"].nunique()
    if inv_override and inv_override > 0:
        inv = int(inv_override)
    days = (period_end - period_start).days + 1
    disponibles = inv * days if days > 0 else 0

    occ_series = (nights_series / disponibles * 100.0) if disponibles > 0 else np.zeros_like(nights_series)
    adr_series = np.where(nights_series > 0, ingresos_series / nights_series, 0.0)
    revpar_series = (ingresos_series / disponibles) if disponibles > 0 else np.zeros_like(ingresos_series)

    return pd.DataFrame({
        "D": D_vals,
        "noches": nights_series,
        "ingresos": ingresos_series,
        "ocupacion_pct": occ_series,
        "adr": adr_series,
        "revpar": revpar_series,
    })

def pace_profiles_for_refs(df: pd.DataFrame, target_start: pd.Timestamp, target_end: pd.Timestamp, ref_years: int, dmax: int, props: Optional[List[str]] = None, inv_override: Optional[int] = None) -> dict:
    """Perfiles F(D) P25/50/75 a partir de años de referencia (mismo mes)."""
    profiles = []
    for k in range(1, ref_years+1):
        s = target_start - pd.DateOffset(years=k)
        e = target_end - pd.DateOffset(years=k)
        base = pace_series(df, s, e, dmax, props, inv_override)
        if base.empty or base['noches'].max() == 0:
            continue
        final_n = base.loc[base['D']==0, 'noches'].values[0]
        if final_n <= 0:
            continue
        F = base['noches'] / final_n
        profiles.append(F.values)
    if not profiles:
        F = np.linspace(0.2, 1.0, dmax+1)
        return {"F25": F, "F50": F, "F75": F}
    M = np.vstack(profiles)
    F25 = np.nanpercentile(M, 25, axis=0)
    F50 = np.nanpercentile(M, 50, axis=0)
    F75 = np.nanpercentile(M, 75, axis=0)
    return {"F25": F25, "F50": F50, "F75": F75}

def pace_forecast_month(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, ref_years: int = 2, dmax: int = 180, props: Optional[List[str]] = None, inv_override: Optional[int] = None) -> dict:
    """Forecast por Pace (P25/50/75), ADR tail y pickup típico/nec."""
    daily = daily_series(df, pd.to_datetime(cutoff), start, end, props, inv_override).sort_values('Fecha')

    D_day = (daily['Fecha'] - pd.to_datetime(cutoff)).dt.days.clip(lower=0)
    dmax = int(max(dmax, D_day.max())) if len(D_day) else dmax

    prof = pace_profiles_for_refs(df, start, end, ref_years, dmax, props, inv_override)
    F25, F50, F75 = prof['F25'], prof['F50'], prof['F75']

    def f_at(arr, d):
        d = int(min(max(d, 0), len(arr)-1))
        return float(arr[d]) if not np.isnan(arr[d]) else 1.0

    eps = 1e-6
    daily['D'] = D_day
    daily['F25'] = daily['D'].apply(lambda d: f_at(F25, d))
    daily['F50'] = daily['D'].apply(lambda d: f_at(F50, d))
    daily['F75'] = daily['D'].apply(lambda d: f_at(F75, d))
    daily['n_final_p25'] = daily['noches_ocupadas'] / daily['F25'].clip(lower=eps)
    daily['n_final_p50'] = daily['noches_ocupadas'] / daily['F50'].clip(lower=eps)
    daily['n_final_p75'] = daily['noches_ocupadas'] / daily['F75'].clip(lower=eps)

    nights_otb = float(daily['noches_ocupadas'].sum())
    nights_p25 = float(daily['n_final_p25'].sum())
    nights_p50 = float(daily['n_final_p50'].sum())
    nights_p75 = float(daily['n_final_p75'].sum())

    _, tot_now = compute_kpis(df, pd.to_datetime(cutoff), start, end, inv_override, props)
    adr_otb = float(tot_now['adr'])
    rev_otb = float(tot_now['ingresos'])

    D_med = int(np.median(D_day)) if len(D_day) else 0
    tail_adrs, tail_nights, finals_hist = [], [], []
    for k in range(1, ref_years+1):
        s = start - pd.DateOffset(years=k)
        e = end - pd.DateOffset(years=k)
        base = pace_series(df, s, e, max(D_med, 0), props, inv_override)
        if base.empty or 0 not in base['D'].values:
            continue
        nights_final = float(base.loc[base['D']==0, 'noches'].values[0])
        rev_final = float(base.loc[base['D']==0, 'ingresos'].values[0])
        finals_hist.append(nights_final)
        if D_med in base['D'].values:
            nights_atD = float(base.loc[base['D']==D_med, 'noches'].values[0])
            rev_atD = float(base.loc[base['D']==D_med, 'ingresos'].values[0])
        else:
            nights_atD = float('nan'); rev_atD = float('nan')
        dn = max(nights_final - (nights_atD if np.isfinite(nights_atD) else 0.0), 0.0)
        dr = max(rev_final - (rev_atD if np.isfinite(rev_atD) else 0.0), 0.0)
        if dn > 0:
            tail_adrs.append(dr/dn)
            tail_nights.append(dn)

    if tail_adrs:
        adr_tail_p25 = float(np.percentile(tail_adrs, 25))
        adr_tail_p50 = float(np.percentile(tail_adrs, 50))
        adr_tail_p75 = float(np.percentile(tail_adrs, 75))
    else:
        adr_tail_p25 = adr_tail_p50 = adr_tail_p75 = adr_otb

    if tail_nights and finals_hist and np.median(finals_hist) > 0:
        scale = nights_p50 / float(np.median(finals_hist))
        pickup_typ_p50 = float(np.percentile(tail_nights, 50)) * scale
        pickup_typ_p75 = float(np.percentile(tail_nights, 75)) * scale
    else:
        pickup_typ_p50 = max(nights_p50 - nights_otb, 0.0)
        pickup_typ_p75 = max(nights_p25 - nights_otb, 0.0)

    nights_rem_p50 = max(nights_p50 - nights_otb, 0.0)
    revenue_final_p50 = rev_otb + adr_tail_p50 * nights_rem_p50
    adr_final_p50 = revenue_final_p50 / nights_p50 if nights_p50 > 0 else 0.0

    pickup_needed_p50 = nights_rem_p50

    return {
        "nights_otb": nights_otb,
        "nights_p25": nights_p25,
        "nights_p50": nights_p50,
        "nights_p75": nights_p75,
        "adr_final_p50": adr_final_p50,
        "revenue_final_p50": revenue_final_p50,
        "adr_tail_p25": adr_tail_p25,
        "adr_tail_p50": adr_tail_p50,
        "adr_tail_p75": adr_tail_p75,
        "pickup_needed_p50": pickup_needed_p50,
        "pickup_typ_p50": pickup_typ_p50,
        "pickup_typ_p75": pickup_typ_p75,
        "daily": daily,
        "n_refs": len(finals_hist),
    }
# =============================
# HELPERS – Eventos / ADR base / m_apto / Calendario
# =============================

EVENTS_CSV_PATH = "eventos_festivos.csv"

@st.cache_data(show_spinner=False)
def load_events_csv(path: str) -> pd.DataFrame:
    """Carga CSV de eventos, normaliza columnas y tipajes."""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            # normaliza nombres
            rename = {}
            cols_lower = {c.lower().strip(): c for c in df.columns}
            for want, candidates in {
                "fecha_inicio": ["fecha_inicio","fecha inicio","inicio","start","start_date"],
                "fecha_fin": ["fecha_fin","fecha fin","fin","end","end_date"],
                "uplift_pct": ["uplift_pct","uplift","pct","porcentaje","porcentaje_aumentar"],
                "nombre": ["nombre","evento","event","descripcion","desc"],
                "prioridad": ["prioridad","priority","prio"],
            }.items():
                if want not in df.columns:
                    for lc, orig in cols_lower.items():
                        if lc in candidates:
                            rename[orig] = want
                            break
            if rename:
                df = df.rename(columns=rename)

            for col in ["fecha_inicio","fecha_fin","uplift_pct"]:
                if col not in df.columns:
                    df[col] = None
            if "nombre" not in df.columns:
                df["nombre"] = ""
            if "prioridad" not in df.columns:
                df["prioridad"] = 1

            df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce").dt.date
            df["fecha_fin"]   = pd.to_datetime(df["fecha_fin"], errors="coerce").dt.date
            df["uplift_pct"]  = pd.to_numeric(df["uplift_pct"], errors="coerce")
            df["prioridad"]   = pd.to_numeric(df["prioridad"], errors="coerce").fillna(1).astype(int)
            df = df.dropna(subset=["fecha_inicio","fecha_fin","uplift_pct"])
            return df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"No pude leer {path}: {e}. Empezamos vacío.")
    return pd.DataFrame(columns=["fecha_inicio","fecha_fin","uplift_pct","nombre","prioridad"])

def save_events_csv(df: pd.DataFrame, path: str):
    out = df.copy()
    out["fecha_inicio"] = pd.to_datetime(out["fecha_inicio"]).dt.date
    out["fecha_fin"]    = pd.to_datetime(out["fecha_fin"]).dt.date
    out.to_csv(path, index=False)

def expand_events_by_day(events_df: pd.DataFrame) -> pd.DataFrame:
    """Expande rangos a filas por día con uplift.
    Si hay solapes, gana mayor 'prioridad'; si empatan, mayor 'uplift_pct'."""
    if events_df.empty:
        return pd.DataFrame(columns=["fecha","uplift_pct","origen","prioridad"])
    rows = []
    for _, r in events_df.iterrows():
        fi, ff = r["fecha_inicio"], r["fecha_fin"]
        if pd.isna(fi) or pd.isna(ff):
            continue
        if fi > ff:
            fi, ff = ff, fi
        days = pd.date_range(pd.to_datetime(fi), pd.to_datetime(ff), freq="D")
        for d in days:
            rows.append({
                "fecha": d.normalize().date(),
                "uplift_pct": float(r["uplift_pct"]) if pd.notna(r["uplift_pct"]) else 0.0,
                "origen": str(r.get("nombre","")).strip() or "Evento",
                "prioridad": int(r.get("prioridad",1)) if pd.notna(r.get("prioridad",1)) else 1,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["fecha","prioridad","uplift_pct"], ascending=[True, False, False])
    df = df.groupby("fecha", as_index=False).first()
    return df

def adr_bands_p50_for_month_by_apto(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    props: List[str],
) -> Dict[str, float]:
    """{alojamiento: P50 ADR_reserva} dentro del periodo seleccionado."""
    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Alquiler con IVA (€)"] / df["los"]
    mask = ~((df["Fecha salida"] <= start) | (df["Fecha entrada"] >= (end + pd.Timedelta(days=1))))
    df = df[mask]
    if df.empty:
        return {}
    out: Dict[str, float] = {}
    for aloj, sub in df.groupby("Alojamiento"):
        arr = sub["adr_reserva"].dropna().values
        if arr.size:
            out[aloj] = float(np.percentile(arr, 50))
    return out

def adr_bands_p50_for_month(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    props: List[str],
) -> float:
    """P50 ADR_reserva del grupo dentro del periodo (una sola cifra)."""
    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Alquiler con IVA (€)"] / df["los"]
    mask = ~((df["Fecha salida"] <= start) | (df["Fecha entrada"] >= (end + pd.Timedelta(days=1))))
    df = df[mask]
    if df.empty or not df["adr_reserva"].notna().any():
        return np.nan
    return float(np.percentile(df["adr_reserva"].values, 50))

def compute_m_apto_by_property(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,       # fecha de corte actual
    start: pd.Timestamp,        # rango actual (lo trasladamos a LY)
    end: pd.Timestamp,
    props: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    m_apto = ADR_P50_apto_LY / ADR_P50_grupo_LY, exige >=3 reservas por apto.
    """
    cut_ly = pd.to_datetime(cutoff) - pd.DateOffset(years=1)
    start_ly = pd.to_datetime(start) - pd.DateOffset(years=1)
    end_ly = pd.to_datetime(end) - pd.DateOffset(years=1)

    df = df_all[(df_all["Fecha alta"] <= cut_ly)].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"])

    mask = ~((df["Fecha salida"] <= start_ly) | (df["Fecha entrada"] >= (end_ly + pd.Timedelta(days=1))))
    df = df[mask]
    if df.empty:
        return {}

    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Alquiler con IVA (€)"] / df["los"]

    arr_group = df["adr_reserva"].dropna().values
    if arr_group.size == 0:
        return {}

    p50_group_ly = np.percentile(arr_group, 50)
    if not np.isfinite(p50_group_ly) or p50_group_ly <= 0:
        return {}

    out: Dict[str, float] = {}
    for aloj, sub in df.groupby("Alojamiento"):
        arr = sub["adr_reserva"].dropna().values
        if arr.size >= 3:
            p50_apto_ly = np.percentile(arr, 50)
            if np.isfinite(p50_apto_ly) and p50_apto_ly > 0:
                out[aloj] = float(p50_apto_ly / p50_group_ly)
    return out

# ---------- Calendario de precios (grid + estilos)

def build_pricing_calendar_grid(
    result_df: pd.DataFrame,
    eventos_daily: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Devuelve:
      - grid_wide: DataFrame wide (Alojamiento x Fecha) con precios (float)
      - meta_cols: DataFrame con metadatos por columna Fecha:
          - is_weekend (bool), is_event (bool), event_name (str)
    """
    if result_df is None or result_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = result_df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
    grid_wide = df.pivot_table(
        index="Alojamiento",
        columns="Fecha",
        values="Precio propuesto",
        aggfunc="mean"
    ).sort_index(axis=1)

    # metadatos columna (día)
    cols = pd.Series(grid_wide.columns)
    is_weekend = cols.dt.weekday.isin([5, 6])

    is_event = pd.Series(False, index=grid_wide.columns)
    event_name = pd.Series("", index=grid_wide.columns)
    if eventos_daily is not None and not eventos_daily.empty:
        ev = eventos_daily.copy()
        ev["fecha"] = pd.to_datetime(ev["fecha"]).dt.normalize()
        ev = ev.drop_duplicates(subset=["fecha"]).set_index("fecha")
        aligned = ev.reindex(grid_wide.columns)
        is_event = aligned["uplift_pct"].notna().fillna(False)
        event_name = aligned["origen"].fillna("")

    meta_cols = pd.DataFrame({
        "Fecha": grid_wide.columns,
        "is_weekend": is_weekend.values,
        "is_event": is_event.values,
        "event_name": event_name.values,
    }).set_index("Fecha")

    return grid_wide, meta_cols


def style_pricing_calendar(grid_wide: pd.DataFrame, meta_cols: pd.DataFrame):
    """
    Aplica estilos:
      - Finde: gris suave
      - Evento: amarillo suave (si coincide con finde, más intenso)
      - NaN: gris claro
    """
    if grid_wide.empty:
        return grid_wide.style

    COLOR_WEEKEND = "#f2f2f2"
    COLOR_EVENT   = "#fff3cd"   # amarillo suave
    COLOR_BOTH    = "#ffe8a1"   # más intenso si coincide
    COLOR_NAN     = "#fafafa"

    styles = pd.DataFrame("", index=grid_wide.index, columns=grid_wide.columns)

    # Fondo por día
    for col in grid_wide.columns:
        weekend = bool(meta_cols.loc[col, "is_weekend"]) if col in meta_cols.index else False
        event   = bool(meta_cols.loc[col, "is_event"])   if col in meta_cols.index else False
        bg = ""
        if weekend and event:
            bg = f"background-color: {COLOR_BOTH};"
        elif event:
            bg = f"background-color: {COLOR_EVENT};"
        elif weekend:
            bg = f"background-color: {COLOR_WEEKEND};"
        if bg:
            styles[col] = bg

    # NaN -> gris claro + texto apagado
    nan_mask = grid_wide.isna()
    styles = styles.mask(nan_mask, f"background-color: {COLOR_NAN}; color: #999;")

    styler = grid_wide.style.format("{:.2f}")
    styler = styler.set_table_styles([
        {"selector": "th.col_heading", "props": [("white-space", "nowrap")]},
        {"selector": "th.row_heading", "props": [("white-space", "nowrap")]},
    ])
    styler = styler.set_properties(**{"white-space": "nowrap"})
    styler = styler.apply(lambda _: styles, axis=None)

    # Tooltips de evento
    if "event_name" in meta_cols.columns and meta_cols["event_name"].astype(str).str.len().gt(0).any():
        tooltips = pd.DataFrame("", index=grid_wide.index, columns=grid_wide.columns)
        for col in grid_wide.columns:
            name = meta_cols.loc[col, "event_name"] if col in meta_cols.index else ""
            if isinstance(name, str) and name:
                tooltips[col] = name
        try:
            styler = styler.set_tooltips(tooltips)
        except Exception:
            pass

    return styler

# Mapa nombres UI -> columnas
METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

# ===========================
# BLOQUE 2/5 — Sidebar + Menú + Consulta normal
# ===========================

# Config básica de página (si no la pusiste arriba)
st.set_page_config(page_title="Consultas OTB & Dashboard", layout="wide")
st.title("📊 OTB Analytics – KPIs & Dashboard")
st.caption("Sube tus Excel una vez, configura parámetros en la barra lateral y usa cualquiera de los modos.")

# -------- Sidebar: periodo global + ficheros + targets --------
with st.sidebar:
    st.checkbox(
        "🧲 Mantener periodo entre modos",
        value=st.session_state.get("keep_period", False),
        key="keep_period",
        help="Si está activo, el periodo (inicio/fin) se guarda y se reutiliza en todos los modos."
    )
    colp1, colp2 = st.columns(2)
    with colp1:
        if st.button("Reset periodo"):
            st.session_state.pop("global_period_start", None)
            st.session_state.pop("global_period_end", None)
            st.success("Periodo global reiniciado")
    with colp2:
        if st.session_state.get("keep_period"):
            st.caption(
                f"Periodo actual: {st.session_state.get('global_period_start','–')} → {st.session_state.get('global_period_end','–')}"
            )

    st.header("Archivos de trabajo (persisten en la sesión)")
    files_master = st.file_uploader(
        "Sube uno o varios Excel",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="files_master",
        help="Se admiten múltiples años (2024, 2025…). Hoja esperada: 'Estado de pagos de las reservas'.",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Usar estos archivos", type="primary"):
            if files_master:
                blobs = [(f.name, f.getvalue()) for f in files_master]
                df_loaded = load_excel_from_blobs(blobs)
                st.session_state["raw_df"] = df_loaded
                st.session_state["file_names"] = [n for n, _ in blobs]
                st.success(f"Cargados {len(blobs)} archivo(s)")
            else:
                st.warning("No seleccionaste archivos.")
    with col_b:
        if st.button("Limpiar archivos"):
            st.session_state.pop("raw_df", None)
            st.session_state.pop("file_names", None)
            st.info("Archivos eliminados de la sesión.")

# Targets opcionales
with st.sidebar.expander("🎯 Cargar Targets (opcional)"):
    tgt_file = st.file_uploader("CSV Targets", type=["csv"], key="tgt_csv")
    if tgt_file is not None:
        try:
            df_tgt = pd.read_csv(tgt_file)
            # Columnas esperadas si las tienes: year, month, target_occ_pct, target_adr, target_revpar, target_nights, target_revenue
            st.session_state["targets_df"] = df_tgt
            st.success("Targets cargados en sesión.")
        except Exception as e:
            st.error(f"No se pudo leer el CSV: {e}")

raw = st.session_state.get("raw_df")
file_names = st.session_state.get("file_names", [])

if raw is not None:
    with st.expander("📂 Archivos cargados"):
        st.write("**Lista:**", file_names)
        st.write(f"**Alojamientos detectados:** {raw['Alojamiento'].nunique()}")
else:
    st.info("Sube archivos en la barra lateral y pulsa **Usar estos archivos** para empezar.")

# ---------------- Menú de modos ----------------
# --- MENÚ FINAL (sustituye el anterior) ---
mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "Resumen Comparativo",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pace (curva D)",
        "Lead time & LOS",
        "Cohortes (Alta × Estancia)",
        "Estacionalidad",
        "Ranking alojamientos",
        "Calidad de datos",
        "Cuadro de mando (PRO)",

    ],
    key="mode_radio"
)

# Mapa métrica UI -> columna (definido también arriba, lo reusamos sin problema)
METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

# =============================
# Vista: Consulta normal
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date.today(), key="cutoff_normal")
        c1, c2 = st.columns(2)
        start_normal, end_normal = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            # valores por defecto sensatos (cámbialos si quieres otro periodo por defecto)
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "normal"
        )
        inv_normal = st.number_input(
            "Sobrescribir inventario (nº alojamientos)",
            min_value=0, value=0, step=1, key="inv_normal"
        )
        props_normal = st.multiselect(
            "Filtrar alojamientos (opcional)",
            options=sorted(raw["Alojamiento"].unique()),
            default=[],
            key="props_normal"
        )
        st.markdown("—")
        compare_normal = st.checkbox(
            "Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal"
        )
        inv_normal_prev = st.number_input(
            "Inventario año anterior (opcional)",
            min_value=0, value=0, step=1, key="inv_normal_prev"
        )

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    help_block("Consulta normal")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")

    # Distribución por portal (si existe columna)
    port_df = compute_portal_share(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        filter_props=props_normal if props_normal else None,
    )
    st.subheader("Distribución por portal (reservas en el periodo)")
    if port_df is None:
        st.info("No se encontró la columna 'Portal'. Si tiene otro nombre, dímelo y lo mapeo.")
    elif port_df.empty:
        st.warning("No hay reservas del periodo a la fecha de corte para calcular distribución por portal.")
    else:
        port_view = port_df.copy()
        port_view["% Reservas"] = port_view["% Reservas"].round(2)
        st.dataframe(port_view, use_container_width=True)
        csv_port = port_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar distribución por portal (CSV)",
            data=csv_port,
            file_name="portales_distribucion.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Descargar detalle (CSV)",
            data=csv,
            file_name="detalle_por_alojamiento.csv",
            mime="text/csv"
        )

    # Comparativa YoY opcional
    if compare_normal:
        cutoff_cmp = (pd.to_datetime(cutoff_normal) - pd.DateOffset(years=1)).date()
        start_cmp = (pd.to_datetime(start_normal) - pd.DateOffset(years=1)).date()
        end_cmp = (pd.to_datetime(end_normal) - pd.DateOffset(years=1)).date()
        _bp_c, total_cmp = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_cmp),
            period_start=pd.to_datetime(start_cmp),
            period_end=pd.to_datetime(end_cmp),
            inventory_override=int(inv_normal_prev) if inv_normal_prev > 0 else None,
            filter_props=props_normal if props_normal else None,
        )
        st.markdown("**Comparativa con año anterior** (corte y periodo -1 año)")
        d1, d2, d3 = st.columns(3)
        d4, d5, d6 = st.columns(3)
        d1.metric(
            "Noches ocupadas (prev)",
            f"{total_cmp['noches_ocupadas']:,}".replace(",", "."),
            delta=total_n['noches_ocupadas']-total_cmp['noches_ocupadas']
        )
        d2.metric(
            "Noches disp. (prev)",
            f"{total_cmp['noches_disponibles']:,}".replace(",", "."),
            delta=total_n['noches_disponibles']-total_cmp['noches_disponibles']
        )
        d3.metric(
            "Ocupación (prev)",
            f"{total_cmp['ocupacion_pct']:.2f}%",
            delta=f"{total_n['ocupacion_pct']-total_cmp['ocupacion_pct']:.2f}%"
        )
        d4.metric(
            "Ingresos (prev)", f"{total_cmp['ingresos']:.2f}",
            delta=f"{total_n['ingresos']-total_cmp['ingresos']:.2f}"
        )
        d5.metric(
            "ADR (prev)", f"{total_cmp['adr']:.2f}",
            delta=f"{total_n['adr']-total_cmp['adr']:.2f}"
        )
        d6.metric(
            "RevPAR (prev)", f"{total_cmp['revpar']:.2f}",
            delta=f"{total_n['revpar']-total_cmp['revpar']:.2f}"
        )
# ---------- Resumen comparativo (por alojamiento) ----------
# ---------- Resumen comparativo (por alojamiento) ----------
elif mode == "Resumen comparativo":
    if raw is None:
        st.warning("⚠️ No hay datos cargados. Sube tus Excel y pulsa **Usar estos archivos** en la barra lateral.")
        st.stop()

    with st.sidebar:
        st.header("Parámetros – Resumen comparativo")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cut_resumen_comp")
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "resumen_comp"
        )
        props_rc = st.multiselect(
            "Alojamientos (opcional)",
            options=sorted(raw["Alojamiento"].unique()),
            default=[],
            key="props_resumen_comp"
        )

    st.subheader("📊 Resumen comparativo por alojamiento")

    # Pequeño panel diagnóstico
    _n_props = (len(props_rc) if props_rc else raw["Alojamiento"].nunique())
    st.caption(f"Periodo: **{pd.to_datetime(start_rc).date()} → {pd.to_datetime(end_rc).date()}** · "
               f"Corte: **{pd.to_datetime(cutoff_rc).date()}** · "
               f"Alojamientos en cálculo: **{_n_props}**")

    # Días del periodo (para ocupación por apto = noches / días)
    days_period = (pd.to_datetime(end_rc) - pd.to_datetime(start_rc)).days + 1
    if days_period <= 0:
        st.error("El periodo no es válido (fin anterior o igual al inicio). Ajusta fechas.")
        st.stop()

    def _by_prop_with_occ(cutoff_dt, start_dt, end_dt, props_sel=None):
        by_prop, _ = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_dt),
            period_start=pd.to_datetime(start_dt),
            period_end=pd.to_datetime(end_dt),
            inventory_override=None,
            filter_props=props_sel if props_sel else None,
        )
        # compute_kpis ya devuelve: Alojamiento | Noches ocupadas | Ingresos | ADR  (por alojamiento)
        # Calculamos ocupación por alojamiento asumiendo 1 unidad por apto: noches/días * 100
        if by_prop.empty:
            return pd.DataFrame(columns=["Alojamiento","ADR","Ocupación %","Ingresos"])
        out = by_prop.copy()
        out["Ocupación %"] = (out["Noches ocupadas"] / days_period * 100.0).astype(float)
        return out[["Alojamiento","ADR","Ocupación %","Ingresos"]]

    props_sel = props_rc if props_rc else None

    # Actual
    now_df = _by_prop_with_occ(cutoff_rc, start_rc, end_rc, props_sel).rename(columns={
        "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
    })

    # LY (mismo periodo y cutoff -1 año)
    ly_df = _by_prop_with_occ(
        pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    ).rename(columns={
        "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
    })

    # LY final (resultado): mismo periodo LY, pero corte = fin del periodo LY
    ly_final_df = _by_prop_with_occ(
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),  # corte = fin del periodo LY
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    )
    # De este solo necesitamos los ingresos finales
    ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    # Merge total
    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer") \
                    .merge(ly_final_df, on="Alojamiento", how="left")

    # Si todo está vacío, mostramos ayuda
    if resumen.empty:
        st.info(
            "No hay reservas que intersecten el periodo **a la fecha de corte** seleccionada.\n"
            "- Prueba a ampliar el periodo o mover la fecha de corte.\n"
            "- Recuerda que se incluyen reservas con **Fecha alta ≤ corte** y estancia dentro del periodo."
        )
        st.stop()

    # Orden columnas
    resumen = resumen.reindex(columns=[
        "Alojamiento",
        "ADR actual","ADR LY",
        "Ocupación actual %","Ocupación LY %",
        "Ingresos actuales (€)","Ingresos LY (€)",
        "Ingresos finales LY (€)"
    ])

    st.dataframe(resumen, use_container_width=True)

    # Descargas
    csv_bytes = resumen.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV", data=csv_bytes,
                       file_name="resumen_comparativo.csv", mime="text/csv")

    import io
    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
            ws = writer.sheets["Resumen"]
            # auto ancho simple
            for j, col in enumerate(resumen.columns):
                width = int(min(38, max(12, resumen[col].astype(str).str.len().max() if not resumen.empty else 12)))
                ws.set_column(j, j, width)
    except Exception:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
            # (sin autosize en openpyxl)
    st.download_button(
        "📥 Descargar Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="resumen_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===========================
# BLOQUE 3/5 — KPIs por meses, Evolución por corte, Pickup, Pace, Predicción
# ===========================

# ---------- KPIs por meses ----------
if mode == "KPIs por meses":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_m = st.date_input("Fecha de corte", value=date.today(), key="cutoff_months")
        props_m = st.multiselect("Filtrar alojamientos (opcional)",
                                 options=sorted(raw["Alojamiento"].unique()),
                                 default=[], key="props_months")
        inv_m = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_months")
        inv_m_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_months_prev")
        # Rango total de meses del dataset
        _min = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).min()
        _max = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).max()
        months_options = [str(p) for p in pd.period_range(_min.to_period("M"), _max.to_period("M"), freq="M")] if pd.notna(_min) and pd.notna(_max) else []
        selected_months_m = st.multiselect("Meses a graficar (YYYY-MM)", options=months_options, default=[], key="months_months")
        metric_choice = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"])
        compare_m = st.checkbox("Comparar con año anterior (mismo mes)", value=False, key="cmp_months")

    st.subheader("📈 KPIs por meses (a fecha de corte)")
    help_block("KPIs por meses")
    if selected_months_m:
        rows_actual, rows_prev = [], []
        for ym in selected_months_m:
            p = pd.Period(ym, freq="M")
            start_m = p.to_timestamp(how="start")
            end_m = p.to_timestamp(how="end")
            _bp, _tot = compute_kpis(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_m),
                period_start=start_m,
                period_end=end_m,
                inventory_override=int(inv_m) if inv_m > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_actual.append({"Mes": ym, **_tot})

            if compare_m:
                p_prev = p - 12
                start_prev = p_prev.to_timestamp(how="start")
                end_prev = p_prev.to_timestamp(how="end")
                cutoff_prev = pd.to_datetime(cutoff_m) - pd.DateOffset(years=1)
                _bp2, _tot_prev = compute_kpis(
                    df_all=raw,
                    cutoff=cutoff_prev,
                    period_start=start_prev,
                    period_end=end_prev,
                    inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                    filter_props=props_m if props_m else None,
                )
                rows_prev.append({"Mes": ym, **_tot_prev})

        df_actual = pd.DataFrame(rows_actual).sort_values("Mes")
        key_col = METRIC_MAP[metric_choice]
        if not compare_m:
            st.line_chart(df_actual.set_index("Mes")[[key_col]].rename(columns={key_col: metric_choice}), height=280)
            st.dataframe(df_actual[["Mes", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]]
                         .rename(columns={"noches_ocupadas": "Noches ocupadas", "noches_disponibles": "Noches disponibles",
                                          "ocupacion_pct": "Ocupación %", "adr": "ADR (€)", "revpar": "RevPAR (€)", "ingresos": "Ingresos (€)"}),
                         use_container_width=True)
        else:
            df_prev = pd.DataFrame(rows_prev).sort_values("Mes") if rows_prev else pd.DataFrame()
            plot_df = pd.DataFrame({"Actual": df_actual[key_col].values}, index=df_actual["Mes"])
            if not df_prev.empty:
                plot_df["Año anterior"] = df_prev[key_col].values
            st.line_chart(plot_df, height=280)

            table_df = df_actual.merge(df_prev, on="Mes", how="left", suffixes=("", " (prev)")) if not df_prev.empty else df_actual
            rename_map = {
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)",
                "noches_ocupadas (prev)": "Noches ocupadas (prev)",
                "noches_disponibles (prev)": "Noches disponibles (prev)",
                "ocupacion_pct (prev)": "Ocupación % (prev)",
                "adr (prev)": "ADR (€) (prev)",
                "revpar (prev)": "RevPAR (€) (prev)",
                "ingresos (prev)": "Ingresos (€) (prev)",
            }
            st.dataframe(table_df.rename(columns=rename_map), use_container_width=True)

        csvm = df_actual.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar KPIs por mes (CSV)", data=csvm, file_name="kpis_por_mes.csv", mime="text/csv")
    else:
        st.info("Selecciona meses en la barra lateral para ver la gráfica.")

# =============================
# MODO: Evolución por fecha de corte
# =============================
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date.today().replace(day=1), key="evo_cut_start")
        evo_cut_end   = st.date_input("Fin de corte",   value=date.today(), key="evo_cut_end")

        st.header("Periodo objetivo")
        evo_target_start, evo_target_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "evo_target"
        )

        props_e = st.multiselect(
            "Filtrar alojamientos (opcional)",
            options=sorted(raw["Alojamiento"].unique()),
            default=[],
            key="props_evo",
        )
        inv_e      = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")

        kpi_options = ["Ocupación %", "ADR (€)", "RevPAR (€)"]
        selected_kpis = st.multiselect("KPIs a mostrar", kpi_options, default=["Ocupación %"], key="kpi_multi")

        compare_e = st.checkbox("Mostrar LY (alineado por día)", value=False, key="cmp_evo")

        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📈 Evolución de KPIs vs fecha de corte")

    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts   = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
            st.stop()

        # ---------- Serie ACTUAL ----------
        rows_now = []
        for c in pd.date_range(cut_start_ts, cut_end_ts, freq="D"):
            _, tot = compute_kpis(
                df_all=raw,
                cutoff=c,
                period_start=pd.to_datetime(evo_target_start),
                period_end=pd.to_datetime(evo_target_end),
                inventory_override=int(inv_e) if inv_e > 0 else None,
                filter_props=props_e if props_e else None,
            )
            rows_now.append({
                "Corte": c.normalize(),
                "ocupacion_pct": float(tot["ocupacion_pct"]),
                "adr": float(tot["adr"]),
                "revpar": float(tot["revpar"]),
            })
        df_now = pd.DataFrame(rows_now)
        if df_now.empty:
            st.info("No hay datos para el rango seleccionado.")
            st.stop()

        # ---------- Serie LY (opcional) ----------
        df_prev = pd.DataFrame()
        if compare_e:
            rows_prev = []
            cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
            cut_end_prev   = cut_end_ts   - pd.DateOffset(years=1)
            target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
            target_end_prev   = pd.to_datetime(evo_target_end)   - pd.DateOffset(years=1)
            for c in pd.date_range(cut_start_prev, cut_end_prev, freq="D"):
                _, tot2 = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=target_start_prev,
                    period_end=target_end_prev,
                    inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_prev.append({
                    "Corte": (pd.to_datetime(c).normalize() + pd.DateOffset(years=1)),  # alineado al año actual
                    "ocupacion_pct": float(tot2["ocupacion_pct"]),
                    "adr": float(tot2["adr"]),
                    "revpar": float(tot2["revpar"]),
                })
            df_prev = pd.DataFrame(rows_prev)

        # ---------- Preparación long-form para graficar ----------
        # map: nombre mostrado -> (columna, tipo)
        kpi_map = {
            "Ocupación %": ("ocupacion_pct", "occ"),
            "ADR (€)":     ("adr", "eur"),
            "RevPAR (€)":  ("revpar", "eur"),
        }
        sel_items = [(k, *kpi_map[k]) for k in selected_kpis]  # [(label, col, kind)]

        def to_long(df, label_suffix="Actual"):
            out = []
            for lbl, col, kind in sel_items:
                if col in df.columns:
                    tmp = df[["Corte", col]].copy()
                    tmp["metric_label"] = lbl if label_suffix == "Actual" else f"{lbl} (LY)"
                    tmp["value"] = tmp[col].astype(float)
                    tmp["kind"] = kind
                    tmp["series"] = label_suffix
                    out.append(tmp[["Corte", "metric_label", "value", "kind", "series"]])
            return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

        long_now  = to_long(df_now, "Actual")
        long_prev = to_long(df_prev, "LY") if compare_e and not df_prev.empty else pd.DataFrame()
        long_all  = pd.concat([long_now, long_prev], ignore_index=True) if not long_prev.empty else long_now

        # ==========================
        #     G R Á F I C A S
        # ==========================
        import altair as alt

        # Selección "nearest" por X con regla vertical
        nearest = alt.selection_point(fields=["Corte"], nearest=True, on="mousemove", empty="none")

        # Eje compartido por ocupación (izquierda) y eje compartido por euros (derecha)
        def build_layer(data, kind, axis_orient="left", color_map=None, dash_ly=True):
            """Devuelve una capa con todas las métricas del tipo 'kind' ('occ' o 'eur')."""
            if data.empty:
                return None
            dfk = data[data["kind"] == kind]
            if dfk.empty:
                return None

            # Color por métrica
            _colors = color_map or {
                "Ocupación %": "#1f77b4",
                "ADR (€)": "#ff7f0e",
                "RevPAR (€)": "#2ca02c",
                "Ocupación % (LY)": "#1f77b4",
                "ADR (€) (LY)": "#ff7f0e",
                "RevPAR (€) (LY)": "#2ca02c",
            }

            # Línea + puntos pequeños siempre visibles
            line = (
                alt.Chart(dfk)
                .mark_line(strokeWidth=2, interpolate="monotone", point=alt.OverlayMarkDef(size=30, filled=True))
                .encode(
                    x=alt.X("Corte:T", title="Fecha de corte"),
                    y=alt.Y("value:Q",
                            axis=alt.Axis(orient=axis_orient,
                                          title=list(dfk["metric_label"].unique())[0].replace(" (LY)", "")
                                          if dfk["metric_label"].str.contains("Ocupación").any() and kind=="occ"
                                          else "ADR/RevPAR (€)" if kind=="eur" else list(dfk["metric_label"].unique())[0]),
                    ),
                    color=alt.Color("metric_label:N", scale=alt.Scale(domain=list(_colors.keys()),
                                                                      range=[_colors[k] for k in _colors]),
                                    legend=None),
                    detail="metric_label:N",
                    tooltip=[alt.Tooltip("Corte:T", title="Día"),
                             alt.Tooltip("metric_label:N", title="KPI"),
                             alt.Tooltip("value:Q", title="Valor", format=".2f")],
                )
            )

            # Puntos grandes al pasar el ratón (misma capa, filtrados por selección)
            pts_hover = (
                alt.Chart(dfk)
                .mark_point(size=90, filled=True)
                .encode(
                    x="Corte:T",
                    y="value:Q",
                    color=alt.Color("metric_label:N", scale=alt.Scale(domain=list(_colors.keys()),
                                                                      range=[_colors[k] for k in _colors]),
                                    legend=None),
                    detail="metric_label:N",
                )
                .transform_filter(nearest)
            )

            # Si hay series LY, las dibujamos con dash y opacidad ligera
            if " (LY)" in " ".join(dfk["metric_label"].unique()):
                line = line.encode(strokeDash=alt.condition(
                    "indexof(datum.metric_label, '(LY)') >= 0",
                    alt.value([5, 3]), alt.value([0, 0])
                ), opacity=alt.condition(
                    "indexof(datum.metric_label, '(LY)') >= 0",
                    alt.value(0.35), alt.value(1.0)
                ))

            return alt.layer(line, pts_hover)

        # Regla vertical y puntos “selectores” invisibles para que el hover sea fácil en todo el panel
        selectors = (
            alt.Chart(long_all)
            .mark_rule(opacity=0)
            .encode(x="Corte:T")
            .add_params(nearest)
        )
        vline = (
            alt.Chart(long_all)
            .mark_rule(color="#666", strokeWidth=1)
            .encode(x="Corte:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        )

        occ_selected   = any(kind == "occ" for _, _, kind in sel_items)
        euros_selected = any(kind == "eur" for _, _, kind in sel_items)

        left_layer  = build_layer(long_all, "occ", axis_orient="left")
        # Si solo hay KPIs en €, queremos un solo eje (izquierda)
        right_orient = "right" if (occ_selected and euros_selected) else "left"
        right_layer = build_layer(long_all, "eur", axis_orient=right_orient)

        layers = [selectors]
        if left_layer is not None:
            layers.append(left_layer)
        if right_layer is not None:
            layers.append(right_layer)
        layers.append(vline)

        chart = alt.layer(*layers).resolve_scale(
            y="independent" if (occ_selected and euros_selected) else "shared"
        ).properties(height=380)

        # Zoom/Pan horizontal
        zoomx = alt.selection_interval(bind="scales", encodings=["x"])
        st.altair_chart(chart.add_params(zoomx), use_container_width=True)

        # Tabla y export
        st.dataframe(df_now, use_container_width=True)
        st.download_button(
            "📥 Descargar evolución (CSV)",
            data=df_now.to_csv(index=False).encode("utf-8-sig"),
            file_name="evolucion_kpis.csv",
            mime="text/csv",
        )
    else:
        st.caption("Configura los parámetros y pulsa **Calcular evolución**.")

# ---------- Pickup (entre dos cortes) ----------
elif mode == "Pickup (entre dos cortes)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutA = st.date_input("Corte A", value=date.today() - timedelta(days=7), key="pickup_cutA")
        cutB = st.date_input("Corte B", value=date.today(), key="pickup_cutB")
        c1, c2 = st.columns(2)
        p_start, p_end = period_inputs("Inicio del periodo", "Fin del periodo",
                                       date(date.today().year, date.today().month, 1),
                                       (pd.Timestamp.today().to_period("M").end_time).date(),
                                       "pickup")
        inv_pick = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_pick")
        props_pick = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_pick")
        metric_pick = st.radio("Métrica gráfica", ["Noches", "Ingresos (€)", "Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=False)
        view_pick = st.radio("Vista", ["Diario", "Acumulado"], horizontal=True)
        topn = st.number_input("Top-N alojamientos (por pickup noches)", min_value=5, max_value=100, value=20, step=5)
        run_pick = st.button("Calcular pickup", type="primary")

    st.subheader("📈 Pickup entre cortes (B – A)")
    help_block("Pickup")
    if run_pick:
        if pd.to_datetime(cutA) > pd.to_datetime(cutB):
            st.error("Corte A no puede ser posterior a Corte B.")
        else:
            inv_override = int(inv_pick) if inv_pick > 0 else None
            # Totales A y B
            _bpA, totA = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            _bpB, totB = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            # Deltas totales
            deltas = {
                "noches": totB['noches_ocupadas'] - totA['noches_ocupadas'],
                "ingresos": totB['ingresos'] - totA['ingresos'],
                "occ_delta": totB['ocupacion_pct'] - totA['ocupacion_pct'],
                "adr_delta": totB['adr'] - totA['adr'],
                "revpar_delta": totB['revpar'] - totA['revpar'],
            }
            c1, c2, c3 = st.columns(3)
            c1.metric("Pickup Noches", f"{deltas['noches']:,}".replace(",", "."))
            c2.metric("Pickup Ingresos (€)", f"{deltas['ingresos']:.2f}")
            c3.metric("Δ Ocupación", f"{deltas['occ_delta']:.2f}%")
            c4, c5 = st.columns(2)
            c4.metric("Δ ADR", f"{deltas['adr_delta']:.2f}")
            c5.metric("Δ RevPAR", f"{deltas['revpar_delta']:.2f}")

            # Series diarias A y B
            serA = daily_series(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            serB = daily_series(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), props_pick if props_pick else None, inv_override)
            # Elegir métrica
            key_map = {"Noches": "noches_ocupadas", "Ingresos (€)": "ingresos", "Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}
            k = key_map[metric_pick]
            df_plot = serA.merge(serB, on="Fecha", suffixes=(" A", " B"))
            df_plot["Δ (B–A)"] = df_plot[f"{k} B"] - df_plot[f"{k} A"]
            if view_pick == "Acumulado":
                for col in [f"{k} A", f"{k} B", "Δ (B–A)"]:
                    df_plot[col] = df_plot[col].cumsum()
            chart_df = pd.DataFrame({
                f"A (≤ {pd.to_datetime(cutA).date()})": df_plot[f"{k} A"].values,
                f"B (≤ {pd.to_datetime(cutB).date()})": df_plot[f"{k} B"].values,
                "Δ (B–A)": df_plot["Δ (B–A)"].values,
            }, index=pd.to_datetime(df_plot["Fecha"]))
            st.line_chart(chart_df, height=320)

            # Top-N alojamientos por pickup
            bpA, _ = compute_kpis(raw, pd.to_datetime(cutA), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            bpB, _ = compute_kpis(raw, pd.to_datetime(cutB), pd.to_datetime(p_start), pd.to_datetime(p_end), inv_override, props_pick if props_pick else None)
            merge = bpA.merge(bpB, on="Alojamiento", how="outer", suffixes=(" A", " B")).fillna(0)
            merge["Pickup noches"] = merge["Noches ocupadas B"] - merge["Noches ocupadas A"]
            merge["Pickup ingresos (€)"] = merge["Ingresos B"] - merge["Ingresos A"]
            top = merge.sort_values("Pickup noches", ascending=False).head(int(topn))
            st.subheader("🏆 Top alojamientos por pickup (noches)")
            st.dataframe(top[["Alojamiento", "Pickup noches", "Pickup ingresos (€)", "Noches ocupadas A", "Noches ocupadas B"]], use_container_width=True)

            csvp = df_plot.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 Descargar detalle pickup (CSV)", data=csvp, file_name="pickup_detalle.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pickup**.")

# ---------- Pace (curva D) ----------
elif mode == "Pace (curva D)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        c1, c2 = st.columns(2)
        p_start, p_end = period_inputs("Inicio del periodo", "Fin del periodo",
                                       date(date.today().year, date.today().month, 1),
                                       (pd.Timestamp.today().to_period("M").end_time).date(),
                                       "pace")
        dmax = st.slider("D máximo (días antes)", min_value=30, max_value=365, value=120, step=10)
        props_p = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="pace_props")
        inv_p = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="pace_inv")
        metric_p = st.radio("Métrica", ["Ocupación %", "Noches", "Ingresos (€)", "ADR (€)", "RevPAR (€)"], horizontal=False)
        compare_yoy = st.checkbox("Comparar con año anterior", value=False)
        inv_p_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="pace_inv_prev")
        run_p = st.button("Calcular pace", type="primary")

    st.subheader("🏁 Pace: evolución hacia la estancia (D)")
    help_block("Pace")
    if run_p:
        base = pace_series(raw, pd.to_datetime(p_start), pd.to_datetime(p_end), int(dmax), props_p if props_p else None, int(inv_p) if inv_p > 0 else None)
        col = METRIC_MAP.get(metric_p, None)
        if metric_p == "Noches":
            y = "noches"
        elif metric_p == "Ingresos (€)":
            y = "ingresos"
        elif col is not None:
            y = col
        else:
            y = "noches"
        plot = pd.DataFrame({"Actual": base[y].values}, index=base["D"])

        if compare_yoy:
            p_start_prev = pd.to_datetime(p_start) - pd.DateOffset(years=1)
            p_end_prev = pd.to_datetime(p_end) - pd.DateOffset(years=1)
            prev = pace_series(raw, p_start_prev, p_end_prev, int(dmax), props_p if props_p else None, int(inv_p_prev) if inv_p_prev > 0 else None)
            plot["Año anterior"] = prev[y].values
        st.line_chart(plot, height=320)
        st.dataframe(base, use_container_width=True)
        csvpace = base.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar pace (CSV)", data=csvpace, file_name="pace_curva.csv", mime="text/csv")
    else:
        st.caption("Configura parámetros y pulsa **Calcular pace**.")

# ---------- Predicción (Pace) ----------
elif mode == "Predicción (Pace)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros de predicción")
        cut_f = st.date_input("Fecha de corte", value=date.today(), key="f_cut")
        c1, c2 = st.columns(2)
        f_start, f_end = period_inputs("Inicio del periodo", "Fin del periodo",
                                       date(date.today().year, date.today().month, 1),
                                       (pd.Timestamp.today().to_period("M").end_time).date(),
                                       "forecast")
        ref_years = st.slider("Años de referencia (mismo mes)", min_value=1, max_value=3, value=2)
        dmax_f = st.slider("D máximo perfil", min_value=60, max_value=365, value=180, step=10)
        props_f = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="f_props")
        inv_f = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="f_inv")
        run_f = st.button("Calcular predicción", type="primary")

    st.subheader("🔮 Predicción mensual por Pace")
    help_block("Predicción")
    if run_f:
        res = pace_forecast_month(raw, pd.to_datetime(cut_f), pd.to_datetime(f_start), pd.to_datetime(f_end),
                                  int(ref_years), int(dmax_f), props_f if props_f else None, int(inv_f) if inv_f>0 else None)
        nights_otb = res['nights_otb']; nights_p25 = res['nights_p25']; nights_p50 = res['nights_p50']; nights_p75 = res['nights_p75']
        adr_final_p50 = res['adr_final_p50']; rev_final_p50 = res['revenue_final_p50']
        adr_tail_p25 = res['adr_tail_p25']; adr_tail_p50 = res['adr_tail_p50']; adr_tail_p75 = res['adr_tail_p75']
        pickup_needed = res['pickup_needed_p50']; pick_typ50 = res['pickup_typ_p50']; pick_typ75 = res['pickup_typ_p75']
        daily = res['daily'].copy()
        daily['OTB acumulado'] = daily['noches_ocupadas'].cumsum()

        # Tarjetas
        c1, c2, c3 = st.columns(3)
        c1.metric("OTB Noches", f"{nights_otb:,.0f}".replace(",",".")) 
        c2.metric("Forecast Noches (P50)", f"{nights_p50:,.0f}".replace(",",".")) 
        c3.metric("Forecast Ingresos (P50)", f"{rev_final_p50:,.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("ADR final (P50)", f"{adr_final_p50:,.2f}")
        low_band = min(nights_p25, nights_p75); high_band = max(nights_p25, nights_p75)
        c5.metric("Banda Noches [P25–P75]", f"[{low_band:,.0f} – {high_band:,.0f}]".replace(",","."))

        # Semáforo pickup
        if pickup_needed <= pick_typ50:
            status = "🟢 Pickup dentro del típico (P50)"
        elif pickup_needed <= pick_typ75:
            status = "🟠 Pickup por encima del P50 pero ≤ P75 histórico"
        else:
            status = "🔴 Pickup por encima del P75 histórico"
        c6.metric("Pickup necesario", f"{pickup_needed:,.0f}".replace(",",".")) 
        st.caption(f"{status} · Típico P50≈ {pick_typ50:,.0f} · P75≈ {pick_typ75:,.0f}".replace(",","."))

        # ADR tail informativo
        st.caption(f"ADR del remanente (histórico): P25≈ {adr_tail_p25:,.2f} · P50≈ {adr_tail_p50:,.2f} · P75≈ {adr_tail_p75:,.2f}")

        # Gráfico con banda y reglas horizontales
        df_band = pd.DataFrame({'Fecha': daily['Fecha'], 'low': low_band, 'high': high_band})
        base = alt.Chart(daily).encode(x=alt.X('Fecha:T', title='Fecha'))
        line = base.mark_line().encode(y=alt.Y('OTB acumulado:Q', title='Noches acumuladas'))
        band = alt.Chart(df_band).mark_area(opacity=0.15).encode(x='Fecha:T', y='low:Q', y2='high:Q')
        rule_p50 = alt.Chart(pd.DataFrame({'y':[nights_p50]})).mark_rule(strokeDash=[6,4]).encode(y='y:Q')
        rule_p25 = alt.Chart(pd.DataFrame({'y':[low_band]})).mark_rule(strokeDash=[2,4]).encode(y='y:Q')
        rule_p75 = alt.Chart(pd.DataFrame({'y':[high_band]})).mark_rule(strokeDash=[2,4]).encode(y='y:Q')
        chart = (band + line + rule_p25 + rule_p50 + rule_p75).properties(height=320)
        st.altair_chart(chart, use_container_width=True)

        csvf = daily.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar detalle diario (CSV)", data=csvf, file_name="forecast_pace_diario.csv", mime="text/csv")
    else:
        st.caption("Configura y pulsa **Calcular predicción**.")
# ===========================
# BLOQUE 4/5 — Modos extra
# ===========================

# ---------- Pipeline 90–180 días ----------
if mode == "Pipeline 90–180 días":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_pl = st.date_input("Fecha de corte", value=date.today(), key="pl_cut")
        pl_start = st.date_input("Inicio del horizonte", value=date.today().replace(day=1), key="pl_start")
        pl_end = st.date_input("Fin del horizonte", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="pl_end")
        inv_pl = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="pl_inv")
        cmp_ly_pl = st.checkbox("Comparar con LY", value=False)
        inv_pl_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="pl_inv_ly")
        run_pl = st.button("Calcular pipeline", type="primary")
    st.subheader("📆 Pipeline de OTB por día")
    if run_pl:
        inv_now = int(inv_pl) if inv_pl>0 else None
        ser = daily_series(raw, pd.to_datetime(cut_pl), pd.to_datetime(pl_start), pd.to_datetime(pl_end), None, inv_now)
        ser = ser.sort_values('Fecha')
        st.line_chart(ser.set_index('Fecha')[['noches_ocupadas','ingresos']].rename(columns={'noches_ocupadas':'Noches','ingresos':'Ingresos (€)'}), height=320)
        if cmp_ly_pl:
            ser_ly = daily_series(raw, pd.to_datetime(cut_pl) - pd.DateOffset(years=1), pd.to_datetime(pl_start) - pd.DateOffset(years=1), pd.to_datetime(pl_end) - pd.DateOffset(years=1), None, int(inv_pl_ly) if inv_pl_ly>0 else None)
            ser_ly['Fecha'] = pd.to_datetime(ser_ly['Fecha']) + pd.DateOffset(years=1)
            merge = ser.merge(ser_ly, on='Fecha', how='left', suffixes=('',' (prev)'))
            st.dataframe(merge, use_container_width=True)
        else:
            st.dataframe(ser, use_container_width=True)
        csvpl = ser.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Descargar pipeline (CSV)", data=csvpl, file_name="pipeline_diario.csv", mime="text/csv")
    else:
        st.caption("Define horizonte y pulsa **Calcular pipeline**.")

# ---------- Gap vs Target ----------
elif mode == "Gap vs Target":
    if raw is None:
        st.stop()
    tgts = st.session_state.get("targets_df")
    with st.sidebar:
        st.header("Parámetros")
        cut_gt = st.date_input("Fecha de corte", value=date.today(), key="gt_cut")
        min_m = raw['Fecha entrada'].min().to_period('M') if pd.notna(raw['Fecha entrada'].min()) else pd.Period(date.today(), 'M')
        max_m = raw['Fecha salida'].max().to_period('M') if pd.notna(raw['Fecha salida'].max()) else pd.Period(date.today(), 'M')
        months_sel = st.multiselect("Meses (YYYY-MM)", options=sorted(pd.period_range(min_m, max_m).astype(str).tolist()))
        inv_gt = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="gt_inv")
        run_gt = st.button("Calcular gaps", type="primary")
    st.subheader("🎯 Brecha a Objetivo (Targets)")
    if tgts is None:
        st.info("Carga un CSV de targets (expansor 🎯 en la barra lateral).")
    elif run_gt and months_sel:
        rows = []
        for ym in months_sel:
            p = pd.Period(ym, freq='M')
            s, e = p.start_time, p.end_time
            _, real = compute_kpis(raw, pd.to_datetime(cut_gt), s, e, int(inv_gt) if inv_gt>0 else None, None)
            y, m = p.year, p.month
            trow = tgts[(tgts['year']==y) & (tgts['month']==m)]
            tgt_occ = float(trow['target_occ_pct'].iloc[0]) if not trow.empty and 'target_occ_pct' in tgts.columns else np.nan
            tgt_adr = float(trow['target_adr'].iloc[0]) if not trow.empty and 'target_adr' in tgts.columns else np.nan
            tgt_revpar = float(trow['target_revpar'].iloc[0]) if not trow.empty and 'target_revpar' in tgts.columns else np.nan
            rows.append({"Mes": ym,
                         "Occ Real %": real['ocupacion_pct'], "Occ Target %": tgt_occ, "Gap Occ p.p.": (tgt_occ - real['ocupacion_pct']) if not np.isnan(tgt_occ) else np.nan,
                         "ADR Real": real['adr'], "ADR Target": tgt_adr, "Gap ADR": (tgt_adr - real['adr']) if not np.isnan(tgt_adr) else np.nan,
                         "RevPAR Real": real['revpar'], "RevPAR Target": tgt_revpar, "Gap RevPAR": (tgt_revpar - real['revpar']) if not np.isnan(tgt_revpar) else np.nan})
        df_gap = pd.DataFrame(rows).set_index('Mes')
        st.dataframe(df_gap, use_container_width=True)
        st.line_chart(df_gap[[c for c in df_gap.columns if 'Occ' in c]], height=280)

# ---------- Lead time & LOS ----------
elif mode == "Lead time & LOS":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        lt_start = st.date_input("Inicio periodo (por llegada)", value=date.today().replace(day=1), key="lt_start")
        lt_end = st.date_input("Fin periodo (por llegada)", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="lt_end")
        props_lt = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="lt_props")
        run_lt = st.button("Calcular", type="primary")
    st.subheader("⏱️ Lead time (por reserva) y LOS")
    help_block("Lead")
    if run_lt:
        df = raw.copy()
        if props_lt:
            df = df[df["Alojamiento"].isin(props_lt)]
        df = df.dropna(subset=["Fecha alta", "Fecha entrada", "Fecha salida"])
        mask = (df["Fecha entrada"] >= pd.to_datetime(lt_start)) & (df["Fecha entrada"] <= pd.to_datetime(lt_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango seleccionado.")
        else:
            df["lead_days"] = (df["Fecha entrada"].dt.normalize() - df["Fecha alta"].dt.normalize()).dt.days.clip(lower=0)
            df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
            c1, c2, c3 = st.columns(3)
            c1.metric("Lead medio (d)", f"{df['lead_days'].mean():.1f}")
            c2.metric("LOS medio (noches)", f"{df['los'].mean():.1f}")
            c3.metric("Lead mediana (d)", f"{np.percentile(df['lead_days'],50):.0f}")
            lt_bins = [0,3,7,14,30,60,120,1e9]; lt_labels = ["0-3","4-7","8-14","15-30","31-60","61-120","120+"]
            los_bins = [1,2,3,4,5,7,10,14,21,30, np.inf]; los_labels = ["1","2","3","4","5-7","8-10","11-14","15-21","22-30","30+"]
            lt_tab = pd.cut(df["lead_days"], bins=lt_bins, labels=lt_labels, right=True).value_counts().reindex(lt_labels).fillna(0).astype(int).rename_axis("Lead bin").reset_index(name="Reservas")
            los_tab = pd.cut(df["los"], bins=los_bins, labels=los_labels, right=True, include_lowest=True).value_counts().reindex(los_labels).fillna(0).astype(int).rename_axis("LOS bin").reset_index(name="Reservas")
            st.markdown("**Lead time (reservas)**"); st.dataframe(lt_tab, use_container_width=True)
            st.markdown("**LOS (reservas)**"); st.dataframe(los_tab, use_container_width=True)
            st.download_button("📥 Descargar Lead bins (CSV)", data=lt_tab.to_csv(index=False).encode("utf-8-sig"), file_name="lead_bins.csv", mime="text/csv")
            st.download_button("📥 Descargar LOS bins (CSV)", data=los_tab.to_csv(index=False).encode("utf-8-sig"), file_name="los_bins.csv", mime="text/csv")

# ---------- DOW heatmap ----------
elif mode == "DOW heatmap":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        h_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="dow_start")
        h_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="dow_end")
        props_h = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="dow_props")
        mode_h = st.radio("Métrica", ["Ocupación (noches)", "Ocupación (%)", "ADR (€)"], horizontal=True)
        inv_h = st.number_input("Inventario (para %)", min_value=0, value=0, step=1, key="dow_inv")
        cutoff_h = st.date_input("Fecha de corte", value=date.today(), key="dow_cutoff")
        run_h = st.button("Generar heatmap", type="primary")
    st.subheader("🗓️ Heatmap Día de la Semana × Mes")
    help_block("DOW")
    if run_h:
        df_cut = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_h)].copy()
        if props_h:
            df_cut = df_cut[df_cut["Alojamiento"].isin(props_h)]
        df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"])
        rows = []
        for _, r in df_cut.iterrows():
            e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Alquiler con IVA (€)"])
            ov_start = max(e, pd.to_datetime(h_start))
            ov_end = min(s, pd.to_datetime(h_end) + pd.Timedelta(days=1))
            n_nights = (s - e).days
            if ov_start >= ov_end or n_nights <= 0:
                continue
            adr_night = p / n_nights if n_nights > 0 else 0.0
            for d in pd.date_range(ov_start, ov_end - pd.Timedelta(days=1), freq='D'):
                rows.append({"Mes": d.strftime('%Y-%m'), "DOW": ("Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo")[d.weekday()], "Noches": 1, "ADR": adr_night, "Fecha": d.normalize()})
        if not rows:
            st.info("Sin datos en el rango.")
        else:
            df_n = pd.DataFrame(rows)
            if mode_h == "Ocupación (noches)":
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="Noches", aggfunc='sum', fill_value=0)
                st.dataframe(piv.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]), use_container_width=True)
            elif mode_h == "Ocupación (%)":
                inv_now = get_inventory(raw, int(inv_h) if inv_h>0 else None)
                occ = occurrences_of_dow_by_month(pd.to_datetime(h_start), pd.to_datetime(h_end))
                nights_piv = df_n.pivot_table(index="DOW", columns="Mes", values="Noches", aggfunc='sum', fill_value=0)
                out_cols = {}
                for mes in nights_piv.columns:
                    for dow in nights_piv.index:
                        n_occ = occ[(occ['Mes']==mes) & (occ['DOW']==dow)]['occ']
                        denom = (inv_now * (int(n_occ.iloc[0]) if not n_occ.empty else 0))
                        val = nights_piv.loc[dow, mes] / denom * 100.0 if denom>0 else 0.0
                        out_cols.setdefault(mes, {})[dow] = val
                pivp = pd.DataFrame(out_cols).reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
                st.dataframe(pivp, use_container_width=True)
            else:
                piv = df_n.pivot_table(index="DOW", columns="Mes", values="ADR", aggfunc='mean', fill_value=0.0)
                st.dataframe(piv.reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]), use_container_width=True)
            st.download_button("📥 Descargar heatmap (CSV)",
                               data=(piv if mode_h!="Ocupación (%)" else pivp).reset_index().to_csv(index=False).encode("utf-8-sig"),
                               file_name="dow_heatmap.csv", mime="text/csv")

# ---------- ADR bands & Targets ----------
elif mode == "ADR bands & Targets":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros ADR bands")
        ab_cutoff = st.date_input("Fecha de corte", value=date.today(), key="ab_cutoff")
        ab_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="ab_start")
        ab_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="ab_end")
        props_ab = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="ab_props")
        run_ab = st.button("Calcular ADR bands", type="primary")
    st.subheader("📦 Bandas de ADR (percentiles por mes)")
    help_block("ADR bands")
    if run_ab:
        df = raw[raw["Fecha alta"] <= pd.to_datetime(ab_cutoff)].copy()
        if props_ab:
            df = df[df["Alojamiento"].isin(props_ab)]
        df = df.dropna(subset=["Fecha entrada", "Fecha salida"])
        df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        df["adr_reserva"] = df["Alquiler con IVA (€)"] / df["los"]
        ov_start, ov_end = pd.to_datetime(ab_start), pd.to_datetime(ab_end) + pd.Timedelta(days=1)
        mask = ~((df["Fecha salida"] <= ov_start) | (df["Fecha entrada"] >= ov_end))
        df = df[mask]
        if df.empty:
            st.info("Sin reservas en el rango.")
        else:
            df["Mes"] = df["Fecha entrada"].dt.to_period('M').astype(str)
            def pct_cols(x):
                arr = x.dropna().values
                if arr.size == 0:
                    return pd.Series({"P10": 0.0, "P25": 0.0, "Mediana": 0.0, "P75": 0.0, "P90": 0.0})
                return pd.Series({
                    "P10": np.percentile(arr, 10),
                    "P25": np.percentile(arr, 25),
                    "Mediana": np.percentile(arr, 50),
                    "P75": np.percentile(arr, 75),
                    "P90": np.percentile(arr, 90),
                })
            bands = df.groupby("Mes")["adr_reserva"].apply(pct_cols).reset_index()
            bands_wide = bands.pivot(index="Mes", columns="level_1", values="adr_reserva").sort_index()
            st.dataframe(bands_wide, use_container_width=True)
            # ADR OTB por mes
            adr_otb_map = {}
            for ym in bands_wide.index.tolist():
                p = pd.Period(ym, freq='M')
                m_start, m_end = p.start_time, p.end_time
                _bp_m, tot_m = compute_kpis(raw, pd.to_datetime(ab_cutoff), m_start, m_end, None, props_ab if props_ab else None)
                adr_otb_map[ym] = float(tot_m['adr'])
            plot = bands_wide[["P10","Mediana","P90"]].copy()
            plot["ADR OTB"] = [adr_otb_map.get(ym, np.nan) for ym in plot.index]
            st.line_chart(plot, height=300)

            # Posición del ADR OTB en la banda
            rows_badge = []
            for ym in bands_wide.index.tolist():
                p = pd.Period(ym, freq='M'); m_start, m_end = p.start_time, p.end_time
                _bp_m, tot_m = compute_kpis(raw, pd.to_datetime(ab_cutoff), m_start, m_end, None, props_ab if props_ab else None)
                adr_otb_m = float(tot_m['adr'])
                q10 = float(bands_wide.loc[ym, 'P10']); q25 = float(bands_wide.loc[ym, 'P25']); q50 = float(bands_wide.loc[ym, 'Mediana']); q75 = float(bands_wide.loc[ym, 'P75']); q90 = float(bands_wide.loc[ym, 'P90'])
                def interp_pct(v, q10,q25,q50,q75,q90):
                    try:
                        if v <= q10: return 5.0
                        if v <= q25: return 20.0
                        if v <= q50: return 40.0
                        if v <= q75: return 65.0
                        if v <= q90: return 85.0
                        return 95.0
                    except Exception:
                        return np.nan
                p_est = interp_pct(adr_otb_m, q10,q25,q50,q75,q90)
                rows_badge.append({"Mes": ym, "ADR OTB (€)": round(adr_otb_m,2), "Posición banda (≈Pxx)": (f"P{int(round(p_est))}" if np.isfinite(p_est) else "–")})
            if rows_badge:
                st.markdown("**ADR actual vs banda (aprox.)**")
                st.dataframe(pd.DataFrame(rows_badge), use_container_width=True)

            st.download_button("📥 Descargar ADR bands (CSV)", data=bands_wide.reset_index().to_csv(index=False).encode("utf-8-sig"), file_name="adr_bands.csv", mime="text/csv")

    st.divider()
    # Targets comparativos opcionales
    tgts = st.session_state.get("targets_df")
    st.subheader("🎯 Targets vs Real vs LY (opcional)")
    if tgts is None:
        st.info("Carga un CSV de targets en la barra lateral (dentro del acordeón 🎯).")
    else:
        with st.sidebar:
            t_cutoff = st.date_input("Fecha de corte para 'Real'", value=date.today(), key="tgt_cutoff")
            options = sorted(tgts.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}", axis=1).unique().tolist())
            months_sel = st.multiselect("Meses (YYYY-MM)", options=options)
            inv_now = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="tgt_inv")
            inv_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="tgt_inv_ly")
        if months_sel:
            rows = []
            for ym in months_sel:
                y, m = map(int, ym.split('-'))
                p = pd.Period(ym, freq='M'); p_start = p.start_time; p_end = p.end_time
                _bp, real = compute_kpis(raw, pd.to_datetime(t_cutoff), p_start, p_end, int(inv_now) if inv_now>0 else None, None)
                p_prev = p - 12
                _bp2, ly = compute_kpis(raw, pd.to_datetime(t_cutoff) - pd.DateOffset(years=1), p_prev.start_time, p_prev.end_time, int(inv_ly) if inv_ly>0 else None, None)
                trow = tgts[(tgts['year']==y) & (tgts['month']==m)]
                tgt_occ = float(trow['target_occ_pct'].iloc[0]) if 'target_occ_pct' in tgts.columns and not trow.empty else np.nan
                tgt_adr = float(trow['target_adr'].iloc[0]) if 'target_adr' in tgts.columns and not trow.empty else np.nan
                tgt_revpar = float(trow['target_revpar'].iloc[0]) if 'target_revpar' in tgts.columns and not trow.empty else np.nan
                rows.append({"Mes": ym,
                             "Occ Real %": real['ocupacion_pct'], "Occ LY %": ly['ocupacion_pct'], "Occ Target %": tgt_occ,
                             "ADR Real": real['adr'], "ADR LY": ly['adr'], "ADR Target": tgt_adr,
                             "RevPAR Real": real['revpar'], "RevPAR LY": ly['revpar'], "RevPAR Target": tgt_revpar})
            df_t = pd.DataFrame(rows).set_index("Mes")
            st.dataframe(df_t, use_container_width=True)
            st.line_chart(df_t[["Occ Real %","Occ LY %","Occ Target %"]].dropna(), height=280)

# ---------- Pricing – Mapa eficiencia ----------
elif mode == "Pricing – Mapa eficiencia":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_px = st.date_input("Fecha de corte", value=date.today(), key="px_cut")
        px_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="px_start")
        px_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="px_end")
        inv_px = st.number_input("Inventario (para Occ%)", min_value=0, value=0, step=1, key="px_inv")
        run_px = st.button("Ver mapa", type="primary")
    st.subheader("💸 Eficiencia diaria: ADR vs Ocupación%")
    if run_px:
        inv_now = get_inventory(raw, int(inv_px) if inv_px>0 else None)
        ser = daily_series(raw, pd.to_datetime(cut_px), pd.to_datetime(px_start), pd.to_datetime(px_end), None, inv_now)
        ser['Occ %'] = np.where(inv_now>0, ser['noches_ocupadas'] / inv_now * 100.0, 0.0)
        ser['ADR día'] = np.where(ser['noches_ocupadas']>0, ser['ingresos']/ser['noches_ocupadas'], np.nan)
        st.scatter_chart(ser.set_index('Fecha')[['ADR día','Occ %']], height=320)
        st.dataframe(ser[['Fecha','noches_ocupadas','Occ %','ADR día','ingresos']], use_container_width=True)

# ---------- Cohortes (Alta × Estancia) ----------
elif mode == "Cohortes (Alta × Estancia)":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        props_c = st.multiselect("Alojamientos (opcional)", options=sorted(raw['Alojamiento'].unique()), default=[], key="coh_props")
        run_c = st.button("Generar cohortes", type="primary")
    st.subheader("🧩 Cohortes: Mes de creación × Mes de llegada (reservas)")
    if run_c:
        dfc = raw.copy()
        if props_c:
            dfc = dfc[dfc['Alojamiento'].isin(props_c)]
        dfc = dfc.dropna(subset=['Fecha alta','Fecha entrada'])
        dfc['Mes alta'] = dfc['Fecha alta'].dt.to_period('M').astype(str)
        dfc['Mes llegada'] = dfc['Fecha entrada'].dt.to_period('M').astype(str)
        piv = pd.pivot_table(dfc, index='Mes alta', columns='Mes llegada', values='Alojamiento', aggfunc='count', fill_value=0)
        st.dataframe(piv, use_container_width=True)
        st.download_button("📥 Descargar cohortes (CSV)", data=piv.reset_index().to_csv(index=False).encode('utf-8-sig'), file_name="cohortes_alta_estancia.csv", mime="text/csv")

# ---------- Estacionalidad ----------
elif mode == "Estacionalidad":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        dim = st.radio("Vista", ["Mes del año", "Día de la semana", "Día del mes"], horizontal=False)
        y_min = min(pd.concat([raw["Fecha entrada"], raw["Fecha salida"]]).dt.year.dropna()); y_max = max(pd.concat([raw["Fecha entrada"], raw["Fecha salida"]]).dt.year.dropna())
        years_opts = list(range(int(y_min), int(y_max) + 1)) if pd.notna(y_min) and pd.notna(y_max) else []
        years_sel = st.multiselect("Años a incluir", options=years_opts, default=years_opts)
        base = st.radio("Base de conteo", ["Noches (estancia)", "Reservas (check-in)"])
        met = st.radio("Métrica", ["Noches", "Ingresos (€)", "ADR"] if base=="Noches (estancia)" else ["Reservas"], horizontal=True)
        show_idx = st.checkbox("Mostrar índice (media=1)", value=True)
        run_s = st.button("Calcular", type="primary")
    st.subheader("🍂 Estacionalidad – distribución por periodo")
    help_block("Estacionalidad")

    def _nightly_rows(df_all: pd.DataFrame, years: list[int]) -> pd.DataFrame:
        df = df_all.dropna(subset=["Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]).copy()
        rows = []
        for _, r in df.iterrows():
            e, s, price = r["Fecha entrada"], r["Fecha salida"], float(r["Alquiler con IVA (€)"])
            n = (s - e).days
            if n <= 0: continue
            adr_n = price / n if n > 0 else 0.0
            for d in pd.date_range(e, s - pd.Timedelta(days=1), freq='D'):
                if years and d.year not in years: continue
                rows.append({"Fecha": d.normalize(),"Año": d.year,"MesN": d.month,"Mes": {1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'}[d.month],
                             "DOW": ("Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo")[d.weekday()],
                             "DOM": d.day,"Noches": 1,"Ingresos": adr_n})
        return pd.DataFrame(rows)

    if run_s:
        if base == "Noches (estancia)":
            nights_df = _nightly_rows(raw, years_sel)
            if nights_df.empty:
                st.info("No hay noches en el filtro seleccionado.")
            else:
                if dim == "Mes del año":
                    g = nights_df.groupby(["Mes","MesN"]).agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index().sort_values("MesN")
                    g["ADR"] = np.where(g["Noches"]>0, g["Ingresos"]/g["Noches"], np.nan); vals_map = {"Noches":"Noches","Ingresos (€)":"Ingresos","ADR":"ADR"}
                    vals = g.set_index("Mes")[ [ vals_map[met] ] ]
                elif dim == "Día de la semana":
                    g = nights_df.groupby("DOW").agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index()
                    g["ADR"] = np.where(g["Noches"]>0, g["Ingresos"]/g["Noches"], np.nan)
                    g = g.set_index("DOW").reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
                    vals_map = {"Noches":"Noches","Ingresos (€)":"Ingresos","ADR":"ADR"}; vals = g[[ vals_map[met] ]]
                else:
                    g = nights_df.groupby("DOM").agg(Noches=("Noches","sum"), Ingresos=("Ingresos","sum")).reset_index(); g["ADR"]=np.where(g["Noches"]>0,g["Ingresos"]/g["Noches"],np.nan)
                    vals = g.set_index("DOM")[ [ {"Noches":"Noches","Ingresos (€)":"Ingresos","ADR":"ADR"}[met] ] ]
                if show_idx & met != "ADR":
                    serie = vals.iloc[:,0]; idx = serie / (serie.mean() if serie.mean()!=0 else 1)
                    st.line_chart(idx.rename("Índice")); st.dataframe(pd.DataFrame({vals.columns[0]: serie, "Índice": idx}).reset_index(), use_container_width=True)
                else:
                    st.line_chart(vals); st.dataframe(vals.reset_index().rename(columns={"index": dim, vals.columns[0]: met}), use_container_width=True)
        else:
            dfr = raw.dropna(subset=["Fecha entrada"]).copy()
            dfr["Año"] = dfr["Fecha entrada"].dt.year
            if years_sel: dfr = dfr[dfr["Año"].isin(years_sel)]
            if dfr.empty: st.info("No hay reservas en el filtro seleccionado.")
            else:
                dfr["Mes"] = dfr["Fecha entrada"].dt.month.map({1:'Ene',2:'Feb',3:'Mar',4:'Abr',5:'May',6:'Jun',7:'Jul',8:'Ago',9:'Sep',10:'Oct',11:'Nov',12:'Dic'})
                dfr["MesN"] = dfr["Fecha entrada"].dt.month; dfr["DOW"]=dfr["Fecha entrada"].dt.weekday.map({0:"Lunes",1:"Martes",2:"Miércoles",3:"Jueves",4:"Viernes",5:"Sábado",6:"Domingo"}); dfr["DOM"]=dfr["Fecha entrada"].dt.day
                if dim == "Mes del año": vals = dfr.groupby(["Mes","MesN"]).size().reset_index(name="Reservas").sort_values("MesN").set_index("Mes")[["Reservas"]]
                elif dim == "Día de la semana": vals = dfr.groupby("DOW").size().reindex(["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"]).fillna(0).astype(int).to_frame("Reservas")
                else: vals = dfr.groupby("DOM").size().to_frame("Reservas").sort_index()
                if show_idx:
                    serie = vals.iloc[:,0]; idx = serie / (serie.mean() if serie.mean()!=0 else 1)
                    st.line_chart(idx.rename("Índice")); st.dataframe(pd.DataFrame({"Reservas": serie, "Índice": idx}).reset_index().rename(columns={"index": dim}), use_container_width=True)
                else:
                    st.line_chart(vals); st.dataframe(vals.reset_index().rename(columns={"index": dim}), use_container_width=True)

# ---------- Ranking alojamientos ----------
elif mode == "Ranking alojamientos":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_rk = st.date_input("Fecha de corte", value=date.today(), key="rk_cut")
        rk_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="rk_start")
        rk_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="rk_end")
        run_rk = st.button("Calcular ranking", type="primary")
    st.subheader("🏅 Ranking de alojamientos")
    if run_rk:
        bp, tot = compute_kpis(raw, pd.to_datetime(cut_rk), pd.to_datetime(rk_start), pd.to_datetime(rk_end), None, None)
        if bp.empty:
            st.info("Sin datos en el rango.")
        else:
            days = (pd.to_datetime(rk_end) - pd.to_datetime(rk_start)).days + 1
            bp['RevPAR estim.'] = np.where(days>0, bp['Ingresos'] / days, 0.0)
            st.dataframe(bp.sort_values('Ingresos', ascending=False), use_container_width=True)
            st.download_button("📥 Descargar ranking (CSV)", data=bp.to_csv(index=False).encode('utf-8-sig'), file_name="ranking_alojamientos.csv", mime="text/csv")

# ---------- Operativa ----------
elif mode == "Operativa":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cut_op = st.date_input("Fecha de corte", value=date.today(), key="op_cut")
        op_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="op_start")
        op_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="op_end")
        inv_op = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="op_inv")
        run_op = st.button("Calcular operativa", type="primary")
    st.subheader("🛎️ Operativa diaria")
    if run_op:
        inv_now = get_inventory(raw, int(inv_op) if inv_op>0 else None)
        dfc = raw[raw['Fecha alta'] <= pd.to_datetime(cut_op)].copy()
        days = pd.date_range(pd.to_datetime(op_start), pd.to_datetime(op_end), freq='D')
        chk_in = dfc['Fecha entrada'].dt.normalize().value_counts()
        chk_out = dfc['Fecha salida'].dt.normalize().value_counts()
        active = daily_series(raw, pd.to_datetime(cut_op), pd.to_datetime(op_start), pd.to_datetime(op_end), None, inv_now)
        out = pd.DataFrame({'Fecha': days})
        out['Check-ins'] = out['Fecha'].map(chk_in).fillna(0).astype(int)
        out['Check-outs'] = out['Fecha'].map(chk_out).fillna(0).astype(int)
        out = out.merge(active[['Fecha','noches_ocupadas']], on='Fecha', how='left').rename(columns={'noches_ocupadas':'Estancias activas'})
        out['Capacidad restante'] = inv_now - out['Estancias activas']
        out = out.fillna(0)
        st.dataframe(out, use_container_width=True)
        st.line_chart(out.set_index('Fecha')[['Estancias activas','Capacidad restante']], height=300)

# ---------- Calidad de datos ----------
elif mode == "Calidad de datos":
    if raw is None:
        st.stop()
    st.subheader("🔧 Chequeo de datos")
    dfq = raw.copy()
    bad_dates = dfq[(dfq['Fecha salida'] <= dfq['Fecha entrada']) | (dfq['Fecha entrada'].isna()) | (dfq['Fecha salida'].isna())]
    if not bad_dates.empty:
        st.warning(f"Fechas incoherentes: {len(bad_dates)} filas"); st.dataframe(bad_dates, use_container_width=True)
    bad_price = dfq[(pd.to_numeric(dfq['Alquiler con IVA (€)'], errors='coerce').fillna(0) <= 0)]
    if not bad_price.empty:
        st.warning(f"Precios nulos/negativos: {len(bad_price)} filas"); st.dataframe(bad_price, use_container_width=True)
    dfq['los'] = (dfq['Fecha salida'].dt.normalize() - dfq['Fecha entrada'].dt.normalize()).dt.days
    los0 = dfq[dfq['los'] <= 0]
    if not los0.empty:
        st.warning(f"LOS ≤ 0: {len(los0)} filas"); st.dataframe(los0, use_container_width=True)

# ---------- Calendario por alojamiento ----------
elif mode == "Calendario por alojamiento":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cutoff_cal = st.date_input("Fecha de corte", value=date.today(), key="cal_cutoff")
        cal_start = st.date_input("Inicio periodo", value=date.today().replace(day=1), key="cal_start")
        cal_end = st.date_input("Fin periodo", value=(pd.Timestamp.today().to_period("M").end_time).date(), key="cal_end")
        props_cal = st.multiselect("Alojamientos", options=sorted(raw["Alojamiento"].unique()), default=[], key="cal_props")
        mode_cal = st.radio("Modo", ["Ocupado/Libre", "ADR"], horizontal=True, key="cal_mode")
        run_cal = st.button("Generar calendario", type="primary", key="btn_cal")
    st.subheader("🗓️ Calendario por alojamiento")
    help_block("Calendario")
    if run_cal:
        if pd.to_datetime(cal_start) > pd.to_datetime(cal_end):
            st.error("El inicio del periodo no puede ser posterior al fin.")
        else:
            piv = build_calendar_matrix(raw, pd.to_datetime(cutoff_cal), pd.to_datetime(cal_start), pd.to_datetime(cal_end), props_cal if props_cal else None, mode_cal)
            if piv.empty:
                st.info("Sin datos para los filtros seleccionados.")
            else:
                piv.columns = [c.strftime('%Y-%m-%d') if isinstance(c, (pd.Timestamp, datetime, date)) else str(c) for c in piv.columns]
                st.dataframe(piv, use_container_width=True)
                st.download_button("📥 Descargar calendario (CSV)", data=piv.reset_index().to_csv(index=False).encode("utf-8-sig"), file_name="calendario_alojamientos.csv", mime="text/csv")

# ===========================
# BLOQUE 5/5 — Resumen + Cuadro de mando PRO + Eventos/Festivos + Tarificación
# ===========================

EVENTS_CSV_PATH = "eventos_festivos.csv"

@st.cache_data(show_spinner=False)
def load_events_csv(path: str) -> pd.DataFrame:
    cols = ["fecha_inicio", "fecha_fin", "uplift_pct", "nombre", "prioridad"]
    if not os.path.exists(path):
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_excel(path)
        except Exception:
            return pd.DataFrame(columns=cols)
    # Normalizar columnas
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols].copy()
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce").dt.date
    df["fecha_fin"]    = pd.to_datetime(df["fecha_fin"], errors="coerce").dt.date
    df["uplift_pct"]   = pd.to_numeric(df["uplift_pct"], errors="coerce")
    df["prioridad"]    = pd.to_numeric(df["prioridad"], errors="coerce").fillna(1).astype(int)
    df["nombre"]       = df["nombre"].fillna("").astype(str)
    df = df.dropna(subset=["fecha_inicio", "fecha_fin", "uplift_pct"])
    return df.reset_index(drop=True)

def save_events_csv(df_in: pd.DataFrame, path: str) -> None:
    df = df_in.copy()
    if df.empty:
        pd.DataFrame(columns=["fecha_inicio","fecha_fin","uplift_pct","nombre","prioridad"]).to_csv(path, index=False, encoding="utf-8-sig")
        return
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce").dt.date
    df["fecha_fin"]    = pd.to_datetime(df["fecha_fin"], errors="coerce").dt.date
    df["uplift_pct"]   = pd.to_numeric(df["uplift_pct"], errors="coerce")
    df["prioridad"]    = pd.to_numeric(df.get("prioridad", 1), errors="coerce").fillna(1).astype(int)
    df["nombre"]       = df.get("nombre", "").fillna("").astype(str)
    df = df.dropna(subset=["fecha_inicio","fecha_fin","uplift_pct"])
    df.to_csv(path, index=False, encoding="utf-8-sig")

def expand_events_by_day(df_events: pd.DataFrame) -> pd.DataFrame:
    """Expande a diario y resuelve solapes por prioridad y luego uplift."""
    if df_events is None or df_events.empty:
        return pd.DataFrame(columns=["fecha","uplift_pct","origen","prioridad"])
    rows = []
    for _, r in df_events.iterrows():
        fi, ff, up = r["fecha_inicio"], r["fecha_fin"], float(r["uplift_pct"])
        nombre = str(r.get("nombre","")).strip()
        prio   = int(r.get("prioridad", 1))
        if pd.isna(fi) or pd.isna(ff) or pd.isna(up):
            continue
        for d in pd.date_range(pd.to_datetime(fi), pd.to_datetime(ff), freq="D"):
            rows.append({"fecha": d.date(), "uplift_pct": up, "origen": nombre, "prioridad": prio})
    if not rows:
        return pd.DataFrame(columns=["fecha","uplift_pct","origen","prioridad"])
    df = pd.DataFrame(rows)
    # Resolver solapes: mayor prioridad; si empatan, mayor uplift
    df = df.sort_values(["fecha","prioridad","uplift_pct"], ascending=[True, False, False])
    df = df.drop_duplicates("fecha", keep="first").reset_index(drop=True)
    return df

def _reservas_en_periodo(df: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]]):
    """Reservas con Fecha alta <= corte y que intersectan el periodo [start, end] (estancia)."""
    d = df.copy()
    d = d[(d["Fecha alta"] <= cutoff)]
    if props:
        d = d[d["Alojamiento"].isin(props)]
    d = d.dropna(subset=["Fecha entrada","Fecha salida","Alquiler con IVA (€)"])
    # Intersección con periodo
    ov_start = start
    ov_end   = end + pd.Timedelta(days=1)
    mask = ~((d["Fecha salida"] <= ov_start) | (d["Fecha entrada"] >= ov_end))
    return d[mask].copy()

def adr_bands_p50_for_month_by_apto(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: List[str]) -> Dict[str, float]:
    """Devuelve P50 (mediana) de ADR por Alojamiento en el periodo."""
    out = {}
    if not props:
        return out
    d = _reservas_en_periodo(df_all, cutoff, start, end, props)
    if d.empty:
        return {a: np.nan for a in props}
    d["los"] = (d["Fecha salida"].dt.normalize() - d["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    d["adr_reserva"] = pd.to_numeric(d["Alquiler con IVA (€)"], errors="coerce") / d["los"]
    for a in props:
        sub = d[d["Alojamiento"] == a]["adr_reserva"].dropna().values
        out[a] = float(np.median(sub)) if sub.size else np.nan
    return out

def adr_bands_p50_for_month(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]]) -> float:
    d = _reservas_en_periodo(df_all, cutoff, start, end, props)
    if d.empty:
        return np.nan
    d["los"] = (d["Fecha salida"].dt.normalize() - d["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    d["adr_reserva"] = pd.to_numeric(d["Alquiler con IVA (€)"], errors="coerce") / d["los"]
    arr = d["adr_reserva"].dropna().values
    return float(np.median(arr)) if arr.size else np.nan

def compute_m_apto_by_property(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: List[str]) -> Dict[str, float]:
    """m_apto = ADR_apto_LY / ADR_grupo_LY para el periodo equivalente del año pasado."""
    if not props:
        return {}
    cut_ly  = cutoff - pd.DateOffset(years=1)
    s_ly    = start  - pd.DateOffset(years=1)
    e_ly    = end    - pd.DateOffset(years=1)
    # P50 grupo LY
    p50_group = adr_bands_p50_for_month(df_all, cut_ly, s_ly, e_ly, props)
    if not np.isfinite(p50_group) or p50_group <= 0:
        p50_group = np.nan
    # P50 por apto LY
    p50_map = adr_bands_p50_for_month_by_apto(df_all, cut_ly, s_ly, e_ly, props)
    out = {}
    for a in props:
        p50_a = p50_map.get(a, np.nan)
        if np.isfinite(p50_a) and np.isfinite(p50_group) and p50_group > 0:
            out[a] = float(p50_a / p50_group)
        else:
            out[a] = 1.0
    return out

def build_pricing_calendar_grid(result_df: pd.DataFrame, ev_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot: filas=Alojamiento, columnas=Fecha (YYYY-MM-DD), valores=Precio propuesto. Meta por columna."""
    if result_df is None or result_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = result_df.copy()
    df["Fecha"] = pd.to_datetime(df["Fecha"]).dt.normalize()
    piv = df.pivot_table(index="Alojamiento", columns="Fecha", values="Precio propuesto", aggfunc="mean")
    piv = piv.sort_index(axis=1)
    # meta de columnas
    meta = pd.DataFrame({"Fecha": piv.columns})
    meta["is_weekend"] = meta["Fecha"].dt.weekday.isin([5,6])
    if ev_daily is not None and not ev_daily.empty:
        e = ev_daily.copy()
        e["Fecha"] = pd.to_datetime(e["fecha"])
        e = e[["Fecha","uplift_pct","origen"]].drop_duplicates("Fecha")
        meta = meta.merge(e, on="Fecha", how="left")
        meta["has_event"] = meta["uplift_pct"].notna()
        meta["event_name"] = meta["origen"].fillna("")
    else:
        meta["has_event"] = False
        meta["event_name"] = ""
        meta["uplift_pct"] = np.nan
    meta = meta.set_index("Fecha")
    # renombrar columnas a string
    piv.columns = [c.strftime("%Y-%m-%d") for c in piv.columns]
    meta.index = [i.strftime("%Y-%m-%d") for i in meta.index]
    return piv, meta

def style_pricing_calendar(grid: pd.DataFrame, meta_cols: pd.DataFrame) -> Styler:
    """Aplica formato: fines de semana gris, evento amarillo; tooltip con evento."""
    if grid is None or grid.empty:
        return grid
    # map meta a arrays alineados con columnas
    cols = list(grid.columns)
    m = meta_cols.reindex(cols).copy()
    is_we = m["is_weekend"].fillna(False).values if "is_weekend" in m.columns else np.array([False]*len(cols))
    has_ev= m["has_event"].fillna(False).values if "has_event" in m.columns else np.array([False]*len(cols))
    toolt = m["event_name"].fillna("").astype(str).values if "event_name" in m.columns else np.array([""]*len(cols))

    def _bgcolor(val, j):
        if has_ev[j]:
            return "background-color: #FFF3B0"  # amarillo suave
        if is_we[j]:
            return "background-color: #F0F0F0"  # gris
        return ""
    # construir matriz de estilos
    styles = pd.DataFrame("", index=grid.index, columns=grid.columns)
    for j, c in enumerate(cols):
        col_style = "background-color: #FFF3B0" if has_ev[j] else ("background-color: #F0F0F0" if is_we[j] else "")
        if col_style:
            styles[c] = col_style
    sty = grid.style.format("{:.2f}").set_table_styles([{
        "selector": "th.col_heading",
        "props": [("position","sticky"),("top","0"),("background","#fff")]
    }]).set_properties(**{"border":"1px solid #EEE"})
    sty = sty.set_td_classes(pd.DataFrame("", index=grid.index, columns=grid.columns)).apply(lambda _: styles, axis=None)
    # Nota: Streamlit no soporta tooltips por celda directamente; podemos añadir columna extra con nombre de evento si se desea.
    return sty

# ---------- Resumen & Simulador ----------
if mode == "Resumen & Simulador":
    if raw is None:
        st.stop()
    with st.sidebar:
        st.header("Parámetros")
        cutoff_r = st.date_input("Fecha de corte", value=date.today(), key="cut_resumen")
        start_r, end_r = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "resumen"
        )
        props_r = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_resumen")
        inv_r = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_resumen")
        ref_years_r = st.slider("Años de referencia (pace)", 1, 3, 2, key="ref_years_r")
        dmax_r = st.slider("D máximo pace", 60, 365, 180, 10, key="dmax_r")
        st.markdown("—")
        st.subheader("Simulador")
        delta_price = st.slider("Ajuste ADR del remanente (%)", -30, 30, 0, 1, key="sim_delta")
        elasticity = st.slider("Elasticidad de demanda", -1.5, -0.2, -0.8, 0.1, key="sim_elast")
        run_r = st.button("Calcular resumen", type="primary", key="btn_resumen")

    st.subheader("📊 Resumen & Simulador")
    help_block("Resumen")

    if run_r:
        props_sel = props_r if props_r else None
        inv_now = int(inv_r) if inv_r > 0 else None

        # KPIs actuales
        _, tot = compute_kpis(raw, pd.to_datetime(cutoff_r), pd.to_datetime(start_r), pd.to_datetime(end_r), inv_now, props_sel)
        noches_otb, ingresos_otb, adr_otb, noches_disp = tot["noches_ocupadas"], tot["ingresos"], tot["adr"], tot["noches_disponibles"]

        # Forecast pace
        forecast = pace_forecast_month(
            raw, pd.to_datetime(cutoff_r), pd.to_datetime(start_r), pd.to_datetime(end_r),
            int(ref_years_r), int(dmax_r), props_sel, inv_now
        )
        nights_p50 = forecast["nights_p50"]
        revenue_final_p50 = forecast["revenue_final_p50"]
        adr_final_p50 = forecast["adr_final_p50"]
        pickup_needed = forecast["pickup_needed_p50"]
        pickup_typ_p50 = forecast["pickup_typ_p50"]
        pickup_typ_p75 = forecast["pickup_typ_p75"]

        # Métricas actuales
        c1, c2, c3 = st.columns(3)
        c1.metric("Noches OTB", f"{noches_otb:.0f}")
        c2.metric("Forecast Noches (P50)", f"{nights_p50:.0f}")
        c3.metric("ADR OTB", f"{adr_otb:.2f}")
        c4, c5, c6 = st.columns(3)
        c4.metric("Pickup necesario", f"{pickup_needed:.0f}")
        c5.metric("ADR final (P50)", f"{adr_final_p50:.2f}")
        c6.metric("Ingresos final (P50)", f"{revenue_final_p50:.2f}")

        # Simulador ADR
        noches_rem = max(nights_p50 - noches_otb, 0.0)
        adj_factor = (1 + delta_price / 100) ** elasticity
        sim_nights = noches_otb + noches_rem * adj_factor
        sim_tail_adr = forecast["adr_tail_p50"] * (1 + delta_price / 100)
        sim_revenue = ingresos_otb + sim_tail_adr * (noches_rem * adj_factor)
        sim_adr = sim_revenue / sim_nights if sim_nights > 0 else 0.0
        sim_occ = sim_nights / noches_disp * 100 if noches_disp > 0 else 0.0

        st.markdown("**Simulación con ajuste de ADR del remanente**")
        s1, s2, s3 = st.columns(3)
        s1.metric("ADR final sim.", f"{sim_adr:.2f}")
        s2.metric("Noches finales", f"{sim_nights:.0f}")
        s3.metric("Ocupación final %", f"{sim_occ:.2f}%")
        st.metric("Ingresos finales", f"{sim_revenue:.2f}")

# ---------- Cuadro de mando PRO (con ingresos YoY) ----------
elif mode == "Cuadro de mando (PRO)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros PRO")
        cutoff_pro = st.date_input("Fecha de corte", value=date.today(), key="cut_pro")
        start_pro, end_pro = period_inputs(
            "Inicio periodo", "Fin periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "pro"
        )
        props_pro = st.multiselect("Alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_pro")
        inv_pro = st.number_input("Inventario (opcional)", min_value=0, value=0, step=1, key="inv_pro")
        dmax_pro = st.slider("D máximo para Pace YoY", 30, 180, 90, 10, help="Ventana de comparación del ritmo (Días antes de la estancia).", key="dmax_pro")

        years_back = st.slider("Años anteriores para comparar ADR", 0, 3, 2, 1, key="pro_years_back")
        delta_ok_band = st.slider("Umbral semáforo ADR vs LY (±%)", 1, 15, 5, 1, key="pro_delta_ok")

        run_pro = st.button("Generar cuadro PRO", type="primary", key="btn_pro")

    st.subheader("📊 Cuadro de Mando PRO")
    if run_pro:
        props_sel = props_pro if props_pro else None
        inv_now = int(inv_pro) if inv_pro > 0 else None

        # KPIs base (a fecha de corte)
        _, tot = compute_kpis(
            raw,
            pd.to_datetime(cutoff_pro),
            pd.to_datetime(start_pro),
            pd.to_datetime(end_pro),
            inv_now,
            props_sel
        )
        occ, adr, revpar = tot["ocupacion_pct"], tot["adr"], tot["revpar"]

        # Health score simple
        health = min(occ / 80 * 40, 40) + min(adr / 100 * 30, 30) + min(revpar / 60 * 30, 30)
        c1, c2, c3 = st.columns(3)
        c1.metric("Ocupación %", f"{occ:.1f}%")
        c2.metric("ADR medio", f"{adr:.2f} €")
        c3.metric("RevPAR", f"{revpar:.2f} €")
        st.progress(int(min(health, 100)), text=f"Health Score: {health:.0f}/100")

        # Forecast y pickup rápido (Pace)
        forecast = pace_forecast_month(
            raw,
            pd.to_datetime(cutoff_pro),
            pd.to_datetime(start_pro),
            pd.to_datetime(end_pro),
            2, 180,
            props_sel,
            inv_now
        )
        st.markdown(
            f"**Pickup necesario**: {forecast['pickup_needed_p50']:.0f} · "
            f"Típico≈ {forecast['pickup_typ_p50']:.0f} (P75≈ {forecast['pickup_typ_p75']:.0f})"
        )

        # Ritmo (Pace) vs LY
        st.markdown("### Ritmo de ocupación (Pace) vs LY")
        base_pace = pace_series(raw, pd.to_datetime(start_pro), pd.to_datetime(end_pro), int(dmax_pro), props_sel, inv_now)
        ly_pace   = pace_series(raw, pd.to_datetime(start_pro)-pd.DateOffset(years=1), pd.to_datetime(end_pro)-pd.DateOffset(years=1), int(dmax_pro), props_sel, inv_now)
        if not base_pace.empty and not ly_pace.empty:
            comp = pd.DataFrame({"D": base_pace["D"]})
            comp["Occ % actual"] = base_pace["ocupacion_pct"].values
            ly_vals = ly_pace["ocupacion_pct"].values
            if len(ly_vals) < len(comp):
                ly_vals = np.pad(ly_vals, (0, len(comp) - len(ly_vals)), constant_values=np.nan)
            comp["Occ % LY"] = ly_vals[:len(comp)]
            import altair as alt
            df_chart = comp[["D", "Occ % actual", "Occ % LY"]].copy()
            melted = df_chart.melt("D", var_name="serie", value_name="value")
            chart_lines = alt.Chart(melted).mark_line().encode(
                x=alt.X("D:Q", title="Días antes de la estancia"),
                y=alt.Y("value:Q", title="Ocupación %"),
                color=alt.Color("serie:N", title=None)
            )
            # Punto de corte (buscar D más cercano al cutoff por robustez)
            try:
                d_today = int((pd.to_datetime(start_pro) - pd.to_datetime(cutoff_pro)).days)
                # Tomamos la fila con D más cercano a d_today, aunque no exista exactamente
                _row = df_chart.iloc[(df_chart['D'] - d_today).abs().argsort()[:1]]
                d_mark = int(_row['D'].iloc[0])
                y_mark = float(_row['Occ % actual'].iloc[0])
                point_df = pd.DataFrame({'D':[d_mark], 'serie':['Occ % actual'], 'value':[y_mark]})
                chart_point = alt.Chart(point_df).mark_point(size=110).encode(x='D:Q', y='value:Q', color='serie:N')
                # Etiqueta opcional pegada al punto
                chart_text = alt.Chart(point_df).mark_text(dx=8, dy=-8).encode(
                    x='D:Q', y='value:Q', text=alt.value(f'D-{d_mark} · ' + f"{y_mark:.1f}%"))
                chart = (chart_lines + chart_point + chart_text).properties(height=260)
            except Exception:
                chart = chart_lines.properties(height=260)
            st.altair_chart(chart, use_container_width=True)

            def _val_at(df, D, col):
                r = df.loc[df["D"] == int(D), col]
                return float(r.values[0]) if len(r) else np.nan
            hitos = [60, 30, 14]
            cols_h = st.columns(len(hitos))
            for i, D in enumerate(hitos):
                now_v = _val_at(base_pace, D, "ocupacion_pct")
                ly_v  = _val_at(ly_pace, D, "ocupacion_pct")
                if np.isfinite(now_v) and np.isfinite(ly_v):
                    delta_pp = now_v - ly_v
                    cols_h[i].metric(f"D-{D}", f"{now_v:.1f}% vs {ly_v:.1f}%", delta=f"{delta_pp:+.1f} pp")
                else:
                    cols_h[i].metric(f"D-{D}", "—", delta="—")
                    
                        # === Resumen fácil del Pace: tabla de hitos + veredicto ===
            st.markdown("#### 📊 Interpretación rápida del Pace (actual vs LY)")

            def _nearest_value(df, D, col="ocupacion_pct"):
                """Devuelve el valor en el D exacto; si no existe, usa el más cercano."""
                if df is None or df.empty:
                    return np.nan
                r = df.loc[df["D"] == int(D), col]
                if len(r):
                    return float(r.values[0])
                # más cercano
                idx = (df["D"] - int(D)).abs().argsort().iloc[0]
                return float(df.iloc[idx][col])

            # Puedes incluir D-7 si quieres una lectura más last-minute
            milestones_full = [60, 30, 14, 7]
            rows_tbl = []
            gaps_pp = []

            for D in milestones_full:
                now_v = _nearest_value(base_pace, D, "ocupacion_pct")
                ly_v  = _nearest_value(ly_pace,   D, "ocupacion_pct")
                gap_pp = (now_v - ly_v) if (np.isfinite(now_v) and np.isfinite(ly_v)) else np.nan
                gaps_pp.append(gap_pp)
                rows_tbl.append({
                    "Hito": f"D-{D}",
                    "Occ % actual": (f"{now_v:.1f}%" if np.isfinite(now_v) else "—"),
                    "Occ % LY":     (f"{ly_v:.1f}%"  if np.isfinite(ly_v)  else "—"),
                    "Gap (p.p.)":   (f"{gap_pp:+.1f}" if np.isfinite(gap_pp) else "—"),
                })

            df_hitos = pd.DataFrame(rows_tbl, columns=["Hito", "Occ % actual", "Occ % LY", "Gap (p.p.)"])
            st.dataframe(df_hitos, use_container_width=True, hide_index=True)

            # Veredicto simple del pace
            gaps_valid = [g for g in gaps_pp if np.isfinite(g)]
            if gaps_valid:
                above = sum(g >= 5 for g in gaps_valid)
                below = sum(g <= -5 for g in gaps_valid)
                if below >= max(2, len(gaps_valid)//2 + 1):
                    pace_veredicto = "🔴 Pickup retrasado"
                    pace_explain   = "La ocupación a igual anticipación está por **debajo** de LY en la mayoría de hitos."
                elif above >= max(2, len(gaps_valid)//2 + 1):
                    pace_veredicto = "🟢 Pickup adelantado"
                    pace_explain   = "La ocupación a igual anticipación está por **encima** de LY en la mayoría de hitos."
                else:
                    pace_veredicto = "🟠 Pickup en línea"
                    pace_explain   = "Diferencias pequeñas o mixtas según el hito."

                num_txt = " · ".join(
                    [f"D-{milestones_full[i]}: {(('+' if gaps_pp[i]>=0 else '') + f'{gaps_pp[i]:.1f}') if np.isfinite(gaps_pp[i]) else '—'} p.p."
                     for i in range(len(milestones_full))]
                )
                st.markdown(
                    f"**Veredicto Pace:** {pace_veredicto}  \n"
                    f"{pace_explain}  \n"
                    f"_Detalle_: {num_txt}"
                )
            else:
                st.info("No hay datos suficientes para la interpretación rápida del pace.")

        else:
            st.info("No hay datos suficientes para comparar el ritmo con LY en este periodo.")

        # Ingresos del periodo vs LY/LY-2
        st.markdown("### Ingresos del periodo vs años anteriores")

        def _kpis_period(df_all, cutoff, start, end, props=None, inv=None):
            _, _tot = compute_kpis(
                df_all=df_all,
                cutoff=pd.to_datetime(cutoff),
                period_start=pd.to_datetime(start),
                period_end=pd.to_datetime(end),
                inventory_override=(int(inv) if (inv is not None and inv > 0) else None),
                filter_props=props if props else None,
            )
            return {"ingresos": float(_tot["ingresos"]), "occ": float(_tot["ocupacion_pct"]), "adr": float(_tot["adr"]), "revpar": float(_tot["revpar"])}

        k_now = _kpis_period(raw, cutoff_pro, start_pro, end_pro, props_sel, None)
        k_ly  = _kpis_period(raw, pd.to_datetime(cutoff_pro)-pd.DateOffset(years=1),
                                  pd.to_datetime(start_pro)-pd.DateOffset(years=1),
                                  pd.to_datetime(end_pro)-pd.DateOffset(years=1),
                                  props_sel, None)
        k_ly2 = _kpis_period(raw, pd.to_datetime(cutoff_pro)-pd.DateOffset(years=2),
                                  pd.to_datetime(start_pro)-pd.DateOffset(years=2),
                                  pd.to_datetime(end_pro)-pd.DateOffset(years=2),
                                  props_sel, None)

        c1r, c2r, c3r = st.columns(3)
        # === Cálculo de ingresos finales del mismo periodo LY (cutoff = fin del periodo LY)
        start_prev = pd.to_datetime(start_pro) - pd.DateOffset(years=1)
        end_prev   = pd.to_datetime(end_pro)   - pd.DateOffset(years=1)
        k_ly_final = _kpis_period(raw, end_prev, start_prev, end_prev, props_sel, None)
        c1r.metric("Ingresos actuales (€)", f"{k_now['ingresos']:,.2f}".replace(",", "."))
        c2r.metric("Ingresos LY (€)", f"{k_ly['ingresos']:,.2f}".replace(",", "."), delta=f"{(k_now['ingresos'] - k_ly['ingresos']):,.2f}".replace(",", "."))
        c3r.metric("Ingresos LY-2 (€)", f"{k_ly2['ingresos']:,.2f}".replace(",", "."), delta=f"{(k_now['ingresos'] - k_ly2['ingresos']):,.2f}".replace(",", "."))

        st.metric("Ingresos finales LY (resultado) (€)", f"{k_ly_final['ingresos']:,.2f}".replace(",", "."))
        df_rev = pd.DataFrame({"Etiqueta": ["Actual","LY","LY-2"], "Ingresos (€)": [k_now["ingresos"], k_ly["ingresos"], k_ly2["ingresos"]]})
        st.bar_chart(df_rev.set_index("Etiqueta"))

        def _pct_delta(new, old):
            if old is None or not np.isfinite(old) or old == 0:
                return np.nan
            return (new - old) / old * 100.0

        d_rev_pct = _pct_delta(k_now["ingresos"], k_ly["ingresos"])
        d_adr_pct = _pct_delta(k_now["adr"], k_ly["adr"])
        d_occ_pp  = (k_now["occ"] - k_ly["occ"]) if (np.isfinite(k_now["occ"]) and np.isfinite(k_ly["occ"])) else np.nan

        if np.isfinite(d_rev_pct):
            trend = "🟢" if d_rev_pct > 1.0 else ("🟠" if abs(d_rev_pct) <= 1.0 else "🔴")
            if (np.isfinite(d_adr_pct)) and (np.isfinite(d_occ_pp)):
                if d_adr_pct >= 0 and d_occ_pp >= 0:
                    driver = "crecimiento apoyado tanto en precio (ADR) como en volumen (ocupación)"
                elif d_adr_pct > 0 and d_occ_pp < 0:
                    driver = "mejora impulsada por ADR, pese a menor ocupación"
                elif d_adr_pct < 0 and d_occ_pp > 0:
                    driver = "mejora impulsada por mayor ocupación, con ADR más ajustado"
                else:
                    driver = "descenso por menor ADR y menor ocupación"
            else:
                driver = "variación respecto a LY"

            st.markdown(
                f"> **Ingresos del periodo**: {k_now['ingresos']:,.2f} € · "
                f"**LY**: {k_ly['ingresos']:,.2f} € · **LY-2**: {k_ly2['ingresos']:,.2f} €".replace(",", ".")
            )
            st.markdown(
                f"{trend} **Ingresos vs LY:** {d_rev_pct:+.1f}%  ·  ADR: {d_adr_pct:+.1f}%  ·  Δ Ocupación: {d_occ_pp:+.1f} p.p."
            )
        else:
            st.info("No se puede calcular la variación vs LY (valores insuficientes).")

        # Bandas ADR
        st.markdown("### Bandas de ADR (posicionamiento de precio)")
        dfb = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_pro)].dropna(subset=["Fecha entrada", "Fecha salida"]).copy()
        if props_sel:
            dfb = dfb[dfb["Alojamiento"].isin(props_sel)]
        dfb["los"] = (dfb["Fecha salida"] - dfb["Fecha entrada"]).dt.days.clip(lower=1)
        dfb["adr_reserva"] = pd.to_numeric(dfb["Alquiler con IVA (€)"], errors="coerce") / dfb["los"]
        mask = ~((dfb["Fecha salida"] <= pd.to_datetime(start_pro)) | (dfb["Fecha entrada"] >= pd.to_datetime(end_pro)))
        dfb = dfb[mask]
        q25 = q50 = q75 = np.nan
        if not dfb.empty:
            arr = dfb["adr_reserva"].dropna().values
            q25, q50, q75 = [np.percentile(arr, p) for p in (25, 50, 75)]
            st.markdown(f"**Bandas ADR:** P25={q25:.2f} · P50={q50:.2f} · P75={q75:.2f}")
            if adr < q25:
                st.warning("ADR por debajo de P25 → posible margen para subir en picos y fines de semana.")
            elif adr > q75:
                st.error("ADR por encima de P75 → riesgo de sobreprecio; revisa conversiones y mínimos.")
            else:
                st.success("ADR dentro de la banda P25–P75.")
        else:
            st.info("No hay reservas suficientes en el periodo para calcular bandas de ADR.")

        # ADR histórico (LY..)
        
        # === Análisis ejecutivo PRO (Kai) ===
        st.markdown("### Análisis ejecutivo PRO")
        try:
            analysis_pro_text = _kai_cdm_pro_analysis(
                {
                    "ocupacion_pct": float(occ),
                    "adr": float(adr),
                    "revpar": float(revpar),
                    "ingresos": float(tot.get("ingresos", np.nan)),
                    "noches_ocupadas": float(tot.get("noches_ocupadas", np.nan)) if "noches_ocupadas" in tot else np.nan,
                },
                {
                    "ocupacion_pct": float(k_ly.get("occ", np.nan)),
                    "adr": float(k_ly.get("adr", np.nan)),
                    "revpar": float(k_ly.get("revpar", np.nan)),
                    "ingresos": float(k_ly.get("ingresos", np.nan)),
                },
                {
                    "noches_ocupadas": np.nan,
                    "ingresos": float(k_ly_final.get("ingresos", np.nan)),
                },
                forecast,
                (float(q50) if np.isfinite(q50) else None)
            )
            st.markdown(analysis_pro_text)
        except Exception as e:
            st.warning(f"No se pudo generar el análisis PRO: {e}")

        st.markdown("### ADR histórico del periodo")
        adr_now_val = float(tot.get("adr", np.nan))
        adr_hist = []
        if years_back > 0:
            for k in range(1, years_back + 1):
                cut_k = pd.to_datetime(cutoff_pro) - pd.DateOffset(years=k)
                start_k = pd.to_datetime(start_pro) - pd.DateOffset(years=k)
                end_k = pd.to_datetime(end_pro) - pd.DateOffset(years=k)
                _bp_k, tot_k = compute_kpis(raw, cut_k, start_k, end_k, inv_now, props_sel)
                adr_k = float(tot_k["adr"])
                adr_hist.append((f"LY-{k}" if k > 1 else "LY", adr_k))
        cols_top = st.columns(3)
        cols_top[0].metric("ADR Actual", f"{adr:.2f} €")
        adr_ly = None
        if adr_hist:
            label_ly, adr_ly = adr_hist[0]
            if adr_ly and adr_ly > 0:
                delta_abs = adr - adr_ly
                delta_pct = (adr / adr_ly - 1.0) * 100.0
                cols_top[1].metric(f"ADR {label_ly}", f"{adr_ly:.2f} €", delta=f"{delta_abs:+.2f} € ({delta_pct:+.1f}%)")
            else:
                cols_top[1].metric(f"ADR {label_ly}", "—")
            if len(adr_hist) > 1:
                label_more, adr_more = adr_hist[1]
                cols_top[2].metric(f"ADR {label_more}", f"{adr_more:.2f} €" if adr_more > 0 else "—")
        else:
            cols_top[1].metric("ADR LY", "—")
            cols_top[2].metric("ADR LY-2", "—")

        if adr_hist:
            comp_df = pd.DataFrame([("Actual", float(adr))] + adr_hist, columns=["Periodo", "ADR"])
            st.bar_chart(comp_df.set_index("Periodo"), height=220)

        if adr_ly and adr_ly > 0:
            delta_pct = (adr / adr_ly - 1.0) * 100.0
            band = float(delta_ok_band)
            if delta_pct >= band:
                status = "🟢 Por encima de LY"
                insight = "Mantén o sube en picos; vigila conversión y cancelaciones."
            elif delta_pct <= -band:
                status = "🔴 Por debajo de LY"
                insight = "Revisar precio/visibilidad; reduce mínimos de estancia si hay gaps."
            else:
                status = "🟠 En línea con LY"
                insight = "Ajustes quirúrgicos por DOW/eventos; monitoriza pickup semanal."
            st.write(f"**{status}** · Δ vs LY = {delta_pct:+.1f}% (umbral ±{band}%). {insight}")
        else:
            st.info("Sin ADR LY suficiente para comparar este periodo con el corte seleccionado.")

        # Recomendaciones rápidas
        recs = []
        if occ < 60:
            recs.append("⚠️ Ocupación baja: prueba bajar mínimos/estancias y abrir huecos cortos.")
        if np.isfinite(q25) and adr < q25:
            recs.append("💡 ADR bajo: empuja visibilidad (OTAs/GHA) antes de tocar precio.")
        elif np.isfinite(q75) and adr > q75:
            recs.append("📉 ADR alto: ajusta en DOW flojos o lanza promos selectivas.")
        if forecast['pickup_needed_p50'] > forecast['pickup_typ_p75']:
            recs.append("🚨 Pickup muy retrasado → revisa mínimos y activa campañas flash.")
        if not recs:
            recs.append("✅ Todo bajo control. Mantén la estrategia actual.")
        st.markdown("### Recomendaciones")
        for r in recs:
            st.write(r)

# =============================
# MODO: Eventos & Festivos
# =============================
elif mode == "Eventos & Festivos":
    st.subheader("🎉 Eventos & Festivos – gestor en-app")
    st.caption("Crea/edita eventos con rango de fechas y uplift %. Se guardan en un CSV local y se aplican en tarificación.")

    base_df = load_events_csv(EVENTS_CSV_PATH).copy()
    if "events_df" not in st.session_state:
        st.session_state["events_df"] = base_df

    ev_df = st.session_state["events_df"].copy()

    with st.expander("➕ Añadir evento"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fi = st.date_input("Fecha inicio", value=None, key="ev_fi")
        with c2:
            ff = st.date_input("Fecha fin", value=None, key="ev_ff")
        with c3:
            up = st.number_input("Uplift %", min_value=-50.0, max_value=200.0, value=10.0, step=1.0, key="ev_up")
        c4, c5 = st.columns(2)
        with c4:
            nombre = st.text_input("Nombre (opcional)", value="", key="ev_nombre")
        with c5:
            prioridad = st.number_input("Prioridad (1..9)", min_value=1, max_value=9, value=1, step=1, key="ev_prio",
                                        help="Si hay solapes, gana la mayor prioridad; si empatan, mayor uplift %.")
        if st.button("Agregar a la lista", type="primary", key="ev_add"):
            if fi and ff and (up is not None):
                new_row = {
                    "fecha_inicio": pd.to_datetime(fi).date(),
                    "fecha_fin": pd.to_datetime(ff).date(),
                    "uplift_pct": float(up),
                    "nombre": nombre.strip(),
                    "prioridad": int(prioridad),
                }
                st.session_state["events_df"] = pd.concat([ev_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Evento añadido. Recuerda pulsar **Guardar** para persistir.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            else:
                st.warning("Completa inicio, fin y uplift %.")

    st.markdown("#### Lista de eventos (editable)")
    edited = st.data_editor(
        st.session_state["events_df"],
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "fecha_inicio": st.column_config.DateColumn("Fecha inicio", format="YYYY-MM-DD", required=True),
            "fecha_fin": st.column_config.DateColumn("Fecha fin", format="YYYY-MM-DD", required=True),
            "uplift_pct": st.column_config.NumberColumn("Uplift %", step=1.0),
            "nombre": st.column_config.TextColumn("Nombre"),
            "prioridad": st.column_config.NumberColumn("Prioridad", min_value=1, max_value=9, step=1),
        },
        hide_index=True,
        key="events_editor",
    )

    colA, colB, colC, colD = st.columns(4)
    with colA:
        if st.button("💾 Guardar", type="primary", key="ev_save"):
            try:
                save_events_csv(edited, EVENTS_CSV_PATH)
                load_events_csv.clear()
                st.session_state["events_df"] = edited.reset_index(drop=True)
                st.success("Eventos guardados en eventos_festivos.csv")
            except Exception as e:
                st.error(f"No se pudo guardar: {e}")

    with colB:
        if st.button("🗑️ Borrar todo (y guardar vacío)", key="ev_clear"):
            try:
                empty = pd.DataFrame(columns=["fecha_inicio","fecha_fin","uplift_pct","nombre","prioridad"])
                save_events_csv(empty, EVENTS_CSV_PATH)
                load_events_csv.clear()
                st.session_state["events_df"] = empty
                st.success("Lista vaciada.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al borrar: {e}")

    with colC:
        upfile = st.file_uploader("📤 Importar Excel/CSV", type=["xlsx","csv"],
                                  help="3 columnas mínimas: fecha_inicio, fecha_fin, uplift_pct.", key="ev_upload")
        if upfile is not None:
            try:
                if upfile.name.lower().endswith(".csv"):
                    imp = pd.read_csv(upfile)
                else:
                    imp = pd.read_excel(upfile)
                tmp_path = "_tmp_import_events.csv"
                imp.to_csv(tmp_path, index=False)
                imp_norm = load_events_csv(tmp_path)
                os.remove(tmp_path)
                merged = pd.concat([edited, imp_norm], ignore_index=True)
                st.session_state["events_df"] = merged
                st.success(f"Importados {len(imp_norm)} eventos. Revisa y pulsa Guardar.")
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"No se pudo importar: {e}")

    with colD:
        if not edited.empty:
            csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Exportar CSV", data=csv_bytes, file_name="eventos_festivos.csv", mime="text/csv", key="ev_export")

    st.markdown("#### Vista diaria (resolviendo solapes)")
    expanded = expand_events_by_day(edited)
    if expanded.empty:
        st.info("Sin eventos diarios. Añade rangos arriba y guarda.")
    else:
        st.dataframe(expanded, use_container_width=True)
        st.caption("Si varios eventos caen el mismo día, se aplica la **mayor prioridad** y, si empatan, el **mayor uplift %**.")

# =============================
# MODO: Tarificación (beta)
# =============================
elif mode == "Tarificación (beta)":
    if raw is None:
        st.stop()

    st.subheader("💸 Tarificación (beta)")
    st.caption("Precio propuesto por día y apartamento usando P50 por apto, m_apto histórico, Pace, DOW, Lead y Eventos.")

    with st.sidebar:
        st.header("Rango y corte")
        today = pd.Timestamp.today()
        start_tar = st.date_input("Inicio tarificación", value=(today.to_period("M").start_time).date(), key="tar_start")
        end_tar   = st.date_input("Fin tarificación", value=(today.to_period("M").end_time).date(), key="tar_end")
        cutoff_tar= st.date_input("Fecha de corte (OTB/Pace)", value=today.date(), key="tar_cut")

        props_sel = st.multiselect(
            "Alojamientos",
            options=sorted(raw["Alojamiento"].unique()),
            default=sorted(raw["Alojamiento"].unique()),
            key="tar_props"
        )

        st.markdown("—")
        st.markdown("**Multiplicadores DOW & Pace**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            m_lun = st.number_input("Lunes", value=1.00, step=0.01, key="tar_m_lun")
            m_mar = st.number_input("Martes", value=1.00, step=0.01, key="tar_m_mar")
        with c2:
            m_mie = st.number_input("Miércoles", value=1.00, step=0.01, key="tar_m_mie")
            m_jue = st.number_input("Jueves", value=1.02, step=0.01, key="tar_m_jue")
        with c3:
            m_vie = st.number_input("Viernes", value=1.05, step=0.01, key="tar_m_vie")
            m_sab = st.number_input("Sábado", value=1.08, step=0.01, key="tar_m_sab")
        with c4:
            m_dom = st.number_input("Domingo", value=1.03, step=0.01, key="tar_m_dom")
            k_pace = st.number_input("k pace (±% por p.p.)", value=0.01, step=0.005,
                                     help="Ajuste por cada punto de diferencia vs LY en el D correspondiente. Cap ±15%.",
                                     key="tar_kpace")

        st.markdown("—")
        st.markdown("**Lead & límites**")
        lt_close  = st.slider("Lead cercano (días)", 0, 30, 7, key="tar_ltclose")
        m_lt_close= st.number_input("Multiplicador lead cercano", value=0.97, step=0.01, key="tar_mltclose")
        m_lt_far  = st.number_input("Multiplicador lead lejano (>21d)", value=1.02, step=0.01, key="tar_mltfar")
        adr_min   = st.number_input("ADR mínimo (€)", value=35.0, step=1.0, key="tar_adrmin")
        adr_max   = st.number_input("ADR máximo (€)", value=500.0, step=5.0, key="tar_adrmax")
        round_rule= st.selectbox("Redondeo final", ["Sin redondeo", "A 1€", "A 5€", "Terminar en ,99"], index=3, key="tar_round")

        st.markdown("—")
        st.markdown("**Base ADR**")
        use_p50_by_apto = st.checkbox("Usar P50 por apartamento (base ADR)", value=True, key="tar_usep50",
                                      help="Si se desactiva, usa P50 del grupo.")

        st.markdown("—")
        st.markdown("**m_apto histórico (LY)**")
        m_apto_strength = st.slider("Intensidad m_apto (0–100%)", 0, 100, 60, 5, key="tar_mapto_str",
                                    help="0% sin efecto; 100% aplica m_apto capado por los límites abajo.")
        m_apto_cap = st.slider("Límites m_apto [min, max] (%)", 70, 140, (85, 115), 1, key="tar_mapto_cap",
                               help="Sugerido ±15%.")
        st.caption("Eventos se leen de 'Eventos & Festivos'.")
        run_tar = st.button("Calcular tarifas", type="primary", key="tar_run")

    if pd.to_datetime(start_tar) > pd.to_datetime(end_tar):
        st.error("El inicio no puede ser posterior al fin.")
        st.stop()

    # Cargar eventos expandidos por día
    try:
        eventos_daily = expand_events_by_day(load_events_csv(EVENTS_CSV_PATH))
    except Exception:
        eventos_daily = pd.DataFrame(columns=["fecha","uplift_pct","origen","prioridad"])

    dow_mult = {0:m_lun,1:m_mar,2:m_mie,3:m_jue,4:m_vie,5:m_sab,6:m_dom}

    if run_tar:
        days = pd.date_range(pd.to_datetime(start_tar), pd.to_datetime(end_tar), freq="D")
        if len(days) == 0:
            st.info("No hay días en el rango.")
            st.stop()

        inv_override = None
        period_start = pd.to_datetime(start_tar).to_period("M").start_time
        period_end   = pd.to_datetime(end_tar).to_period("M").end_time

        if use_p50_by_apto:
            base_p50_map = adr_bands_p50_for_month_by_apto(raw, pd.to_datetime(cutoff_tar), period_start, period_end, props_sel)
        else:
            base_p50_value = adr_bands_p50_for_month(raw, pd.to_datetime(cutoff_tar), period_start, period_end, props_sel)
            base_p50_map = {a: base_p50_value for a in props_sel}

        m_apto_raw = compute_m_apto_by_property(raw, pd.to_datetime(cutoff_tar), pd.to_datetime(start_tar), pd.to_datetime(end_tar), props_sel)
        m_min = m_apto_cap[0] / 100.0
        m_max = m_apto_cap[1] / 100.0
        strength = m_apto_strength / 100.0
        m_apto = {}
        for a in props_sel:
            base = m_apto_raw.get(a, 1.0)
            base_capped = float(np.clip(base, m_min, m_max))
            blended = 1.0 * (1.0 - strength) + base_capped * strength
            m_apto[a] = blended

        # Contexto vectorial
        ctx = pd.DataFrame({"Fecha": days})
        ctx["Fecha"] = pd.to_datetime(ctx["Fecha"]).dt.normalize()
        ctx["dow"] = ctx["Fecha"].dt.weekday
        ctx["m_dow"] = ctx["dow"].map(dow_mult).astype(float)
        lead_days = (ctx["Fecha"] - pd.to_datetime(cutoff_tar).normalize()).dt.days
        ctx["m_lead"] = 1.0
        ctx.loc[lead_days <= lt_close, "m_lead"] = m_lt_close
        ctx.loc[lead_days > 21, "m_lead"] = m_lt_far
        ctx["D"] = np.clip(lead_days, 0, None).astype(int)
        ev_daily = expand_events_by_day(load_events_csv(EVENTS_CSV_PATH))
        if not ev_daily.empty:
            ev_daily = ev_daily.copy()
            ev_daily["Fecha"] = pd.to_datetime(ev_daily["fecha"]).dt.normalize()
            ev_daily = ev_daily[["Fecha", "uplift_pct", "origen"]].drop_duplicates("Fecha")
            ctx = ctx.merge(ev_daily, on="Fecha", how="left")
            ctx["m_event"] = 1.0 + ctx["uplift_pct"].fillna(0.0) / 100.0
            ctx["Evento"] = ctx["origen"].fillna("")
        else:
            ctx["m_event"] = 1.0
            ctx["Evento"] = ""

        fast_mode = st.checkbox("⚡ Modo rápido (pace por grupo)", value=True, key="tar_fast",
                                help="Calcula Δ pace vs LY una única vez (grupo). Más rápido.")
        dmax = int(max(180, ctx["D"].max()))
        if fast_mode:
            base = pace_series(raw, pd.to_datetime(start_tar), pd.to_datetime(end_tar), dmax, props_sel, inv_override)
            ly   = pace_series(raw, pd.to_datetime(start_tar)-pd.DateOffset(years=1), pd.to_datetime(end_tar)-pd.DateOffset(years=1), dmax, props_sel, inv_override)
            if not base.empty and not ly.empty:
                m_now = base.set_index("D")["ocupacion_pct"].reindex(range(0, dmax+1)).interpolate().bfill().ffill()
                m_ly  = ly.set_index("D")["ocupacion_pct"].reindex(range(0, dmax+1)).interpolate().bfill().ffill()
                delta_pp_map = (m_now - m_ly).to_dict()
            else:
                delta_pp_map = {int(d): 0.0 for d in range(0, dmax + 1)}
            ctx["Δpp_pace"] = ctx["D"].map(delta_pp_map).fillna(0.0)
        else:
            ctx["Δpp_pace"] = 0.0

        # Diagnóstico
        st.markdown("#### Diagnóstico bases (P50 y m_apto)")
        diag = pd.DataFrame({
            "Alojamiento": props_sel,
            "P50 base": [base_p50_map.get(a, np.nan) for a in props_sel],
            "m_apto_raw": [m_apto_raw.get(a, np.nan) for a in props_sel],
            "m_apto_aplicado": [m_apto.get(a, 1.0) for a in props_sel],
        })
        st.dataframe(diag, use_container_width=True)

        # Cálculo por alojamiento (vectorial por columnas)
        results = []
        pbar = st.progress(0.0)
        for i, aloj in enumerate(props_sel, start=1):
            ADR_base = base_p50_map.get(aloj, np.nan)
            if not np.isfinite(ADR_base) or ADR_base <= 0:
                f = pace_forecast_month(raw, pd.to_datetime(cutoff_tar), period_start, period_end, 2, 180, [aloj], inv_override)
                ADR_base = float(f.get("adr_tail_p50", np.nan))
                if not np.isfinite(ADR_base) or ADR_base <= 0:
                    ADR_base = 60.0
            ADR_base = ADR_base * float(m_apto.get(aloj, 1.0))

            if fast_mode:
                delta_pp = ctx["Δpp_pace"].values
            else:
                base_a = pace_series(raw, pd.to_datetime(start_tar), pd.to_datetime(end_tar), dmax, [aloj], inv_override)
                ly_a   = pace_series(raw, pd.to_datetime(start_tar)-pd.DateOffset(years=1), pd.to_datetime(end_tar)-pd.DateOffset(years=1), dmax, [aloj], inv_override)
                if not base_a.empty and not ly_a.empty:
                    now_s = base_a.set_index("D")["ocupacion_pct"].reindex(range(0, dmax+1)).interpolate().bfill().ffill()
                    ly_s  = ly_a.set_index("D")["ocupacion_pct"].reindex(range(0, dmax+1)).interpolate().bfill().ffill()
                    dmap  = (now_s - ly_s).to_dict()
                    delta_pp = ctx["D"].map(dmap).fillna(0.0).values
                else:
                    delta_pp = np.zeros(len(ctx), dtype=float)

            m_dow_arr   = ctx["m_dow"].values.astype(float)
            m_event_arr = ctx["m_event"].values.astype(float)
            m_lead_arr  = ctx["m_lead"].values.astype(float)
            m_pace_arr  = np.clip(1.0 + k_pace * delta_pp, 0.85, 1.15)

            price_raw = ADR_base * m_dow_arr * m_event_arr * m_lead_arr * m_pace_arr
            price_cap = np.clip(price_raw, adr_min, adr_max)

            round_rule = st.session_state.get("tar_round", "Terminar en ,99")
            if round_rule == "A 1€":
                price_final = np.round(price_cap)
            elif round_rule == "A 5€":
                price_final = (5 * np.round(price_cap / 5)).astype(float)
            elif round_rule == "Terminar en ,99":
                price_final = np.floor(price_cap) + 0.99
            else:
                price_final = price_cap

            df_out = pd.DataFrame({
                "Alojamiento": aloj,
                "Fecha": ctx["Fecha"].values,
                "Precio propuesto": price_final,
                "ADR_base": ADR_base,
                "m_apto": float(m_apto.get(aloj, 1.0)),
                "m_dow": m_dow_arr,
                "m_pace": m_pace_arr,
                "Δpp pace vs LY": delta_pp,
                "m_event": m_event_arr,
                "Evento": ctx["Evento"].values,
                "m_lead": m_lead_arr,
                "Cap_minmax": price_raw != price_cap,
            })
            results.append(df_out)
            pbar.progress(i / len(props_sel))

        result_df = pd.concat(results, ignore_index=True)
        st.success(f"Calculadas {len(result_df):,} celdas de tarifa.".replace(",", "."))

        st.dataframe(result_df.head(500), use_container_width=True)
        grid_wide, meta_cols = build_pricing_calendar_grid(result_df, ev_daily)
        if not grid_wide.empty:
            styled = style_pricing_calendar(grid_wide, meta_cols)
            st.dataframe(styled, use_container_width=True)

        st.session_state["pricing_grid"] = grid_wide
        st.session_state["pricing_grid_meta"] = meta_cols

        # Descargas
        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Exportar tarifas (CSV)", data=csv_bytes, file_name="tarifas_propuestas.csv", mime="text/csv", key="tar_csv")

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            result_df.to_excel(writer, index=False, sheet_name="Tarifas")
            if not grid_wide.empty:
                grid_wide.to_excel(writer, sheet_name="Calendario")
            diag.to_excel(writer, index=False, sheet_name="Diagnóstico")
            ctx_export = ctx.copy()
            if "uplift_pct" not in ctx_export.columns: ctx_export["uplift_pct"] = np.nan
            if "origen" not in ctx_export.columns: ctx_export["origen"] = ""
            ctx_export.to_excel(writer, index=False, sheet_name="Contexto")
        st.download_button(
            "📥 Exportar Excel (Tarifas+Calendario+Diag)",
            data=output.getvalue(),
            file_name="tarifas_propuestas.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="tar_xlsx"
        )

# =============================
# MODO: Calendario de tarifas (opcional)
# =============================
elif mode == "Calendario de tarifas":
    st.subheader("📅 Calendario de tarifas (desde Tarificación)")
    grid = st.session_state.get("pricing_grid")
    meta = st.session_state.get("pricing_grid_meta")
    if grid is None or meta is None or grid.empty:
        st.info("No hay datos aún. Genera tarifas en el modo 'Tarificación (beta)'.")
    else:
        unique_months = sorted({pd.Timestamp(c).to_period("M").strftime("%Y-%m") for c in grid.columns})
        month_sel = st.selectbox("Mes a mostrar", unique_months, index=0, key="cal_mes")
        month_cols = [c for c in grid.columns if pd.Timestamp(c).to_period("M").strftime("%Y-%m") == month_sel]
        grid_m = grid[month_cols]
        meta_m = meta.loc[month_cols]

        styled = style_pricing_calendar(grid_m, meta_m)
        st.dataframe(styled, use_container_width=True)

        csv_month = grid_m.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Descargar mes (CSV)", data=csv_month, file_name=f"tarifas_{month_sel}.csv", mime="text/csv", key="cal_csv")

        st.caption("Gris = fin de semana · Amarillo = evento")

# ---------- Resumen Comparativo (por alojamiento) ----------
elif mode == "Resumen Comparativo":
    if raw is None:
        st.warning("⚠️ No hay datos cargados. Sube tus Excel y pulsa **Usar estos archivos** en la barra lateral.")
        st.stop()

    with st.sidebar:
        st.header("Parámetros – Resumen comparativo")
        cutoff_rc = st.date_input("Fecha de corte", value=date.today(), key="cut_resumen_comp")
        start_rc, end_rc = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "resumen_comp"
        )
        props_rc = st.multiselect(
            "Alojamientos (opcional)",
            options=sorted(raw["Alojamiento"].unique()),
            default=[],
            key="props_resumen_comp"
        )

    st.subheader("📊 Resumen Comparativo por alojamiento")

    days_period = (pd.to_datetime(end_rc) - pd.to_datetime(start_rc)).days + 1
    if days_period <= 0:
        st.error("El periodo no es válido (fin anterior o igual al inicio). Ajusta fechas.")
        st.stop()

    def _by_prop_with_occ(cutoff_dt, start_dt, end_dt, props_sel=None):
        by_prop, _ = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_dt),
            period_start=pd.to_datetime(start_dt),
            period_end=pd.to_datetime(end_dt),
            inventory_override=None,
            filter_props=props_sel if props_sel else None,
        )
        if by_prop.empty:
            return pd.DataFrame(columns=["Alojamiento","ADR","Ocupación %","Ingresos"])
        out = by_prop.copy()
        out["Ocupación %"] = (out["Noches ocupadas"] / days_period * 100.0).astype(float)
        return out[["Alojamiento","ADR","Ocupación %","Ingresos"]]

    props_sel = props_rc if props_rc else None

    now_df = _by_prop_with_occ(cutoff_rc, start_rc, end_rc, props_sel).rename(columns={
        "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
    })

    ly_df = _by_prop_with_occ(
        pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    ).rename(columns={
        "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
    })

    ly_final_df = _by_prop_with_occ(
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel
    )
    ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer").merge(ly_final_df, on="Alojamiento", how="left")

    if resumen.empty:
        st.info("No hay reservas que intersecten el periodo a la fecha de corte seleccionada.\n- Amplía el periodo o mueve la fecha de corte.\n- Recuerda que se incluyen reservas con **Fecha alta ≤ corte**.")
        st.stop()

    def _safe_pct(new, old):
        try:
            if pd.notna(new) and pd.notna(old) and float(old) != 0:
                return (float(new)/float(old) - 1.0) * 100.0
        except Exception:
            pass
        return np.nan

    resumen["Δ ADR vs LY (%)"] = [ _safe_pct(a, b) for a,b in zip(resumen.get("ADR actual"), resumen.get("ADR LY")) ]
    resumen["Δ Ocupación vs LY (pp)"] = resumen.get("Ocupación actual %", np.nan) - resumen.get("Ocupación LY %", np.nan)
    resumen["Δ Ingresos vs LY (%)"] = [ _safe_pct(a, b) for a,b in zip(resumen.get("Ingresos actuales (€)"), resumen.get("Ingresos LY (€)")) ]

    resumen = resumen.reindex(columns=[
        "Alojamiento",
        "ADR actual","ADR LY","Δ ADR vs LY (%)",
        "Ocupación actual %","Ocupación LY %","Δ Ocupación vs LY (pp)",
        "Ingresos actuales (€)","Ingresos LY (€)","Δ Ingresos vs LY (%)",
        "Ingresos finales LY (€)",
    ])

    st.dataframe(resumen, use_container_width=True)

    csv_bytes = resumen.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV", data=csv_bytes,
                       file_name="resumen_comparativo.csv", mime="text/csv")

    import io
    buffer = io.BytesIO()
    try:
        import xlsxwriter
        engine = "xlsxwriter"
    except Exception:
        engine = "openpyxl"

    with pd.ExcelWriter(buffer, engine=engine) as writer:
        df_xlsx = resumen.copy()
        for _c in ["Ocupación actual %","Ocupación LY %","Δ ADR vs LY (%)","Δ Ingresos vs LY (%)"]:
            if _c in df_xlsx.columns:
                df_xlsx[_c] = pd.to_numeric(df_xlsx[_c], errors="coerce")/100.0
        df_xlsx.to_excel(writer, index=False, sheet_name="Resumen")

        if engine == "xlsxwriter":
            wb = writer.book
            ws = writer.sheets["Resumen"]

            for j, col in enumerate(df_xlsx.columns):
                try:
                    width = int(min(44, max(12, df_xlsx[col].astype(str).str.len().max() if not df_xlsx.empty else 12)))
                except Exception:
                    width = 18
                ws.set_column(j, j, width)

            fmt_green = wb.add_format({"bg_color": "#C6EFCE", "font_color": "#006100"})
            fmt_red   = wb.add_format({"bg_color": "#FFC7CE", "font_color": "#9C0006"})
            fmt_pct   = wb.add_format({"num_format": "0.0%"})
            fmt_pp    = wb.add_format({"num_format": "0.0"})
            fmt_eur   = wb.add_format({"num_format": "#,##0.00"})
            fmt_adr   = wb.add_format({"num_format": "#,##0.00"})

            cols = {name: idx for idx, name in enumerate(df_xlsx.columns)}
            for name, fmt in [
                ("ADR actual", fmt_adr),
                ("ADR LY", fmt_adr),
                ("Ocupación actual %", fmt_pct),
                ("Ocupación LY %", fmt_pct),
                ("Ingresos actuales (€)", fmt_eur),
                ("Ingresos LY (€)", fmt_eur),
                ("Ingresos finales LY (€)", fmt_eur),
                ("Δ ADR vs LY (%)", fmt_pct),
                ("Δ Ingresos vs LY (%)", fmt_pct),
                ("Δ Ocupación vs LY (pp)", fmt_pp),
            ]:
                if name in cols:
                    ws.set_column(cols[name], cols[name], None, fmt)

            def col_letter(c):
                s = ""
                n = c
                while True:
                    n, r = divmod(n, 26)
                    s = chr(65 + r) + s
                    if n == 0:
                        break
                    n -= 1
                return s

            nrows = len(df_xlsx) + 1
            r1, r2 = 1, nrows

            for a_name, b_name in [
                ("ADR actual", "ADR LY"),
                ("Ocupación actual %", "Ocupación LY %"),
                ("Ingresos actuales (€)", "Ingresos LY (€)"),
            ]:
                if a_name in cols and b_name in cols:
                    ca, cb = cols[a_name], cols[b_name]
                    la, lb = col_letter(ca), col_letter(cb)
                    base_row = r1 + 1
                    ws.conditional_format(r1, ca, r2, ca, {
                        "type": "formula",
                        "criteria": f"=${la}{base_row}>={lb}{base_row}",
                        "format": fmt_green,
                    })
                    ws.conditional_format(r1, ca, r2, ca, {
                        "type": "formula",
                        "criteria": f"=${la}{base_row}<{lb}{base_row}",
                        "format": fmt_red,
                    })

            for d_name in ["Δ ADR vs LY (%)", "Δ Ocupación vs LY (pp)", "Δ Ingresos vs LY (%)"]:
                if d_name in cols:
                    c = cols[d_name]
                    ws.conditional_format(r1, c, r2, c, {"type": "cell", "criteria": ">", "value": 0, "format": fmt_green})
                    ws.conditional_format(r1, c, r2, c, {"type": "cell", "criteria": "<", "value": 0, "format": fmt_red})

    st.download_button(
        "📥 Descargar Excel (.xlsx) con colores",
        data=buffer.getvalue(),
        file_name="resumen_comparativo_colores.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

