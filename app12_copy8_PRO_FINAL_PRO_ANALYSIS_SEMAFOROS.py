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
with st.sidebar:
    st.header("Menú principal")
    menu_options = {
        "KPIs": [
            "Consulta normal",
            "Resumen Comparativo",
            "KPIs por meses",
        ],
        "Evolución": [
            "Evolución por fecha de corte",
            "Pickup (entre dos cortes)",
            "Pace (curva D)",
            "Predicción (Pace)",
            "Cuadro de mando (PRO)",
        ],
        "Análisis avanzado": [
            "Lead time & LOS",
            "DOW heatmap",
            "ADR bands & Targets",
            "Pricing – Mapa eficiencia",
            "Cohortes (Alta × Estancia)",
            "Estacionalidad",
            "Ranking alojamientos",
            "Gap vs Target",
            "Pipeline 90–180 días",
            "Calidad de datos",
        ],
        "Visualización": [
            "Calendario por alojamiento",
            "Resumen & Simulador",
        ],
        "Tarifas & Eventos": [
            "Eventos & Festivos",
            "Tarificación (beta)",
            "Calendario de tarifas",
        ],
    }
    category = st.selectbox("Categoría", list(menu_options.keys()), key="menu_cat")
    mode = st.selectbox("Módulo", menu_options[category], key="mode_select")
    # Alias por compatibilidad con bloques existentes
    if mode == "Resumen comparativo":
        mode = "Resumen Comparativo"

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

# ---------- Resumen comparativo (por alojamiento) ----------
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
        evo_cut_start = st.date_input("Inicio de corte", value=date.today() - timedelta(days=30), key="evo_cut_start")
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
                "ingresos": float(tot["ingresos"]),
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
                    "ingresos": float(tot2["ingresos"]),
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
# MODO: Cuadro de mando (PRO)
# ===========================
elif mode == "Cuadro de mando (PRO)":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros – PRO")
        pro_cut = st.date_input("Fecha de corte", value=date.today(), key="pro_cut")
        pro_start, pro_end = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "pro_period"
        )
        props_pro = st.multiselect(
            "Alojamientos (opcional)",
            options=sorted(raw["Alojamiento"].unique()),
            default=[],
            key="pro_props"
        )
        inv_pro = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="pro_inv")
        inv_pro_ly = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="pro_inv_ly")
        ref_years_pro = st.slider("Años de referencia Pace", min_value=1, max_value=3, value=2, key="pro_ref_years")

    st.subheader("📊 Cuadro de mando (PRO)")

    # Actual y LYs
    by_prop_now, tot_now = compute_kpis(
        raw,
        pd.to_datetime(pro_cut),
        pd.to_datetime(pro_start),
        pd.to_datetime(pro_end),
        int(inv_pro) if inv_pro > 0 else None,
        props_pro if props_pro else None,
    )
    _, tot_ly_cut = compute_kpis(
        raw,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=1),
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )
    cutoff_ly_final = pd.to_datetime(pro_end) - pd.DateOffset(years=1)
    _, tot_ly_final = compute_kpis(
        raw,
        cutoff_ly_final,
        pd.to_datetime(pro_start) - pd.DateOffset(years=1),
        pd.to_datetime(pro_end) - pd.DateOffset(years=1),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )
    # Predicción Pace (para estado de ritmo y semáforos)
    pace_res = pace_forecast_month(
        raw,
        pd.to_datetime(pro_cut),
        pd.to_datetime(pro_start),
        pd.to_datetime(pro_end),
        ref_years=int(ref_years_pro),
        dmax=180,
        props=props_pro if props_pro else None,
        inv_override=int(inv_pro) if inv_pro > 0 else None,
    )

    # ====== Ingresos ======
    st.subheader("💶 Ingresos (periodo seleccionado)")
    g1, g2, g3 = st.columns(3)
    g1.metric("Ingresos actuales (€)", f"{tot_now['ingresos']:.2f}")
    g2.metric("Ingresos LY a este corte (€)", f"{tot_ly_cut['ingresos']:.2f}")
    g3.metric("Ingresos LY final (€)", f"{tot_ly_final['ingresos']:.2f}")

    # ====== ADR ======
    st.subheader("🏷️ ADR (a fecha de corte)")
    _, tot_ly2_cut = compute_kpis(
        raw,
        pd.to_datetime(pro_cut) - pd.DateOffset(years=2),
        pd.to_datetime(pro_start) - pd.DateOffset(years=2),
        pd.to_datetime(pro_end) - pd.DateOffset(years=2),
        int(inv_pro_ly) if inv_pro_ly > 0 else None,
        props_pro if props_pro else None,
    )
    a1, a2, a3 = st.columns(3)
    a1.metric("ADR actual (€)", f"{tot_now['adr']:.2f}")
    a2.metric("ADR LY (€)", f"{tot_ly_cut['adr']:.2f}")
    a3.metric("ADR LY-2 (€)", f"{tot_ly2_cut['adr']:.2f}")

    # Bandas ADR en tabla (P10, P50, P90)
    start_dt = pd.to_datetime(pro_start); end_dt = pd.to_datetime(pro_end)
    dfb = raw[(raw["Fecha alta"] <= pd.to_datetime(pro_cut))].dropna(
        subset=["Fecha entrada", "Fecha salida", "Alquiler con IVA (€)"]
    ).copy()
    if props_pro:
        dfb = dfb[dfb["Alojamiento"].isin(props_pro)]
    mask_b = ~((dfb["Fecha salida"] <= start_dt) | (dfb["Fecha entrada"] >= (end_dt + pd.Timedelta(days=1))))
    dfb = dfb[mask_b]
    if not dfb.empty:
        dfb["los"] = (dfb["Fecha salida"].dt.normalize() - dfb["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
        dfb["adr_reserva"] = dfb["Alquiler con IVA (€)"] / dfb["los"]
        dfb["Mes"] = dfb["Fecha entrada"].dt.to_period("M").astype(str)
        def _pct_cols(x):
            arr = x.dropna().values
            if arr.size == 0:
                return pd.Series({"P10": 0.0, "Mediana": 0.0, "P90": 0.0})
            return pd.Series({"P10": np.percentile(arr,10), "Mediana": np.percentile(arr,50), "P90": np.percentile(arr,90)})
        bands = dfb.groupby("Mes")["adr_reserva"].apply(_pct_cols).reset_index()
        bands_wide = bands.pivot(index="Mes", columns="level_1", values="adr_reserva").sort_index()
        st.dataframe(bands_wide[["P10","Mediana","P90"]], use_container_width=True)
        st.download_button(
            "📥 Descargar bandas ADR (CSV)",
            data=bands_wide[["P10","Mediana","P90"]].reset_index().to_csv(index=False).encode("utf-8-sig"),
            file_name="adr_bands_cdmpro.csv", mime="text/csv"
        )
    else:
        st.info("Sin datos suficientes para bandas ADR en el periodo.")

    # ====== Ocupación ======
    st.subheader("🏨 Ocupación (periodo seleccionado)")
    o1, o2, o3 = st.columns(3)
    o1.metric("Ocupación actual", f"{tot_now['ocupacion_pct']:.2f}%")
    o2.metric("Ocupación LY (a este corte)", f"{tot_ly_cut['ocupacion_pct']:.2f}%")
    o3.metric("Ocupación LY final", f"{tot_ly_final['ocupacion_pct']:.2f}%")
    st.caption("Actual y LY: reservas con Fecha alta ≤ corte. LY final: corte = fin del periodo LY.")

    # ====== Ritmo de reservas (Pace) ======
    st.subheader("🏁 Ritmo de reservas (Pace)")
    n_otb = float(pace_res.get("nights_otb", 0.0))
    n_p50 = float(pace_res.get("nights_p50", 0.0))
    pick_need = float(pace_res.get("pickup_needed_p50", 0.0))
    pick_typ50 = float(pace_res.get("pickup_typ_p50", 0.0))
    adr_tail_p50 = float(pace_res.get("adr_tail_p50", np.nan)) if pace_res else np.nan
    # OTB esperado ≈ noches P50 – pickup típico a este D
    expected_otb_typ = max(n_p50 - pick_typ50, 0.0)
    if expected_otb_typ > 0:
        ratio = n_otb / expected_otb_typ
        if ratio >= 1.10:
            pace_state = "🟢 Adelantado"
        elif ratio <= 0.90:
            pace_state = "🔴 Retrasado"
        else:
            pace_state = "🟠 En línea"
    else:
        pace_state = "—"
    p1, p2, p3 = st.columns(3)
    p1.metric("OTB noches", f"{n_otb:,.0f}".replace(",", "."))
    p2.metric("Forecast noches (P50)", f"{n_p50:,.0f}".replace(",", "."))
    p3.metric("Pickup necesario (→ P50)", f"{pick_need:,.0f}".replace(",", "."))
    st.caption(f"Ritmo: {pace_state} · Pickup típico (P50) ≈ {pick_typ50:,.0f} · ADR tail (P50) ≈ {adr_tail_p50:,.2f}".replace(",", "."))

    # ====== Evolución por fecha de corte: Ocupación (izq) y ADR (dcha) ======
    st.subheader("📈 Evolución por fecha de corte: Ocupación (izq) y ADR (dcha)")
    with st.expander("Ver evolución", expanded=True):
        evo_cut_start = st.date_input(
            "Inicio de corte", value=pd.to_datetime(pro_cut).date().replace(day=1), key="evo_cut_start_pro"
        )
        evo_cut_end   = st.date_input("Fin de corte", value=pd.to_datetime(pro_cut).date(), key="evo_cut_end_pro")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=int(inv_pro), step=1, key="inv_evo_pro")
        run_evo = st.button("Calcular evolución (Ocupación y ADR)", type="primary", key="btn_evo_pro")

        if run_evo:
            cstart = pd.to_datetime(evo_cut_start); cend = pd.to_datetime(evo_cut_end)
            if cstart > cend:
                st.error("El inicio del rango de corte no puede ser posterior al fin.")
            else:
                rows = []
                for c in pd.date_range(cstart, cend, freq="D"):
                    _, tot_now_e = compute_kpis(
                        df_all=raw,
                        cutoff=c,
                        period_start=pd.to_datetime(pro_start),
                        period_end=pd.to_datetime(pro_end),
                        inventory_override=int(inv_e) if inv_e > 0 else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    _, tot_ly_e = compute_kpis(
                        df_all=raw,
                        cutoff=c - pd.DateOffset(years=1),
                        period_start=pd.to_datetime(pro_start) - pd.DateOffset(years=1),
                        period_end=pd.to_datetime(pro_end) - pd.DateOffset(years=1),
                        inventory_override=int(inv_pro_ly) if (isinstance(inv_pro_ly, int) and inv_pro_ly > 0) else None,
                        filter_props=props_pro if props_pro else None,
                    )
                    rows.append({
                        "Corte": c.normalize(),
                        "occ_now": float(tot_now_e["ocupacion_pct"]),
                        "adr_now": float(tot_now_e["adr"]),
                        "occ_ly": float(tot_ly_e["ocupacion_pct"]),
                        "adr_ly": float(tot_ly_e["adr"]),
                    })
                evo_df = pd.DataFrame(rows)
                if evo_df.empty:
                    st.info("Sin datos en el rango seleccionado.")
                else:
                    occ_long = evo_df.melt(id_vars=["Corte"], value_vars=["occ_now","occ_ly"], var_name="serie", value_name="valor")
                    occ_long["serie"] = occ_long["serie"].map({"occ_now": "Ocupación actual", "occ_ly": "Ocupación LY"})
                    adr_long = evo_df.melt(id_vars=["Corte"], value_vars=["adr_now","adr_ly"], var_name="serie", value_name="valor")
                    adr_long["serie"] = adr_long["serie"].map({"adr_now": "ADR actual (€)", "adr_ly": "ADR LY (€)"})
                    nearest = alt.selection_point(fields=["Corte"], nearest=True, on="mousemove", empty="none")
                    occ_colors = {"Ocupación actual": "#1f77b4", "Ocupación LY": "#6baed6"}
                    adr_colors = {"ADR actual (€)": "#ff7f0e", "ADR LY (€)": "#fdae6b"}
                    occ_line = alt.Chart(occ_long).mark_line(strokeWidth=2, interpolate="monotone").encode(
                        x=alt.X("Corte:T", title="Fecha de corte"),
                        y=alt.Y("valor:Q", axis=alt.Axis(orient="left", title="Ocupación %")),
                        color=alt.Color("serie:N", scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None),
                        strokeDash=alt.condition("datum.serie == 'Ocupación LY'", alt.value([5,3]), alt.value([0,0])),
                        opacity=alt.condition("datum.serie == 'Ocupación LY'", alt.value(0.7), alt.value(1.0)),
                        tooltip=[alt.Tooltip("Corte:T","Día"), alt.Tooltip("serie:N","Serie"), alt.Tooltip("valor:Q","Valor", format=".2f")],
                    )
                    occ_pts = alt.Chart(occ_long).mark_point(size=80, filled=True).encode(
                        x="Corte:T", y="valor:Q",
                        color=alt.Color("serie:N", scale=alt.Scale(domain=list(occ_colors.keys()), range=[occ_colors[k] for k in occ_colors]), title=None)
                    ).transform_filter(nearest)
                    adr_line = alt.Chart(adr_long).mark_line(strokeWidth=2, interpolate="monotone").encode(
                        x=alt.X("Corte:T"),
                        y=alt.Y("valor:Q", axis=alt.Axis(orient="right", title="ADR (€)")),
                        color=alt.Color("serie:N", scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None),
                        strokeDash=alt.condition("datum.serie == 'ADR LY (€)'", alt.value([5,3]), alt.value([0,0])),
                        opacity=alt.condition("datum.serie == 'ADR LY (€)'", alt.value(0.7), alt.value(1.0)),
                        tooltip=[alt.Tooltip("Corte:T","Día"), alt.Tooltip("serie:N","Serie"), alt.Tooltip("valor:Q","Valor", format=".2f")],
                    )
                    adr_pts = alt.Chart(adr_long).mark_point(size=80, filled=True).encode(
                        x="Corte:T", y="valor:Q",
                        color=alt.Color("serie:N", scale=alt.Scale(domain=list(adr_colors.keys()), range=[adr_colors[k] for k in adr_colors]), title=None)
                    ).transform_filter(nearest)
                    selectors = alt.Chart(evo_df).mark_rule(opacity=0).encode(x="Corte:T").add_params(nearest)
                    vline = alt.Chart(evo_df).mark_rule(color="#666", strokeWidth=1).encode(x="Corte:T", opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
                    chart = alt.layer(selectors, occ_line, occ_pts, adr_line, adr_pts, vline).resolve_scale(y="independent").properties(height=380)
                    zoomx = alt.selection_interval(bind="scales", encodings=["x"])
                    st.altair_chart(chart.add_params(zoomx), use_container_width=True)
                    out = evo_df.rename(columns={
                        "occ_now":"Ocupación % (Actual)", "occ_ly":"Ocupación % (LY)",
                        "adr_now":"ADR (€) (Actual)", "adr_ly":"ADR (€) (LY)",
                    })
                    st.dataframe(out, use_container_width=True)
                    st.download_button("📥 Descargar evolución (CSV)", data=out.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="evolucion_occ_adr_cdmpro.csv", mime="text/csv")

    # ====== Semáforos y análisis ======
    st.subheader("🚦 Semáforos y análisis")
    tech_block = _kai_cdm_pro_analysis(
        tot_now=tot_now,
        tot_ly_cut=tot_ly_cut,
        tot_ly_final=tot_ly_final,
        pace=pace_res,
        price_ref_p50=None
    )
    st.markdown(tech_block)

