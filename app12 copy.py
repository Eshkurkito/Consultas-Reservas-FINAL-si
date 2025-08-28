# ===========================
# BLOQUE 1/5 — Núcleo & Utils
# ===========================
import io
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------
# Utilidades comunes
# ---------------------------

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas clave y tipos."""
    required = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]
    for col in required:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"], errors="coerce")
    df["Fecha salida"] = pd.to_datetime(df["Fecha salida"], errors="coerce")
    df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").fillna(0.0)
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

# =============================
# HELPERS – Eventos / ADR base / m_apto
# =============================
import os

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
            df["fecha_fin"] = pd.to_datetime(df["fecha_fin"], errors="coerce").dt.date
            df["uplift_pct"] = pd.to_numeric(df["uplift_pct"], errors="coerce")
            df["prioridad"] = pd.to_numeric(df["prioridad"], errors="coerce").fillna(1).astype(int)
            df = df.dropna(subset=["fecha_inicio","fecha_fin","uplift_pct"])
            return df.reset_index(drop=True)
        except Exception as e:
            st.warning(f"No pude leer {path}: {e}. Empezamos vacío.")
    return pd.DataFrame(columns=["fecha_inicio","fecha_fin","uplift_pct","nombre","prioridad"])

def save_events_csv(df: pd.DataFrame, path: str):
    out = df.copy()
    out["fecha_inicio"] = pd.to_datetime(out["fecha_inicio"]).dt.date
    out["fecha_fin"] = pd.to_datetime(out["fecha_fin"]).dt.date
    out.to_csv(path, index=False)

def expand_events_by_day(events_df: pd.DataFrame) -> pd.DataFrame:
    """Expande rangos a filas por día con uplift. Si hay solapes,
    gana mayor 'prioridad'; si empatan, mayor 'uplift_pct'."""
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
    props: list[str],
) -> dict[str, float]:
    """{alojamiento: P50 ADR_reserva} dentro del periodo seleccionado."""
    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Precio"])
    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Precio"] / df["los"]
    mask = ~((df["Fecha salida"] <= start) | (df["Fecha entrada"] >= (end + pd.Timedelta(days=1))))
    df = df[mask]
    if df.empty:
        return {}
    out = {}
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
    props: list[str],
) -> float:
    """P50 ADR_reserva del grupo dentro del periodo (una sola cifra)."""
    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Precio"])
    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Precio"] / df["los"]
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
    props: list[str] | None = None,
) -> dict[str, float]:
    """
    m_apto = ADR_P50_apto_LY / ADR_P50_grupo_LY, cap y blending lo aplicas en el modo.
    Exige al menos 3 reservas por apto para robustez.
    """
    cut_ly = pd.to_datetime(cutoff) - pd.DateOffset(years=1)
    start_ly = pd.to_datetime(start) - pd.DateOffset(years=1)
    end_ly = pd.to_datetime(end) - pd.DateOffset(years=1)

    df = df_all[(df_all["Fecha alta"] <= cut_ly)].copy()
    if props:
        df = df[df["Alojamiento"].isin(props)]
    df = df.dropna(subset=["Fecha entrada","Fecha salida","Precio"])

    mask = ~((df["Fecha salida"] <= start_ly) | (df["Fecha entrada"] >= (end_ly + pd.Timedelta(days=1))))
    df = df[mask]
    if df.empty:
        return {}

    df["los"] = (df["Fecha salida"].dt.normalize() - df["Fecha entrada"].dt.normalize()).dt.days.clip(lower=1)
    df["adr_reserva"] = df["Precio"] / df["los"]

    arr_group = df["adr_reserva"].dropna().values
    if arr_group.size == 0:
        return {}

    p50_group_ly = np.percentile(arr_group, 50)
    if not np.isfinite(p50_group_ly) or p50_group_ly <= 0:
        return {}

    out = {}
    for aloj, sub in df.groupby("Alojamiento"):
        arr = sub["adr_reserva"].dropna().values
        if arr.size >= 3:
            p50_apto_ly = np.percentile(arr, 50)
            if np.isfinite(p50_apto_ly) and p50_apto_ly > 0:
                out[aloj] = float(p50_apto_ly / p50_group_ly)
    return out


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

    price = df_cut["Precio"].values.astype('float64')
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
    if "Portal" not in df_all.columns:
        return None

    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df = df[df["Alojamiento"].isin(filter_props)]
    df = df.dropna(subset=["Fecha entrada", "Fecha salida", "Portal"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"]) 

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
        return pd.DataFrame(columns=["Portal", "Reservas", "% Reservas"]) 

    df_sel = df.loc[mask]
    counts = df_sel.groupby("Portal").size().reset_index(name="Reservas").sort_values("Reservas", ascending=False)
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
        e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
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
    price = df["Precio"].values.astype('float64')

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
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Pickup (entre dos cortes)",
        "Pace (curva D)",
        "Predicción (Pace)",
        "Pipeline 90–180 días",
        "Gap vs Target",
        "Lead time & LOS",
        "DOW heatmap",
        "ADR bands & Targets",
        "Pricing – Mapa eficiencia",
        "Cohortes (Alta × Estancia)",
        "Estacionalidad",
        "Ranking alojamientos",
        "Operativa",
        "Calidad de datos",
        "Calendario por alojamiento",
        "Resumen & Simulador",
        "Cuadro de mando (PRO)",
        "Eventos & Festivos",
        "Tarificación (beta)",

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

# ---------- Evolución por fecha de corte ----------
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date.today().replace(day=1), key="evo_cut_start")
        evo_cut_end = st.date_input("Fin de corte", value=date.today(), key="evo_cut_end")

        st.header("Periodo objetivo")
        evo_target_start, evo_target_end = period_inputs("Inicio del periodo", "Fin del periodo",
                                                         date(date.today().year, date.today().month, 1),
                                                         (pd.Timestamp.today().to_period("M").end_time).date(),
                                                         "evo_target")

        props_e = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_evo")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")
        metric_choice_e = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="metric_evo")
        compare_e = st.checkbox("Comparar con año anterior (alineado por día)", value=False, key="cmp_evo")
        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📉 Evolución de KPIs vs fecha de corte")
    help_block("Evolución por corte")
    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
        else:
            rows_e = []
            for c in pd.date_range(cut_start_ts, cut_end_ts, freq='D'):
                _bp, tot_c = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=pd.to_datetime(evo_target_start),
                    period_end=pd.to_datetime(evo_target_end),
                    inventory_override=int(inv_e) if inv_e > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_e.append({"Corte": c.normalize(), **tot_c})
            df_evo = pd.DataFrame(rows_e)

            if df_evo.empty:
                st.info("No hay datos para el rango seleccionado.")
            else:
                key_col = METRIC_MAP[metric_choice_e]
                idx = pd.to_datetime(df_evo["Corte"])
                plot_df = pd.DataFrame({"Actual": df_evo[key_col].values}, index=idx)

                if compare_e:
                    rows_prev = []
                    cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
                    cut_end_prev = cut_end_ts - pd.DateOffset(years=1)
                    target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
                    target_end_prev = pd.to_datetime(evo_target_end) - pd.DateOffset(years=1)
                    prev_dates = list(pd.date_range(cut_start_prev, cut_end_prev, freq='D'))
                    for c in prev_dates:
                        _bp2, tot_c2 = compute_kpis(
                            df_all=raw,
                            cutoff=c,
                            period_start=target_start_prev,
                            period_end=target_end_prev,
                            inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                            filter_props=props_e if props_e else None,
                        )
                        rows_prev.append(tot_c2[key_col])
                    prev_idx_aligned = pd.to_datetime(prev_dates) + pd.DateOffset(years=1)
                    s_prev = pd.Series(rows_prev, index=prev_idx_aligned)
                    plot_df["Año anterior"] = s_prev.reindex(idx).values

                st.line_chart(plot_df, height=300)
                st.dataframe(df_evo[["Corte", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]]
                             .rename(columns={"noches_ocupadas": "Noches ocupadas", "noches_disponibles": "Noches disponibles",
                                              "ocupacion_pct": "Ocupación %", "adr": "ADR (€)", "revpar": "RevPAR (€)", "ingresos": "Ingresos (€)"}),
                             use_container_width=True)
                csve = df_evo.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar evolución (CSV)", data=csve, file_name="evolucion_kpis.csv", mime="text/csv")
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
            e, s, p = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
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
        df["adr_reserva"] = df["Precio"] / df["los"]
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
        df = df_all.dropna(subset=["Fecha entrada", "Fecha salida", "Precio"]).copy()
        rows = []
        for _, r in df.iterrows():
            e, s, price = r["Fecha entrada"], r["Fecha salida"], float(r["Precio"])
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
    bad_price = dfq[(pd.to_numeric(dfq['Precio'], errors='coerce').fillna(0) <= 0)]
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
# BLOQUE 5/5 — Resumen + Cuadro de mando PRO (con Pace YoY)
# ===========================

# ---------- Resumen & Simulador ----------
elif mode == "Resumen & Simulador":
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
        ref_years_r = st.slider("Años de referencia (pace)", 1, 3, 2)
        dmax_r = st.slider("D máximo pace", 60, 365, 180, 10)
        st.markdown("—")
        st.subheader("Simulador")
        delta_price = st.slider("Ajuste ADR del remanente (%)", -30, 30, 0, 1)
        elasticity = st.slider("Elasticidad de demanda", -1.5, -0.2, -0.8, 0.1)
        run_r = st.button("Calcular resumen", type="primary")

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

# ---------- Cuadro de mando PRO (Pace YoY + ADR bands por props + ADR histórico con delta/semáforo) ----------
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
        dmax_pro = st.slider("D máximo para Pace YoY", 30, 180, 90, 10, help="Ventana de comparación del ritmo (Días antes de la estancia).")

        # 👇 NUEVOS controles para ADR histórico y semáforo
        years_back = st.slider(
            "Años anteriores para comparar ADR",
            min_value=0, max_value=3, value=2, step=1,
            help="Compara el ADR del periodo con el de los mismos periodos de años previos."
        )
        delta_ok_band = st.slider(
            "Umbral semáforo ADR vs LY (±%)",
            min_value=1, max_value=15, value=5, step=1,
            help="Rango donde consideramos que el ADR actual está en línea con LY."
        )

        run_pro = st.button("Generar cuadro PRO", type="primary")

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

        # Health score (peso 40/30/30; ajusta umbrales a tu mercado)
        health = 0
        health += min(occ / 80 * 40, 40)     # 80% occ ≈ 40/40
        health += min(adr / 100 * 30, 30)    # 100€ ADR ≈ 30/30
        health += min(revpar / 60 * 30, 30)  # 60€ RevPAR ≈ 30/30
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
            2, 180,  # años de ref y Dmax por defecto
            props_sel,
            inv_now
        )
        st.markdown(
            f"**Pickup necesario**: {forecast['pickup_needed_p50']:.0f} · "
            f"Típico≈ {forecast['pickup_typ_p50']:.0f} (P75≈ {forecast['pickup_typ_p75']:.0f})"
        )

        # Ritmo (Pace) vs LY
        st.markdown("### Ritmo de ocupación (Pace) vs LY")
        base_pace = pace_series(
            raw,
            pd.to_datetime(start_pro), pd.to_datetime(end_pro),
            int(dmax_pro),
            props_sel, inv_now
        )
        ly_pace = pace_series(
            raw,
            pd.to_datetime(start_pro) - pd.DateOffset(years=1),
            pd.to_datetime(end_pro) - pd.DateOffset(years=1),
            int(dmax_pro),
            props_sel, inv_now
        )
        if not base_pace.empty and not ly_pace.empty:
            comp = pd.DataFrame({"D": base_pace["D"]})
            comp["Occ % actual"] = base_pace["ocupacion_pct"].values
            ly_vals = ly_pace["ocupacion_pct"].values
            if len(ly_vals) < len(comp):
                ly_vals = np.pad(ly_vals, (0, len(comp) - len(ly_vals)), constant_values=np.nan)
            comp["Occ % LY"] = ly_vals[:len(comp)]
            st.line_chart(comp.set_index("D")[["Occ % actual", "Occ % LY"]], height=260)

            # Hitos D-60 / D-30 / D-14 con delta p.p.
            def _val_at(df, D, col):
                r = df.loc[df["D"] == int(D), col]
                return float(r.values[0]) if len(r) else np.nan
            hitos = [60, 30, 14]
            cols_h = st.columns(len(hitos))
            for i, D in enumerate(hitos):
                now_v = _val_at(base_pace, D, "ocupacion_pct")
                ly_v = _val_at(ly_pace, D, "ocupacion_pct")
                if np.isfinite(now_v) and np.isfinite(ly_v):
                    delta_pp = now_v - ly_v
                    cols_h[i].metric(f"D-{D}", f"{now_v:.1f}% vs {ly_v:.1f}%", delta=f"{delta_pp:+.1f} pp")
                else:
                    cols_h[i].metric(f"D-{D}", "—", delta="—")
        else:
            st.info("No hay datos suficientes para comparar el ritmo con LY en este periodo.")

        # Bandas ADR (filtradas por alojamientos seleccionados y periodo)
        st.markdown("### Bandas de ADR (posicionamiento de precio)")
        dfb = raw[raw["Fecha alta"] <= pd.to_datetime(cutoff_pro)].dropna(subset=["Fecha entrada", "Fecha salida"]).copy()
        if props_sel:
            dfb = dfb[dfb["Alojamiento"].isin(props_sel)]
        dfb["los"] = (dfb["Fecha salida"] - dfb["Fecha entrada"]).dt.days.clip(lower=1)
        dfb["adr_reserva"] = dfb["Precio"] / dfb["los"]
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

        # ADR histórico (LY, LY-2, LY-3) con delta, semáforo e interpretación
        st.markdown("### ADR histórico del periodo")
        adr_hist = []  # [(label, value)]
        if years_back > 0:
            for k in range(1, years_back + 1):
                cut_k = pd.to_datetime(cutoff_pro) - pd.DateOffset(years=k)
                start_k = pd.to_datetime(start_pro) - pd.DateOffset(years=k)
                end_k = pd.to_datetime(end_pro) - pd.DateOffset(years=k)
                _bp_k, tot_k = compute_kpis(
                    df_all=raw,
                    cutoff=cut_k,
                    period_start=start_k,
                    period_end=end_k,
                    inventory_override=int(inv_now) if inv_now else None,  # inventario no afecta ADR
                    filter_props=props_sel
                )
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

        st.markdown("### Interpretación ADR vs LY")
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

        # Recomendaciones rápidas (contextuales a occ/ADR/pickup)
        recs = []
        if occ < 60:
            recs.append("⚠️ Ocupación baja: prueba bajar mínimos/estancias y abrir huecos cortos.")
        if np.isfinite(q25) and adr < q25:
            recs.append("💡 ADR bajo: parece más un problema de visibilidad que de precio. Empuja OTAs, Google Hotel Ads y campañas sociales.")
        elif np.isfinite(q75) and adr > q75:
            recs.append("📉 ADR alto: probable sobreprecio. Baja ADR en días flojos y usa promos selectivas (última hora / LTS).")
        if forecast['pickup_needed_p50'] > forecast['pickup_typ_p75']:
            if np.isfinite(q25) and adr < q25:
                recs.append("🚨 Pickup muy retrasado con ADR bajo → foco en visibilidad/posicionamiento en portales.")
            elif np.isfinite(q75) and adr > q75:
                recs.append("🚨 Pickup muy retrasado con ADR alto → baja tarifas, abre disponibilidad y lanza promos flash.")
            else:
                recs.append("🚨 Pickup muy retrasado → revisa mínimos y activa campañas flash de demanda.")

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

    ev_df = load_events_csv(EVENTS_CSV_PATH).copy()

    with st.expander("➕ Añadir evento"):
        c1, c2, c3 = st.columns(3)
        with c1:
            fi = st.date_input("Fecha inicio", value=None, key="ev_fi")
        with c2:
            ff = st.date_input("Fecha fin", value=None, key="ev_ff")
        with c3:
            up = st.number_input("Uplift %", min_value=-50.0, max_value=200.0, value=10.0, step=1.0)
        c4, c5 = st.columns(2)
        with c4:
            nombre = st.text_input("Nombre (opcional)", value="", key="ev_nombre")
        with c5:
            prioridad = st.number_input("Prioridad (1..9)", min_value=1, max_value=9, value=1, step=1,
                                        help="Si hay solapes, gana la mayor prioridad; si empatan, mayor uplift %.")
        if st.button("Agregar a la lista", type="primary"):
            if fi and ff and up is not None:
                new_row = {
                    "fecha_inicio": pd.to_datetime(fi).date(),
                    "fecha_fin": pd.to_datetime(ff).date(),
                    "uplift_pct": float(up),
                    "nombre": nombre.strip(),
                    "prioridad": int(prioridad),
                }
                ev_df = pd.concat([ev_df, pd.DataFrame([new_row])], ignore_index=True)
                st.success("Evento añadido. Recuerda pulsar **Guardar** para persistir.")
                st.experimental_rerun()
            else:
                st.warning("Completa inicio, fin y uplift %.")

    st.markdown("#### Lista de eventos (editable)")
    edited = st.data_editor(
        ev_df,
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
        if st.button("💾 Guardar", type="primary"):
            try:
                save_events_csv(edited, EVENTS_CSV_PATH)
                load_events_csv.clear()
                st.success("Eventos guardados en eventos_festivos.csv")
            except Exception as e:
                st.error(f"No se pudo guardar: {e}")
    with colB:
        if st.button("🗑️ Borrar todo (y guardar vacío)"):
            try:
                empty = pd.DataFrame(columns=["fecha_inicio","fecha_fin","uplift_pct","nombre","prioridad"])
                save_events_csv(empty, EVENTS_CSV_PATH)
                load_events_csv.clear()
                st.success("Lista vaciada.")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error al borrar: {e}")
    with colC:
        upfile = st.file_uploader("📤 Importar Excel/CSV", type=["xlsx","csv"],
                                  help="3 columnas mínimas: fecha_inicio, fecha_fin, uplift_pct.")
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
                st.session_state["events_editor"] = merged
                st.success(f"Importados {len(imp_norm)} eventos. Revisa y pulsa Guardar.")
            except Exception as e:
                st.error(f"No se pudo importar: {e}")
    with colD:
        if not edited.empty:
            csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
            st.download_button("⬇️ Exportar CSV", data=csv_bytes, file_name="eventos_festivos.csv", mime="text/csv")

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
        start_tar = st.date_input("Inicio tarificación", value=(pd.Timestamp.today().to_period("M").start_time).date())
        end_tar = st.date_input("Fin tarificación", value=(pd.Timestamp.today().to_period("M").end_time).date())
        cutoff_tar = st.date_input("Fecha de corte (OTB/Pace)", value=pd.Timestamp.today().date())

        props_sel = st.multiselect(
            "Alojamientos",
            options=sorted(raw["Alojamiento"].unique()),
            default=sorted(raw["Alojamiento"].unique())
        )

        st.markdown("—")
        st.markdown("**Multiplicadores DOW & Pace**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            m_lun = st.number_input("Lunes", value=1.00, step=0.01)
            m_mar = st.number_input("Martes", value=1.00, step=0.01)
        with c2:
            m_mie = st.number_input("Miércoles", value=1.00, step=0.01)
            m_jue = st.number_input("Jueves", value=1.02, step=0.01)
        with c3:
            m_vie = st.number_input("Viernes", value=1.05, step=0.01)
            m_sab = st.number_input("Sábado", value=1.08, step=0.01)
        with c4:
            m_dom = st.number_input("Domingo", value=1.03, step=0.01)
            k_pace = st.number_input("k pace (±% por p.p.)", value=0.01, step=0.005,
                                     help="Ajuste por cada punto de diferencia vs LY en el D correspondiente. Cap ±15%.")

        st.markdown("—")
        st.markdown("**Lead & límites**")
        lt_close = st.slider("Lead cercano (días)", 0, 30, 7)
        m_lt_close = st.number_input("Multiplicador lead cercano", value=0.97, step=0.01)
        m_lt_far = st.number_input("Multiplicador lead lejano (>21d)", value=1.02, step=0.01)
        adr_min = st.number_input("ADR mínimo (€)", value=35.0, step=1.0)
        adr_max = st.number_input("ADR máximo (€)", value=500.0, step=5.0)
        round_rule = st.selectbox("Redondeo final", ["Sin redondeo", "A 1€", "A 5€", "Terminar en ,99"], index=3)

        st.markdown("—")
        st.markdown("**Base ADR**")
        use_p50_by_apto = st.checkbox(
            "Usar P50 por apartamento (base ADR)",
            value=True,
            help="Si se desactiva, usa P50 del grupo."
        )

        st.markdown("—")
        st.markdown("**m_apto histórico (LY)**")
        m_apto_strength = st.slider(
            "Intensidad m_apto (0–100%)", 0, 100, 60, 5,
            help="0% sin efecto; 100% aplica m_apto capado por los límites abajo."
        )
        m_apto_cap = st.slider(
            "Límites m_apto [min, max] (%)", 70, 140, (85, 115), 1,
            help="Sugerido ±15%."
        )

        st.markdown("—")
        st.caption("Eventos se leen de 'Eventos & Festivos'.")
        run_tar = st.button("Calcular tarifas", type="primary")

    if pd.to_datetime(start_tar) > pd.to_datetime(end_tar):
        st.error("El inicio no puede ser posterior al fin.")
        st.stop()

    # --- Cargar eventos expandidos por día
    try:
        eventos_daily = expand_events_by_day(load_events_csv(EVENTS_CSV_PATH))
    except Exception:
        eventos_daily = pd.DataFrame(columns=["fecha","uplift_pct","origen","prioridad"])

    # --- Helpers internos del motor ---
    dow_mult = {0:m_lun,1:m_mar,2:m_mie,3:m_jue,4:m_vie,5:m_sab,6:m_dom}

    def redondear(v: float) -> float:
        if not np.isfinite(v):
            return v
        if round_rule == "A 1€":
            return round(v)
        elif round_rule == "A 5€":
            return int(5 * round(v/5))
        elif round_rule == "Terminar en ,99":
            return np.floor(v) + 0.99
        return v

    def event_multiplier_for_date(d: pd.Timestamp) -> tuple[float,str]:
        if eventos_daily is None or eventos_daily.empty:
            return 1.0, ""
        r = eventos_daily.loc[eventos_daily["fecha"] == d.date()]
        if r.empty:
            return 1.0, ""
        uplift = float(r["uplift_pct"].iloc[0])
        nombre = str(r["origen"].iloc[0]) if "origen" in r.columns else "Evento"
        return (1.0 + uplift/100.0), nombre

    def lead_multiplier_for_date(cutoff: pd.Timestamp, d: pd.Timestamp) -> float:
        lead = (d.normalize() - cutoff.normalize()).days
        if lead <= lt_close:
            return m_lt_close
        elif lead > 21:
            return m_lt_far
        return 1.0

    def pace_delta_vs_ly_pp(df_all, start, end, dmax: int, props: list[str], inv_override: int, cutoff: pd.Timestamp, d_for_day: int) -> float:
        """Δ pp de ocupación vs LY en el D indicado (para el periodo completo)."""
        base = pace_series(df_all, start, end, dmax, props, inv_override)
        ly = pace_series(df_all, start - pd.DateOffset(years=1), end - pd.DateOffset(years=1), dmax, props, inv_override)
        if base.empty or ly.empty:
            return 0.0
        D = int(np.clip(d_for_day, 0, dmax))
        now_row = base.loc[base["D"]==D, "ocupacion_pct"]
        ly_row = ly.loc[ly["D"]==D, "ocupacion_pct"]
        if not len(now_row) or not len(ly_row):
            return 0.0
        return float(now_row.values[0] - ly_row.values[0])

    def compute_price_for_cell(
        aloj: str,
        d: pd.Timestamp,
        base_p50_map: dict[str, float],
        inv_override: int | None,
        m_apto_map: dict[str, float],
    ) -> dict:
        """Precio propuesto + desglose por celda."""
        # Base ADR: P50 por apto (o P50 grupo) del periodo
        ADR_base = base_p50_map.get(aloj, np.nan)

        # Fallback 1: ADR OTB del día para ese apto
        if not np.isfinite(ADR_base) or ADR_base <= 0:
            ser = daily_series(raw, pd.to_datetime(cutoff_tar), d, d, [aloj], inv_override)
            ADR_base = float(ser["adr"].iloc[0]) if not ser.empty and np.isfinite(ser["adr"].iloc[0]) and ser["adr"].iloc[0] > 0 else np.nan

        # Fallback 2: ADR tail P50 del forecast del mes para ese apto
        if not np.isfinite(ADR_base) or ADR_base <= 0:
            month_start_local = d.to_period("M").start_time
            month_end_local = d.to_period("M").end_time
            f = pace_forecast_month(raw, pd.to_datetime(cutoff_tar), month_start_local, month_end_local, 2, 180, [aloj], inv_override)
            ADR_base = float(f.get("adr_tail_p50", np.nan))
            if not np.isfinite(ADR_base) or ADR_base <= 0:
                ADR_base = 60.0  # red mínima de seguridad

        # m_apto histórico (blending ya aplicado fuera)
        m_apto_factor = float(m_apto_map.get(aloj, 1.0))
        ADR_base = ADR_base * m_apto_factor

        # Multiplicadores diarios
        m_dow = float(dow_mult[d.weekday()])

        dmax = 180
        D = max((d.normalize() - pd.to_datetime(cutoff_tar)).days, 0)
        delta_pp = pace_delta_vs_ly_pp(raw, pd.to_datetime(start_tar), pd.to_datetime(end_tar), dmax, [aloj], None, pd.to_datetime(cutoff_tar), D)
        m_pace = 1.0 + k_pace * (delta_pp)
        m_pace = float(np.clip(m_pace, 0.85, 1.15))

        m_event, ev_name = event_multiplier_for_date(d)
        m_lead = lead_multiplier_for_date(pd.to_datetime(cutoff_tar), d)

        price_raw = ADR_base * m_dow * m_pace * m_event * m_lead
        price_cap = float(np.clip(price_raw, adr_min, adr_max))
        price_final = redondear(price_cap)

        return {
            "Alojamiento": aloj,
            "Fecha": d.normalize(),
            "Precio propuesto": price_final,
            "ADR_base": ADR_base,
            "m_apto": m_apto_factor,
            "m_dow": m_dow,
            "m_pace": m_pace,
            "Δpp pace vs LY": delta_pp,
            "m_event": m_event,
            "Evento": ev_name,
            "m_lead": m_lead,
            "Cap_minmax": (price_raw != price_cap),
        }

    # --- Ejecución del motor ---
    if run_tar:
        days = list(pd.date_range(pd.to_datetime(start_tar), pd.to_datetime(end_tar), freq="D"))
        if not days:
            st.info("No hay días en el rango.")
            st.stop()

        inv_override = None  # no imprescindible para ADR

        # Base ADR del periodo: P50 por apto o P50 del grupo
        period_start = pd.to_datetime(start_tar).to_period("M").start_time
        period_end = pd.to_datetime(end_tar).to_period("M").end_time
        if use_p50_by_apto:
            base_p50_map = adr_bands_p50_for_month_by_apto(raw, pd.to_datetime(cutoff_tar), period_start, period_end, props_sel)
        else:
            base_p50_value = adr_bands_p50_for_month(raw, pd.to_datetime(cutoff_tar), period_start, period_end, props_sel)
            base_p50_map = {aloj: base_p50_value for aloj in props_sel}

        # m_apto histórico (LY) + caps + intensidad
        m_apto_raw = compute_m_apto_by_property(raw, pd.to_datetime(cutoff_tar), pd.to_datetime(start_tar), pd.to_datetime(end_tar), props_sel)
        m_min = m_apto_cap[0]/100.0
        m_max = m_apto_cap[1]/100.0
        strength = m_apto_strength/100.0
        m_apto = {}
        for a in props_sel:
            base = m_apto_raw.get(a, 1.0)
            base_capped = float(np.clip(base, m_min, m_max))
            blended = 1.0*(1.0-strength) + base_capped*strength
            m_apto[a] = blended

        # Diagnóstico de bases
        st.markdown("#### Diagnóstico bases (P50 y m_apto)")
        diag = pd.DataFrame({
            "Alojamiento": props_sel,
            "P50 base": [base_p50_map.get(a, np.nan) for a in props_sel],
            "m_apto_raw": [m_apto_raw.get(a, np.nan) for a in props_sel],
            "m_apto_aplicado": [m_apto.get(a, 1.0) for a in props_sel],
        })
        st.dataframe(diag, use_container_width=True)

        # Cálculo
        out_rows = []
        for aloj in props_sel:
            for d in days:
                out_rows.append(
                    compute_price_for_cell(aloj, pd.to_datetime(d), base_p50_map, inv_override, m_apto)
                )
        result_df = pd.DataFrame(out_rows)
        st.success(f"Calculadas {len(result_df)} celdas de tarifa.")
        st.dataframe(result_df, use_container_width=True)

        # Export
        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Exportar tarifas (CSV)", data=csv_bytes, file_name="tarifas_propuestas.csv", mime="text/csv")

        # Impacto por evento y apartamento (heurístico)
        st.markdown("—")
        st.subheader("🔎 Impacto por evento y apartamento (heurístico)")
        expanded = eventos_daily.copy()
        if expanded.empty:
            st.info("No hay eventos diarios para evaluar impacto. Añádelos en el modo correspondiente.")
        else:
            expanded = expanded[expanded["fecha"].between(pd.to_datetime(start_tar).date(), pd.to_datetime(end_tar).date())]
            if expanded.empty:
                st.info("No hay eventos dentro del rango seleccionado.")
            else:
                impact_rows = []
                for aloj in props_sel:
                    for _, ev in expanded.iterrows():
                        d = pd.to_datetime(ev["fecha"])
                        uplift = float(ev["uplift_pct"])
                        d_ly = d - pd.DateOffset(years=1)
                        month_ly_start = d_ly.to_period("M").start_time
                        month_ly_end = d_ly.to_period("M").end_time

                        # P50 grupo LY y día LY del apto
                        p50_ly_apto_map = adr_bands_p50_for_month_by_apto(raw, pd.to_datetime(cutoff_tar) - pd.DateOffset(years=1), month_ly_start, month_ly_end, [aloj])
                        p50_ly_apto = p50_ly_apto_map.get(aloj, np.nan)
                        p50_group_ly = adr_bands_p50_for_month(raw, pd.to_datetime(cutoff_tar) - pd.DateOffset(years=1), month_ly_start, month_ly_end, props_sel)

                        _bp, tot_ly_day = compute_kpis(raw, pd.to_datetime(cutoff_tar) - pd.DateOffset(years=1), d_ly, d_ly, None, [aloj])
                        adr_ly_day = float(tot_ly_day["adr"]) if tot_ly_day["noches_ocupadas"] > 0 else np.nan

                        idx_hist = (adr_ly_day / p50_group_ly) if (np.isfinite(adr_ly_day) and np.isfinite(p50_group_ly) and p50_group_ly>0) else 1.0
                        _bp_now, tot_now_day = compute_kpis(raw, pd.to_datetime(cutoff_tar), d, d, None, [aloj])
                        otb_now = float(tot_now_day["noches_ocupadas"])

                        score = uplift * idx_hist * (1 + min(otb_now, 1)*0.2)
                        impact_rows.append({
                            "Alojamiento": aloj,
                            "Fecha evento": d.date(),
                            "Evento": str(ev.get("origen","")),
                            "Uplift %": uplift,
                            "ADR LY día": adr_ly_day,
                            "P50 grupo LY": p50_group_ly,
                            "Índice histórico": idx_hist,
                            "OTB noches (hoy)": otb_now,
                            "Impact score": score,
                        })
                if impact_rows:
                    impact_df = pd.DataFrame(impact_rows)
                    agg = impact_df.groupby(["Alojamiento","Evento"], as_index=False).agg({
                        "Uplift %":"mean",
                        "Índice histórico":"mean",
                        "OTB noches (hoy)":"sum",
                        "Impact score":"sum",
                    }).sort_values(["Impact score"], ascending=False)
                    st.dataframe(agg, use_container_width=True)
                    st.download_button("⬇️ Exportar impacto (CSV)",
                                       data=agg.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="impacto_eventos.csv",
                                       mime="text/csv")
                else:
                    st.info("No hubo suficiente histórico/OTB para calcular impacto.")
