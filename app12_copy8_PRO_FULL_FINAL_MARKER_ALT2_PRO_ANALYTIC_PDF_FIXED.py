
# ===========================
# Florit Flats — OTB Analytics & Dashboard (PRO)
# ===========================
# Features:
# - File upload & caching
# - Core KPIs calculation at cutoff & period with proration
# - Consulta normal (KPIs + portal share)
# - Resumen comparativo (actual vs LY vs LY final) con export a CSV/XLSX
# - KPIs por meses (serie) con comparación LY
# - Evolución por fecha de corte (serie interactiva)
# - Cuadro de mando (PRO): KPIs, análisis automático (profesional), sensibilidad ADR→RevPAR,
#   detalle por alojamiento y exportación a PDF con branding (logo + #163e64) y
#   encabezado/pie fijos. (PDF sin matplotlib)
#
# Requisitos: pip install streamlit pandas numpy altair reportlab xlsxwriter openpyxl

import io
import os
from datetime import datetime, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ===========================
# Utilidades de datos
# ===========================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
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

def period_inputs(label_start: str, label_end: str, default_start: date, default_end: date, key_prefix: str) -> tuple[date, date]:
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

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv * max(days, 0)

    if df_cut.empty or days <= 0:
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
    cand_cols = [c for c in df_all.columns if c.lower() in ["agente/intermediario", "portal", "canal", "agente", "intermediario"]]
    if not cand_cols:
        return None
    portal_col = cand_cols[0]

    df = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df = df[df["Alojamiento"].isin(filter_props)]
    df = df.dropna(subset=["Fecha entrada", "Fecha salida", portal_col]).copy()
    if df.empty:
        return pd.DataFrame(columns=[portal_col, "Reservas", "% Reservas"])

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
        return pd.DataFrame(columns=[portal_col, "Reservas", "% Reservas"])

    df_sel = df.loc[mask]
    counts = df_sel.groupby(portal_col).size().reset_index(name="Reservas").sort_values("Reservas", ascending=False)
    total = counts["Reservas"].sum()
    counts["% Reservas"] = np.where(total > 0, counts["Reservas"] / total * 100.0, 0.0)
    counts.rename(columns={portal_col: "Portal"}, inplace=True)
    return counts

def daily_series(df_all: pd.DataFrame, cutoff: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, props: Optional[List[str]], inventory_override: Optional[int]) -> pd.DataFrame:
    days = list(pd.date_range(start, end, freq='D'))
    rows = []
    for d in days:
        _, tot = compute_kpis(
            df_all=df_all,
            cutoff=cutoff,
            period_start=d,
            period_end=d,
            inventory_override=inventory_override,
            filter_props=props,
        )
        rows.append({"Fecha": d.normalize(), **tot})
    return pd.DataFrame(rows)

def help_block(kind: str):
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
        "KPIs por meses": "Serie por **meses** con KPIs a la **misma fecha de corte**.",
        "Resumen": "Vista compacta + simulador.",
    }
    txt = texts.get(kind, None)
    if txt:
        with st.expander("ℹ️ Cómo leer esta sección", expanded=False):
            st.markdown(txt)

# ===========================
# Config de página
# ===========================
st.set_page_config(page_title="OTB Analytics – Florit Flats", layout="wide")
st.title("📊 OTB Analytics – KPIs & Dashboard")
st.caption("Sube tus Excel una vez, configura parámetros en la barra lateral y usa cualquiera de los modos.")

# -------- Sidebar: periodo global + ficheros --------
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

raw = st.session_state.get("raw_df")
file_names = st.session_state.get("file_names", [])

if raw is not None:
    with st.expander("📂 Archivos cargados"):
        st.write("**Lista:**", file_names)
        try:
            st.write(f"**Alojamientos detectados:** {raw['Alojamiento'].nunique()}")
        except Exception:
            pass
else:
    st.info("Sube archivos en la barra lateral y pulsa **Usar estos archivos** para empezar.")

# ---------------- Menú de modos ----------------
mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "Resumen comparativo",
        "KPIs por meses",
        "Evolución por fecha de corte",
        "Cuadro de mando (PRO)",
    ],
    key="mode_radio"
)

# =============================
# Vista: Consulta normal
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date.today(), key="cutoff_normal")
        start_normal, end_normal = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "normal"
        )
        inv_normal = st.number_input("Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1, key="inv_normal")
        props_normal = st.multiselect("Filtrar alojamientos (opcional)",
                                      options=sorted(raw["Alojamiento"].unique()),
                                      default=[], key="props_normal")

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

    port_df = compute_portal_share(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        filter_props=props_normal if props_normal else None,
    )
    st.subheader("Distribución por portal (reservas en el periodo)")
    if port_df is None:
        st.info("No se encontró la columna de portal/canal. Si tiene otro nombre, dímelo y lo mapeo.")
    elif port_df.empty:
        st.warning("No hay reservas del periodo a la fecha de corte para calcular distribución por portal.")
    else:
        port_view = port_df.copy()
        port_view["% Reservas"] = port_view["% Reservas"].round(2)
        st.dataframe(port_view, use_container_width=True)
        csv_port = port_view.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar distribución por portal (CSV)", data=csv_port,
                           file_name="portales_distribucion.csv", mime="text/csv")

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar detalle (CSV)", data=csv,
                           file_name="detalle_por_alojamiento.csv", mime="text/csv")

# =============================
# Resumen comparativo
# =============================
elif mode == "Resumen comparativo":
    if raw is None:
        st.warning("⚠️ No hay datos cargados.")
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
        props_rc = st.multiselect("Alojamientos (opcional)",
                                  options=sorted(raw["Alojamiento"].unique()),
                                  default=[], key="props_resumen_comp")

    st.subheader("📊 Resumen comparativo por alojamiento")
    _n_props = (len(props_rc) if props_rc else raw["Alojamiento"].nunique())
    st.caption(f"Periodo: **{pd.to_datetime(start_rc).date()} → {pd.to_datetime(end_rc).date()}** · "
               f"Corte: **{pd.to_datetime(cutoff_rc).date()}** · "
               f"Alojamientos en cálculo: **{_n_props}**")

    days_period = (pd.to_datetime(end_rc) - pd.to_datetime(start_rc)).days + 1
    if days_period <= 0:
        st.error("El periodo no es válido (fin anterior o igual al inicio).")
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

    props_sel_rc = props_rc if props_rc else None

    now_df = _by_prop_with_occ(cutoff_rc, start_rc, end_rc, props_sel_rc).rename(columns={
        "ADR":"ADR actual", "Ocupación %":"Ocupación actual %", "Ingresos":"Ingresos actuales (€)"
    })

    ly_df = _by_prop_with_occ(
        pd.to_datetime(cutoff_rc) - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel_rc
    ).rename(columns={
        "ADR":"ADR LY", "Ocupación %":"Ocupación LY %", "Ingresos":"Ingresos LY (€)"
    })

    ly_final_df = _by_prop_with_occ(
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        pd.to_datetime(start_rc) - pd.DateOffset(years=1),
        pd.to_datetime(end_rc)   - pd.DateOffset(years=1),
        props_sel_rc
    )
    ly_final_df = ly_final_df[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer") \
                    .merge(ly_final_df, on="Alojamiento", how="left")

    if resumen.empty:
        st.info("No hay reservas que intersecten el periodo a la fecha de corte seleccionada.")
        st.stop()

    resumen = resumen.reindex(columns=[
        "Alojamiento",
        "ADR actual","ADR LY",
        "Ocupación actual %","Ocupación LY %",
        "Ingresos actuales (€)","Ingresos LY (€)",
        "Ingresos finales LY (€)"
    ])

    st.dataframe(resumen, use_container_width=True)

    csv_bytes = resumen.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Descargar CSV", data=csv_bytes,
                       file_name="resumen_comparativo.csv", mime="text/csv")

    buffer = io.BytesIO()
    try:
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")
            ws = writer.sheets["Resumen"]
            for j, col in enumerate(resumen.columns):
                width = int(min(38, max(12, resumen[col].astype(str).str.len().max() if not resumen.empty else 12)))
                ws.set_column(j, j, width)
    except Exception:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            resumen.to_excel(writer, index=False, sheet_name="Resumen")

    st.download_button(
        "📥 Descargar Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="resumen_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =============================
# KPIs por meses
# =============================
elif mode == "KPIs por meses":
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

        if raw[["Fecha entrada","Fecha salida"]].dropna().empty:
            months_options = []
        else:
            _min = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).min()
            _max = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).max()
            months_options = [str(p) for p in pd.period_range(_min.to_period("M"), _max.to_period("M"), freq="M")] if pd.notna(_min) and pd.notna(_max) else []

        selected_months_m = st.multiselect("Meses a graficar (YYYY-MM)", options=months_options, default=[], key="months_months")
        metric_choice = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"])
        compare_m = st.checkbox("Comparar con año anterior (mismo mes)", value=False, key="cmp_months")

    st.subheader("📈 KPIs por meses (a fecha de corte)")
    help_block("KPIs por meses")
    METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

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
# Evolución por fecha de corte
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

        props_e = st.multiselect("Filtrar alojamientos (opcional)",
                                 options=sorted(raw["Alojamiento"].unique()),
                                 default=[], key="props_evo")
        inv_e      = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")

        selected_kpis = st.multiselect("KPIs a mostrar", ["Ocupación %","ADR (€)","RevPAR (€)"],
                                       default=["Ocupación %"], key="kpi_multi")

        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📈 Evolución de KPIs vs fecha de corte")

    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts   = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
            st.stop()

        rows_now = []
        for c in pd.date_range(cut_start_ts, cut_end_ts, freq="D"):
            _, tot = compute_kpis(
                df_all=raw, cutoff=c,
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

        key_map = {"Ocupación %":"ocupacion_pct","ADR (€)":"adr","RevPAR (€)":"revpar"}
        for lbl in selected_kpis:
            col = key_map[lbl]
            chart = alt.Chart(df_now).mark_line(point=True).encode(
                x="Corte:T",
                y=alt.Y(col, title=lbl),
                tooltip=["Corte:T", alt.Tooltip(col, format=".2f", title=lbl)]
            ).properties(height=300, title=f"Evolución de {lbl}")
            st.altair_chart(chart, use_container_width=True)

# =============================
# MODO: Cuadro de mando (PRO)
# =============================
elif mode == "Cuadro de mando (PRO)":
    if raw is None:
        st.warning("⚠️ No hay datos cargados. Sube tus Excel y pulsa **Usar estos archivos**.")
        st.stop()

    with st.sidebar:
        st.header("Parámetros – PRO")
        cutoff_pro = st.date_input("Fecha de corte", value=date.today(), key="cutoff_pro")
        start_pro, end_pro = period_inputs(
            "Inicio del periodo", "Fin del periodo",
            date(date.today().year, date.today().month, 1),
            (pd.Timestamp.today().to_period("M").end_time).date(),
            "pro"
        )
        props_pro = st.multiselect("Alojamientos (opcional)",
                                   options=sorted(raw["Alojamiento"].unique()),
                                   default=[], key="props_pro")
        inv_pro = st.number_input("Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1, key="inv_pro")
        inv_pro_prev = st.number_input("Inventario LY (opcional)", min_value=0, value=0, step=1, key="inv_pro_prev")

        st.markdown("—")
        st.caption("Sensibilidad ADR–RevPAR")
        sens_min = st.slider("Variación mínima ADR", -30, -5, value=-15, step=1, key="sens_min")
        sens_max = st.slider("Variación máxima ADR", 5, 30, value=15, step=1, key="sens_max")
        sens_step = st.select_slider("Paso (p.p.)", options=[1,2,5,10], value=5, key="sens_step")

    st.subheader("🧠 Cuadro de mando (PRO)")
    help_block("Resumen")

    props_sel = props_pro if props_pro else None
    inv_override_now = int(inv_pro) if inv_pro > 0 else None

    by_prop_now, tot_now = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_pro),
        period_start=pd.to_datetime(start_pro),
        period_end=pd.to_datetime(end_pro),
        inventory_override=inv_override_now,
        filter_props=props_sel
    )

    cutoff_ly = pd.to_datetime(cutoff_pro) - pd.DateOffset(years=1)
    start_ly  = pd.to_datetime(start_pro) - pd.DateOffset(years=1)
    end_ly    = pd.to_datetime(end_pro)   - pd.DateOffset(years=1)
    inv_override_ly = int(inv_pro_prev) if inv_pro_prev > 0 else None

    _bp_ly, tot_ly = compute_kpis(
        df_all=raw,
        cutoff=cutoff_ly,
        period_start=start_ly,
        period_end=end_ly,
        inventory_override=inv_override_ly,
        filter_props=props_sel
    )

    _bp_lyf, tot_lyf = compute_kpis(
        df_all=raw,
        cutoff=end_ly,  # corte = fin del periodo LY
        period_start=start_ly,
        period_end=end_ly,
        inventory_override=inv_override_ly,
        filter_props=props_sel
    )

    # Panel KPIs
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Ocupación", f"{tot_now['ocupacion_pct']:.2f}%", delta=f"{tot_now['ocupacion_pct']-tot_ly['ocupacion_pct']:.2f} p.p.")
    c2.metric("ADR (€)", f"{tot_now['adr']:.2f}", delta=f"{tot_now['adr']-tot_ly['adr']:.2f}")
    c3.metric("RevPAR (€)", f"{tot_now['revpar']:.2f}", delta=f"{tot_now['revpar']-tot_ly['revpar']:.2f}")
    c4.metric("Ingresos (€)", f"{tot_now['ingresos']:.2f}", delta=f"{tot_now['ingresos']-tot_ly['ingresos']:.2f}")
    c5.metric("Noches ocupadas", f"{tot_now['noches_ocupadas']:,}".replace(",", "."),
              delta=int(tot_now['noches_ocupadas']-tot_ly['noches_ocupadas']))
    c6.metric("Noches disponibles", f"{tot_now['noches_disponibles']:,}".replace(",", "."),
              delta=int(tot_now['noches_disponibles']-tot_ly['noches_disponibles']))

    st.caption(f"LY final RevPAR: **{tot_lyf['revpar']:.2f} €** · ADR: **{tot_lyf['adr']:.2f} €** · Ocup.: **{tot_lyf['ocupacion_pct']:.2f}%**")

    def _safe(v):
        return float(v) if (v is not None and pd.notna(v) and np.isfinite(v)) else 0.0

    occ_now, occ_ly = _safe(tot_now["ocupacion_pct"]), _safe(tot_ly["ocupacion_pct"])
    adr_now, adr_ly = _safe(tot_now["adr"]), _safe(tot_ly["adr"])
    rpar_now, rpar_ly = _safe(tot_now["revpar"]), _safe(tot_ly["revpar"])

    delta_occ_pp = occ_now - occ_ly
    delta_adr = adr_now - adr_ly
    delta_revpar = rpar_now - rpar_ly
    delta_revpar_pct = (delta_revpar / rpar_ly * 100.0) if rpar_ly > 0 else 0.0

    # ADR objetivo para igualar RevPAR LY con la ocupación actual
    nights_disp = _safe(tot_now["noches_disponibles"])
    nights_occ = _safe(tot_now["noches_ocupadas"])
    adr_needed = np.nan
    if nights_occ > 0 and nights_disp > 0 and rpar_ly > 0:
        adr_needed = rpar_ly * nights_disp / nights_occ

    # ======= Motor de análisis dinámico (texto profesional) =======
    def analytic_reflection_text(_occ_now, _occ_ly, _adr_now, _adr_ly, _rpar_now, _rpar_ly):
        # Tasas como 0-1
        _occ_now_r = _occ_now/100.0
        _occ_ly_r  = _occ_ly/100.0

        # Descomposición de RevPAR: RevPAR = ADR * Occ
        delta_revpar_local = _rpar_now - _rpar_ly
        occ_contrib  = _adr_ly * (_occ_now_r - _occ_ly_r)              # € por efecto ocupación
        adr_contrib  = (_adr_now - _adr_ly) * _occ_ly_r                # € por efecto ADR
        cross_term   = (_adr_now - _adr_ly) * (_occ_now_r - _occ_ly_r) # interacción
        d_rpar_pct = (delta_revpar_local/_rpar_ly*100.0) if _rpar_ly > 0 else 0.0

        # Señales con umbrales suaves
        d_occ_pp  = _occ_now - _occ_ly
        d_adr_eur = _adr_now - _adr_ly
        d_rpar_eur = delta_revpar_local

        occ_up  = d_occ_pp > 0.2
        occ_dn  = d_occ_pp < -0.2
        adr_up  = d_adr_eur > 0.5
        adr_dn  = d_adr_eur < -0.5

        lines = []
        dir_occ = "por encima" if d_occ_pp >= 0 else "por debajo"
        dir_rpar = "por encima" if d_rpar_eur >= 0 else "por debajo"
        lines.append(f"**Resumen:** Este año estás **{dir_occ}** en ocupación (**{_occ_now:.2f}%** vs **{_occ_ly:.2f}%**) y **{dir_rpar}** en RevPAR (**{_rpar_now:.2f} €** vs **{_rpar_ly:.2f} €**, {d_rpar_pct:+.1f}%).")

        if adr_dn and occ_up:
            if d_rpar_eur < 0:
                lines.append("**Lectura:** Has **bajado ADR** para ganar ocupación, pero **no ha compensado**: el RevPAR cae. El volumen adicional no cubre la pérdida de precio.")
                if abs(adr_contrib) > abs(occ_contrib) + abs(cross_term):
                    lines.append("**Causa principal:** La **caída de ADR** pesa más que el efecto positivo de ocupación.")
                else:
                    lines.append("**Causa principal:** La mejora de ocupación **no ha sido suficiente** para compensar la pérdida de ADR.")
            else:
                lines.append("**Lectura:** Has **bajado ADR** y **subido ocupación**, y **sí ha compensado**: el RevPAR sube. El mix volumen-precio ha sido eficiente.")
        elif adr_up and occ_dn:
            if d_rpar_eur >= 0:
                lines.append("**Lectura:** Has **protegido tarifa (ADR ↑)** sacrificando algo de ocupación, y **ha compensado**: el RevPAR mejora.")
            else:
                lines.append("**Lectura:** Aunque **subiste ADR**, la **pérdida de ocupación** fue mayor y el RevPAR cae. El punto de precio puede haber superado la elasticidad del mercado.")
        elif adr_up and occ_up:
            lines.append("**Lectura:** **ADR y ocupación mejoran a la vez**: señal de **demanda sólida** y **posicionamiento**. Es buen momento para testar incrementos de precio selectivos.")
        elif adr_dn and occ_dn:
            lines.append("**Lectura:** **ADR y ocupación caen** simultáneamente: alerta roja. Revisa competitividad, visibilidad y condiciones; puede haber un problema de posicionamiento o distribución.")
        else:
            if abs(d_rpar_eur) < max(1.0, 0.02*_rpar_ly):
                lines.append("**Lectura:** Cambios leves: el RevPAR se mantiene estable. Ajustes finos de precio o estancias mínimas pueden optimizar el rendimiento.")
            else:
                lines.append("**Lectura:** Patrón mixto; revisa segmentos, ventanas de reserva y días con mayor sensibilidad al precio.")

        if d_rpar_eur < 0 and adr_dn and occ_up:
            lines.append("**Implicación operativa:** Recupera **precio** en picos y limita descuentos en días de baja elasticidad. Revisa **estancias mínimas** y cercas tarifarias.")
        elif d_rpar_eur < 0 and adr_up and occ_dn:
            lines.append("**Implicación operativa:** Ajusta **precio por demanda** y mejora conversión (fotos, título, políticas). Usa ofertas tácticas en huecos sueltos sin erosionar tarifa base.")
        elif d_rpar_eur >= 0 and adr_up:
            lines.append("**Implicación operativa:** Mantén la **protección de tarifa**. Prueba incrementos controlados y consolida segmentos de alta disposición a pagar.")
        elif d_rpar_eur >= 0 and occ_up and adr_dn:
            lines.append("**Implicación operativa:** El enfoque de volumen funciona; monitoriza **coste marginal** y aplica cercas para no diluir ADR en fines de semana o eventos.")
        else:
            lines.append("**Implicación operativa:** Ajuste fino: sensibilidad por día de semana y eventos. A/B tests con restricciones y suplementos.")

        return "\n\n".join(lines)

    with st.expander("💬 Análisis automático", expanded=True):
        analytic_md = analytic_reflection_text(occ_now, occ_ly, adr_now, adr_ly, rpar_now, rpar_ly)
        st.markdown(analytic_md)
        if np.isfinite(adr_needed):
            st.markdown(f"**ADR objetivo** para igualar el **RevPAR LY** con tu **ocupación actual**: **{adr_needed:.2f} €** (ADR actual: {adr_now:.2f} €).")

    st.divider()

    # Sensibilidad ADR→RevPAR (manteniendo noches actuales)
    st.subheader("🧪 Sensibilidad ADR → RevPAR (noches actuales)")
    sens_df = pd.DataFrame()
    if tot_now["noches_ocupadas"] <= 0 or tot_now["noches_disponibles"] <= 0:
        st.info("No hay noches u oferta para construir la sensibilidad.")
    else:
        base_adr = adr_now
        base_nights = float(tot_now["noches_ocupadas"])
        base_disp = float(tot_now["noches_disponibles"])
        steps = list(range(sens_min, sens_max + 1, sens_step))
        rows = []
        for p in steps:
            factor = 1.0 + p / 100.0
            adr_new = base_adr * factor
            rev_new = adr_new * base_nights
            revpar_new = rev_new / base_disp
            rows.append({
                "Δ ADR (%)": p,
                "ADR sim (€)": round(adr_new, 2),
                "RevPAR sim (€)": round(revpar_new, 2),
                "Δ RevPAR vs actual (€)": round(revpar_new - rpar_now, 2),
                "Δ RevPAR vs LY (€)": round(revpar_new - rpar_ly, 2),
            })
        sens_df = pd.DataFrame(rows)
        st.dataframe(sens_df, use_container_width=True)

        # Descargar Excel con colores
        buffer_sens = io.BytesIO()
        try:
            with pd.ExcelWriter(buffer_sens, engine="xlsxwriter") as writer:
                sens_df.to_excel(writer, index=False, sheet_name="Sensibilidad")
                ws = writer.sheets["Sensibilidad"]
                green = writer.book.add_format({"font_color": "#006400"})
                red   = writer.book.add_format({"font_color": "#b00020"})
                for j, col in enumerate(sens_df.columns):
                    width = max(12, min(30, int(sens_df[col].astype(str).str.len().max()) + 2))
                    ws.set_column(j, j, width)
                col_idx = sens_df.columns.get_loc("Δ RevPAR vs actual (€)")
                ws.conditional_format(1, col_idx, len(sens_df), col_idx, {"type": "cell", "criteria": ">", "value": 0, "format": green})
                ws.conditional_format(1, col_idx, len(sens_df), col_idx, {"type": "cell", "criteria": "<", "value": 0, "format": red})
                col_idx2 = sens_df.columns.get_loc("Δ RevPAR vs LY (€)")
                ws.conditional_format(1, col_idx2, len(sens_df), col_idx2, {"type": "cell", "criteria": ">", "value": 0, "format": green})
                ws.conditional_format(1, col_idx2, len(sens_df), col_idx2, {"type": "cell", "criteria": "<", "value": 0, "format": red})
        except Exception:
            with pd.ExcelWriter(buffer_sens, engine="openpyxl") as writer:
                sens_df.to_excel(writer, index=False, sheet_name="Sensibilidad")

        st.download_button("📥 Descargar sensibilidad (.xlsx)",
                           data=buffer_sens.getvalue(),
                           file_name="sensibilidad_adr_revpar.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()

    # Detalle por alojamiento (actual)
    st.subheader("🏠 Detalle por alojamiento (actual)")
    detail_now = pd.DataFrame()
    if by_prop_now.empty:
        st.info("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        detail_now = by_prop_now[["Alojamiento","Noches ocupadas","Ingresos","ADR"]].copy()
        days_period = (pd.to_datetime(end_pro) - pd.to_datetime(start_pro)).days + 1
        detail_now["Ocupación %"] = np.where(days_period > 0, detail_now["Noches ocupadas"] / days_period * 100.0, 0.0)
        st.dataframe(detail_now.sort_values("Ingresos", ascending=False), use_container_width=True)

        csv_det = detail_now.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar detalle actual (CSV)", data=csv_det, file_name="detalle_actual_por_alojamiento.csv", mime="text/csv")

    # --------- Exportar panel a PDF (ReportLab, sin matplotlib)
    st.divider()
    st.subheader("📑 Exportar panel (PDF con branding, encabezado/pie)")

    # Dependencia reportlab
    try:
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import mm
        rl_ok = True
    except Exception:
        rl_ok = False
        st.info("Para exportar a PDF instala ReportLab: `pip install reportlab`")

    if rl_ok and st.button("📥 Generar PDF (branding + header/footer)"):
        buffer_pdf = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer_pdf,
            pagesize=A4,
            leftMargin=18*mm, rightMargin=18*mm,
            topMargin=35*mm, bottomMargin=18*mm
        )

        styles = getSampleStyleSheet()
        style_title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#163e64"), spaceAfter=8)
        style_h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, textColor=colors.HexColor("#163e64"), spaceBefore=8, spaceAfter=6)
        style_normal = ParagraphStyle("normal", parent=styles["Normal"], fontSize=10, leading=14)

        def header_footer(canv: rl_canvas.Canvas, doc_):
            canv.saveState()
            width, height = A4
            x_left = doc_.leftMargin
            logo_path = "logo.png"
            if os.path.exists(logo_path):
                try:
                    canv.drawImage(logo_path, x_left, height - 20*mm, width=35*mm, height=12*mm, preserveAspectRatio=True, mask='auto')
                except Exception:
                    pass
            canv.setFont("Helvetica-Bold", 12)
            canv.setFillColorRGB(0.086, 0.243, 0.392)  # #163e64
            canv.drawString(x_left + 40*mm, height - 12*mm, "Cuadro de mando PRO — Florit Flats")
            canv.setFont("Helvetica", 9)
            canv.setFillColorRGB(0, 0, 0)
            sub1 = f"Periodo: {pd.to_datetime(start_pro).date()} → {pd.to_datetime(end_pro).date()}"
            sub2 = f"Corte: {pd.to_datetime(cutoff_pro).date()}"
            props_txt = ", ".join(props_pro) if props_pro else "Grupo completo"
            canv.drawString(x_left + 40*mm, height - 16*mm, sub1)
            canv.drawString(x_left + 40*mm, height - 20*mm, sub2 + "   |   " + f"Filtro: {props_txt}")
            canv.setLineWidth(0.5)
            canv.setStrokeColorRGB(0.6, 0.6, 0.6)
            canv.line(x_left, height - 22*mm, width - doc_.rightMargin, height - 22*mm)
            now_es = datetime.now().strftime("%Y-%m-%d %H:%M")
            canv.setFont("Helvetica", 8)
            canv.setFillColorRGB(0.2, 0.2, 0.2)
            canv.drawString(x_left, 12*mm, f"Generado: {now_es}")
            canv.drawRightString(width - doc_.rightMargin, 12*mm, f"Página {doc_.page}")
            canv.restoreState()

        flow = []
        flow.append(Paragraph("📊 Cuadro de mando PRO – Reporte", style_title))

        # KPIs
        flow.append(Paragraph("KPIs principales", style_h2))
        kpi_data = [
            ["Métrica", "Actual", "LY", "Δ"],
            ["Ocupación %", f"{occ_now:.2f}%", f"{occ_ly:.2f}%", f"{(occ_now-occ_ly):.2f} p.p."],
            ["ADR (€)", f"{adr_now:.2f}", f"{adr_ly:.2f}", f"{(adr_now-adr_ly):.2f}"],
            ["RevPAR (€)", f"{rpar_now:.2f}", f"{rpar_ly:.2f}", f"{(rpar_now-rpar_ly):.2f}"],
            ["Ingresos (€)", f"{tot_now['ingresos']:.2f}", f"{tot_ly['ingresos']:.2f}", f"{(tot_now['ingresos']-tot_ly['ingresos']):.2f}"],
        ]
        table_kpi = Table(kpi_data, hAlign="LEFT", repeatRows=1)
        table_kpi.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#163e64")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0,0), (-1,0), 6),
            ("TOPPADDING", (0,0), (-1,0), 6),
        ]))
        flow.append(table_kpi)
        flow.append(Spacer(1, 8))

        # Análisis (mismo texto que en la UI)
        flow.append(Paragraph("💬 Análisis", style_h2))
        analytic_md_pdf = analytic_reflection_text(occ_now, occ_ly, adr_now, adr_ly, rpar_now, rpar_ly)
        for line in analytic_md_pdf.split("\n\n"):
            flow.append(Paragraph(line.replace("**","<b>").replace("__","<b>").replace("*",""), style_normal))
        if np.isfinite(adr_needed):
            flow.append(Spacer(1, 4))
            flow.append(Paragraph(f"ADR objetivo para igualar el RevPAR LY con tu ocupación actual: {adr_needed:.2f} € (ADR actual: {adr_now:.2f} €).", style_normal))

        # Detalle por alojamiento
        if not detail_now.empty:
            flow.append(Spacer(1, 8))
            flow.append(Paragraph("🏠 Detalle por alojamiento (actual)", style_h2))
            detail_table = [list(detail_now.columns)] + detail_now.round(2).values.tolist()
            table_det = Table(detail_table, hAlign="LEFT", repeatRows=1)
            table_det.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
            ]))
            flow.append(table_det)

        doc.build(flow, onFirstPage=header_footer, onLaterPages=header_footer)

        st.download_button(
            "⬇️ Descargar PDF Florit Flats",
            data=buffer_pdf.getvalue(),
            file_name="cuadro_mando_pro_floritflats.pdf",
            mime="application/pdf"
        )
