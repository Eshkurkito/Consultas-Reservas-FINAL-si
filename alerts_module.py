from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ==============================
# Helpers: columnas y ventanas
# ==============================

def _get_config(config: Optional[dict]) -> dict:
    """Obtiene el mapeo de columnas desde config o st.session_state.config_cols."""
    if config is None:
        config = st.session_state.get("config_cols", {})
    return config or {}


def normalize_columns(df: pd.DataFrame, config: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Normaliza nombres/tipos y devuelve (df_norm, used_map).
    Campos estándar esperados:
      - unidad (str)
      - fecha_llegada (datetime)
      - ocupacion (float 0–100)
      - objetivo_ocupacion (float %, opcional)
      - adr (float)
      - adr_compset (float, opcional)
      - pickup_7d (float, opcional)
      - pace_vs_ly (float %, opcional)
      - zona (str, opcional)
    """
    cfg = _get_config(config)

    # Sugerencias por defecto si no viene mapeo
    default_map = {
        "unidad": "Alojamiento",
        "fecha_llegada": "Fecha entrada",
        "ocupacion": "Ocupación %",
        "objetivo_ocupacion": "objetivo_ocupacion",
        "adr": "ADR (€)",
        "adr_compset": "ADR compset (€)",
        "pickup_7d": "pickup_7d",
        "pace_vs_ly": "pace_vs_ly (%)",
        "zona": "Zona",
        "cluster": "Cluster",
    }
    # Permite alias 'cluster'→'zona'
    effective = {}
    for k in ["unidad", "fecha_llegada", "ocupacion", "objetivo_ocupacion", "adr", "adr_compset", "pickup_7d", "pace_vs_ly", "zona", "cluster"]:
        v = cfg.get(k)
        if v is None:
            v = default_map.get(k)
        if v in df.columns:
            effective[k] = v

    # Validar mínimos
    missing = [k for k in ["unidad", "fecha_llegada", "ocupacion", "adr"] if k not in effective]
    if missing:
        st.error(
            "Faltan columnas mínimas para el módulo de alertas: "
            + ", ".join(missing)
            + ". Configura config_cols en session_state o pasa 'config' al módulo."
        )
        raise ValueError("Column mapping missing: " + ", ".join(missing))

    out = pd.DataFrame()
    out["unidad"] = df[effective["unidad"]].astype(str).str.strip()

    # Fechas
    fechas = pd.to_datetime(df[effective["fecha_llegada"]], errors="coerce")
    out["fecha_llegada"] = fechas.dt.normalize()

    # Ocupación (normaliza a % 0–100)
    occ = pd.to_numeric(df[effective["ocupacion"]], errors="coerce")
    max_occ = np.nanmax(occ.values) if occ.notna().any() else np.nan
    if np.isfinite(max_occ) and max_occ <= 1.5:
        occ = occ * 100.0
    out["ocupacion"] = occ.astype(float)

    # ADR y compset
    out["adr"] = pd.to_numeric(df[effective["adr"]], errors="coerce").astype(float)
    if "adr_compset" in effective:
        out["adr_compset"] = pd.to_numeric(df[effective["adr_compset"]], errors="coerce").astype(float)
    else:
        out["adr_compset"] = np.nan

    # Pace vs LY (%)
    if "pace_vs_ly" in effective:
        pv = pd.to_numeric(df[effective["pace_vs_ly"]], errors="coerce")
        pv_max = np.nanmax(pv.values) if pv.notna().any() else np.nan
        if np.isfinite(pv_max) and pv_max <= 1.5:
            pv = pv * 100.0
        out["pace_vs_ly"] = pv.astype(float)
    else:
        out["pace_vs_ly"] = np.nan

    # Pickup 7d
    if "pickup_7d" in effective:
        out["pickup_7d"] = pd.to_numeric(df[effective["pickup_7d"]], errors="coerce").astype(float)
    else:
        out["pickup_7d"] = np.nan

    # Objetivo ocupación
    if "objetivo_ocupacion" in effective and effective["objetivo_ocupacion"] in df.columns:
        tgt = pd.to_numeric(df[effective["objetivo_ocupacion"]], errors="coerce")
        tgt_max = np.nanmax(tgt.values) if tgt.notna().any() else np.nan
        if np.isfinite(tgt_max) and tgt_max <= 1.5:
            tgt = tgt * 100.0
        out["objetivo_ocupacion"] = tgt.astype(float)
    else:
        out["objetivo_ocupacion"] = np.nan

    # Zona/cluster
    if "zona" in effective:
        out["zona"] = df[effective["zona"]].astype(str)
    elif "cluster" in effective:
        out["zona"] = df[effective["cluster"]].astype(str)
    else:
        out["zona"] = ""

    return out, effective


def compute_windows(df: pd.DataFrame, today: date) -> pd.DataFrame:
    """Añade days_to_arrival, window, mes_tab ('Mes actual'|'Mes siguiente'|'Otro')."""
    out = df.copy()
    base_day = pd.to_datetime(pd.Timestamp(today).normalize())
    out["days_to_arrival"] = (out["fecha_llegada"] - base_day).dt.days

    # Ventanas
    def _win(d: float) -> str:
        if pd.isna(d):
            return "fuera"
        d = int(d)
        if 0 <= d <= 7:
            return "0-7"
        if 8 <= d <= 30:
            return "8-30"
        if 31 <= d <= 60:
            return "31-60"
        return "fuera"

    out["window"] = out["days_to_arrival"].apply(_win)

    # Tabs por mes de llegada
    this_month = base_day.to_period("M")
    next_month = (base_day + pd.DateOffset(months=1)).to_period("M")
    month_period = out["fecha_llegada"].dt.to_period("M")
    out["mes_tab"] = np.where(
        month_period == this_month, "Mes actual",
        np.where(month_period == next_month, "Mes siguiente", "Otro")
    )
    return out


# ==============================
# Reglas, acciones y KPIs
# ==============================

def _pickup_threshold_p25(series: pd.Series) -> float:
    """Umbral de pickup bajo: P25; fallback media - 1*std; fallback 0."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    try:
        return float(np.percentile(s.values, 25))
    except Exception:
        m, sd = float(np.nanmean(s)), float(np.nanstd(s))
        return max(m - sd, 0.0)


def suggest_action(row: pd.Series) -> str:
    """Devuelve texto breve de acción sugerida basado en ventana y color + reglas delta."""
    win = str(row.get("window", ""))
    color = str(row.get("semaforo", ""))
    delta_adr = float(row.get("delta_adr", np.nan)) if pd.notna(row.get("delta_adr", np.nan)) else np.nan
    pickup_low = bool(row.get("pickup_low", False))

    base = ""
    if win == "0-7":
        if color == "Rojo":
            base = "Bajar ADR 5–12 %, abrir 1N/gap-filler 24–48 h, late-deal móvil."
        elif color == "Verde":
            base = "Subir ADR 5–8 %, considerar min-stay 2N en fin de semana."
        else:
            base = "Ajuste fino de precio; revisar min-stay y disponibilidad."
    elif win == "8-30":
        if color == "Rojo":
            base = "Ajuste -5–8 % ADR, early pick-up (NR blanda), revisar min-stay."
        elif color == "Verde":
            base = "+5–10 % ADR, reinstaurar min-stay fin de semana, limitar promos."
        else:
            base = "Microajustes y control de ritmo; vigilar conversión."
    elif win == "31-60":
        if color == "Rojo":
            base = "Micro-promos (-3–5 %) selectivas y revisión contenido OTA."
        elif color == "Verde":
            base = "Mantener estrategia; test A/B de precio en días valle."
        else:
            base = "Ajuste suave y optimización de contenido."

    extra = []
    if pd.notna(delta_adr) and delta_adr > 10 and pickup_low:
        extra.append("Rebajar a +2–5€ sobre compset.")
    if win == "0-7" and float(row.get("ocupacion", 0.0)) < 92:
        extra.append("Abrir 1N con suplemento/gap-filler.")

    txt = base
    if extra:
        txt += " " + " ".join(extra)
    return txt


def _target_occ_for_window(row: pd.Series) -> float:
    """Objetivo de ocupación por ventana si no hay objetivo explícito."""
    if pd.notna(row.get("objetivo_ocupacion", np.nan)) and row.get("objetivo_ocupacion", 0) > 0:
        return float(row["objetivo_ocupacion"])
    win = row.get("window", "")
    if win == "0-7":
        return 90.0
    if win == "8-30":
        return 85.0
    if win == "31-60":
        return 80.0
    return 80.0


def compute_alerts(df: pd.DataFrame, config: Optional[dict] = None, today: Optional[date] = None) -> pd.DataFrame:
    """Calcula semáforos y columnas de salida para el panel."""
    df0, used_map = normalize_columns(df, config)
    today = today or date.today()
    dfx = compute_windows(df0, today)

    # Delta ADR vs compset
    dfx["delta_adr"] = np.where(dfx["adr"].notna() & dfx["adr_compset"].notna(), dfx["adr"] - dfx["adr_compset"], np.nan)

    # Pickup bajo umbral global
    p25 = _pickup_threshold_p25(dfx["pickup_7d"])
    dfx["pickup_low"] = np.where(dfx["pickup_7d"].notna(), dfx["pickup_7d"] <= p25, False)

    # Objetivo de ocupación por fila
    dfx["target_occ"] = dfx.apply(_target_occ_for_window, axis=1)

    # Reglas de semáforo
    def _semaforo(row: pd.Series) -> str:
        win = row["window"]
        occ = float(row.get("ocupacion", np.nan))
        pace = float(row.get("pace_vs_ly", np.nan)) if pd.notna(row.get("pace_vs_ly", np.nan)) else np.nan
        delta = float(row.get("delta_adr", np.nan)) if pd.notna(row.get("delta_adr", np.nan)) else np.nan
        pk_low = bool(row.get("pickup_low", False))
        tgt = float(row.get("target_occ", 0.0))

        # 0–7
        if win == "0-7":
            if pd.notna(occ) and occ < 85 and (row.get("pickup_7d", np.nan) is np.nan or pk_low or np.isnan(row.get("pickup_7d", np.nan))):
                return "Rojo"
            if pd.notna(occ) and 85 <= occ < 92:
                return "Ámbar"
            if pd.notna(occ) and occ >= 92:
                return "Verde"
            # Fallback por objetivo
            if pd.notna(occ) and occ < max(0.0, tgt - 5):
                return "Ámbar"
            return "Ámbar"

        # 8–30
        if win == "8-30":
            if pd.notna(pace):
                if pace < 80 or (pd.notna(delta) and delta > 10 and pk_low):
                    return "Rojo"
                if (80 <= pace < 95) or (pd.notna(delta) and 5 < delta <= 10):
                    return "Ámbar"
                if pace >= 95 and (not pd.notna(delta) or abs(delta) <= 5):
                    return "Verde"
                return "Ámbar"
            # Fallback por objetivo de ocupación
            if pd.notna(occ):
                if occ < max(0.0, tgt - 10):
                    return "Rojo"
                if occ < tgt:
                    return "Ámbar"
                return "Verde"
            return "Ámbar"

        # 31–60
        if win == "31-60":
            if pd.notna(pace):
                if pace < 90:
                    return "Rojo"
                if 90 <= pace < 100:
                    return "Ámbar"
                return "Verde"
            # Fallback por objetivo
            if pd.notna(occ):
                if occ < max(0.0, tgt - 10):
                    return "Rojo"
                if occ < tgt:
                    return "Ámbar"
                return "Verde"
            return "Ámbar"

        return "—"

    dfx["semaforo"] = dfx.apply(_semaforo, axis=1)

    # Acción sugerida
    dfx["accion_sugerida"] = dfx.apply(suggest_action, axis=1)

    # Columnas de salida
    out = dfx.copy()
    out["Fecha llegada"] = out["fecha_llegada"].dt.strftime("%Y-%m-%d")
    out["Días a llegada"] = out["days_to_arrival"].astype("Int64")
    out["Ocupación %"] = out["ocupacion"].round(2)
    out["Pace vs LY %"] = out["pace_vs_ly"].round(2)
    out["ΔADR vs compset €"] = out["delta_adr"].round(2)

    return out


def kpi_counts(df_alerts: pd.DataFrame) -> Dict[str, float]:
    """Porcentaje de filas por color."""
    if df_alerts.empty or "semaforo" not in df_alerts.columns:
        return {"Rojo": 0.0, "Ámbar": 0.0, "Verde": 0.0}
    total = max(len(df_alerts), 1)
    pct = {
        k: float((df_alerts["semaforo"] == k).sum()) / total * 100.0
        for k in ["Rojo", "Ámbar", "Verde"]
    }
    return pct


def style_alerts(df_alerts: pd.DataFrame) -> pd.io.formats.style.Styler:
    """Styler con fondos suaves por color."""
    if df_alerts.empty:
        return df_alerts.style

    COLORS = {
        "Rojo": "#f8d7da",
        "Ámbar": "#fff3cd",
        "Verde": "#d4edda",
    }
    def _row_style(r: pd.Series):
        color = COLORS.get(str(r.get("semaforo", "")), "")
        return [f"background-color: {color};" if color else "" for _ in r]

    num_fmt = {
        "Ocupación %": "{:.2f}",
        "Pace vs LY %": "{:.2f}",
        "ΔADR vs compset €": "{:.2f}",
    }
    cols = [c for c in ["Unidad","Fecha llegada","Días a llegada","Ocupación %","Pace vs LY %","ΔADR vs compset €","Semáforo","Acción sugerida"] if c in df_alerts.columns]
    view = df_alerts.rename(columns={
        "unidad": "Unidad",
        "semaforo": "Semáforo",
        "accion_sugerida": "Acción sugerida",
    })[cols]
    styler = (
        view.style
        .apply(_row_style, axis=1)
        .format(num_fmt)
    )
    return styler


# ==============================
# Render principal del módulo
# ==============================

def _severity_key(row: pd.Series) -> Tuple[int, float, float]:
    """Orden: Rojo(0) > Ámbar(1) > Verde(2). Dentro: menor ocupación / peor pace primero."""
    sev = {"Rojo": 0, "Ámbar": 1, "Verde": 2}.get(str(row.get("semaforo", "")), 3)
    occ = float(row.get("ocupacion", np.nan))
    pace = float(row.get("pace_vs_ly", np.nan))
    return (sev, occ if pd.notna(occ) else 9999.0, -pace if pd.notna(pace) else 0.0)


def _filter_current_scope(df_alerts: pd.DataFrame, tab_label: str, days_bucket: str, zona_sel: Optional[str], search: str) -> pd.DataFrame:
    """Aplica filtros del UI."""
    d = df_alerts.copy()
    if tab_label in ("Mes actual", "Mes siguiente"):
        d = d[d["mes_tab"] == tab_label]
    if days_bucket in ("0-7", "8-30", "31-60"):
        d = d[d["window"] == days_bucket]
    if zona_sel and "zona" in d.columns and zona_sel != "(todas)":
        d = d[d["zona"] == zona_sel]
    if search:
        s = search.strip().lower()
        d = d[d["unidad"].str.lower().str.contains(s)]
    return d


def render_alerts_module(df: pd.DataFrame, config: dict | None = None, today: date | None = None) -> None:
    """
    Renderiza el panel de alertas automáticas del portfolio.
    Uso: render_alerts_module(st.session_state.raw_df)  (df ya cargado por la app principal)
    """
    st.header("🚨 Panel de alertas del portfolio")
    nowts = datetime.now().strftime("%Y-%m-%d %H:%M")
    st.caption(f"Cálculo: {nowts}")

    # Ayuda
    with st.expander("Cómo se calculan las alertas"):
        st.markdown(
            "- Ventanas: 0–7 (última milla), 8–30 (control de ritmo), 31–60 (previsión).\n"
            "- 0–7: Rojo si ocupación < 85% (y si hay, pickup_7d bajo). Ámbar 85–92%, Verde ≥ 92%.\n"
            "- 8–30: Rojo si pace_vs_ly < 80% o ΔADR>+10€ con pickup bajo; Ámbar 80–95% o ΔADR+5–10€; Verde pace≥95% y |ΔADR| ≤ 5€.\n"
            "- 31–60: Rojo pace<90%, Ámbar 90–100%, Verde ≥ 100%.\n"
            "- Si no hay pace/pickup/compset: degradación por ocupación vs objetivo por ventana."
        )

    # Calcular alertas
    try:
        alerts = compute_alerts(df, config=config, today=today or date.today())
    except Exception as e:
        st.stop()

    # Tabs por mes
    tabs = st.tabs(["Mes actual", "Mes siguiente"])
    for tab_label, tab in zip(["Mes actual", "Mes siguiente"], tabs):
        with tab:
            # Filtros
            zonas = ["(todas)"]
            if "zona" in alerts.columns and alerts["zona"].astype(str).str.strip().replace("nan","").any():
                zonas += sorted([z for z in alerts["zona"].dropna().astype(str).unique().tolist() if z])
            c1, c2, c3 = st.columns(3)
            with c1:
                zona_sel = st.selectbox("Zona/cluster", zonas, key=f"alerts_zona_{tab_label}")
            with c2:
                bucket = st.selectbox("Rango de días", ["0-7", "8-30", "31-60"], key=f"alerts_bucket_{tab_label}")
            with c3:
                search = st.text_input("Buscar unidad", key=f"alerts_search_{tab_label}", placeholder="Nombre o parte...")

            cur = _filter_current_scope(alerts, tab_label, bucket, zona_sel, search)

            # KPIs de color
            pct = kpi_counts(cur)
            k1, k2, k3 = st.columns(3)
            k1.metric("% Rojo", f"{pct['Rojo']:.1f}%")
            k2.metric("% Ámbar", f"{pct['Ámbar']:.1f}%")
            k3.metric("% Verde", f"{pct['Verde']:.1f}%")

            # Top 10 riesgos hoy
            st.subheader("Top 10 riesgos hoy")
            cur_sorted = cur.sort_values(by=["semaforo", "ocupacion", "pace_vs_ly"], key=lambda c: np.argsort(cur.apply(_severity_key, axis=1)) if c.name == "semaforo" else c, ascending=[True, True, False])
            # Fallback si vacío: usa todo el tab
            if cur_sorted.empty:
                cur_sorted = _filter_current_scope(alerts, tab_label, bucket, None, "")
            cur_sorted = cur_sorted.copy()
            cur_sorted["Unidad"] = cur_sorted["unidad"]
            cur_sorted["Semáforo"] = cur_sorted["semaforo"]
            cur_sorted["Acción sugerida"] = cur_sorted["accion_sugerida"]
            keep_cols = ["Unidad", "Fecha llegada", "Días a llegada", "Ocupación %", "Pace vs LY %", "ΔADR vs compset €", "Semáforo", "Acción sugerida"]
            top10 = cur_sorted.sort_values(by=["semaforo", "ocupacion", "pace_vs_ly"], key=lambda c: np.argsort(cur_sorted.apply(_severity_key, axis=1)) if c.name == "semaforo" else c, ascending=[True, True, False]).head(10)
            st.dataframe(style_alerts(top10[keep_cols]), use_container_width=True)

            # Panel completo
            with st.expander("Ver todas las alertas filtradas", expanded=False):
                st.dataframe(style_alerts(cur_sorted[keep_cols]), use_container_width=True)

                # Exportar CSV visibles
                csv_bytes = cur_sorted[keep_cols].to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Exportar alertas (CSV)", data=csv_bytes, file_name="alertas_visibles.csv", mime="text/csv", key=f"dl_csv_{tab_label}")

                # Copiar acciones sugeridas (fallback con st.code)
                joined_actions = "\n".join(f"- {u}: {a}" for u, a in zip(cur_sorted["Unidad"], cur_sorted["Acción sugerida"]))
                st.code(joined_actions, language="text")
                st.caption("Selecciona y copia manualmente las acciones (Ctrl+C).")

            # Registro de acciones (log)
            st.subheader("Registro de decisiones")
            # Identificador de fila: unidad + fecha
            cur_sorted["key_id"] = cur_sorted["Unidad"].astype(str) + " | " + cur_sorted["Fecha llegada"].astype(str)
            sel_id = st.selectbox("Selecciona alerta", cur_sorted["key_id"].tolist(), key=f"log_sel_{tab_label}")

            # Campos del log
            act_opts = [
                "Bajada ADR 5–12 %",
                "Subida ADR 5–10 %",
                "Apertura 1N/gap-filler",
                "Revisión min-stay",
                "Lanzar promo selectiva",
                "Revisión contenido OTA",
                "Otra"
            ]
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                act_applied = st.selectbox("Acción aplicada", act_opts, key=f"log_action_{tab_label}")
            with c2:
                comment = st.text_input("Comentario", key=f"log_comment_{tab_label}")
            with c3:
                follow = st.selectbox("Revisar en", ["24 h", "48 h", "72 h"], key=f"log_follow_{tab_label}")

            if st.button("Guardar en log", type="primary", key=f"btn_save_log_{tab_label}"):
                logs = st.session_state.get("alert_logs", [])
                row = cur_sorted[cur_sorted["key_id"] == sel_id].iloc[0].to_dict() if not cur_sorted.empty else {}
                logs.append({
                    "ts": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "tab": tab_label,
                    "bucket": bucket,
                    "key": sel_id,
                    "unidad": row.get("Unidad", ""),
                    "fecha_llegada": row.get("Fecha llegada", ""),
                    "accion": act_applied,
                    "comentario": comment,
                    "revisar_en": follow,
                })
                st.session_state["alert_logs"] = logs
                st.success("Guardado en el log.")

            # Exportar log
            logs = st.session_state.get("alert_logs", [])
            if logs:
                df_logs = pd.DataFrame(logs)
                st.dataframe(df_logs, use_container_width=True)
                st.download_button(
                    "📥 Exportar log (CSV)",
                    data=df_logs.to_csv(index=False).encode("utf-8-sig"),
                    file_name="alert_logs.csv",
                    mime="text/csv",
                    key=f"dl_log_{tab_label}"
                )