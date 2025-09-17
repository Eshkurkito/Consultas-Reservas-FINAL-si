# ============================================================================
# app12_copy8_PRO_FULL_FINAL_MARKER_ALT2_RESUMEN.py
# Archivo original con integración del modo "Resumen Comparativo"
# ============================================================================

# ... [todo tu código original sin cambios previos] ...

# ---------------- Menú de modos ----------------
mode = st.sidebar.radio(
    "Modo de consulta",
    [
        "Consulta normal",
        "Resumen Comparativo",   # <-- NUEVO MODO
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
        "Calendario de tarifas",
    ],
    key="mode_radio"
)

# ============================================================================
# NUEVO MODO: Resumen Comparativo
# ============================================================================
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
    )[["Alojamiento","Ingresos"]].rename(columns={"Ingresos":"Ingresos finales LY (€)"})

    resumen = now_df.merge(ly_df, on="Alojamiento", how="outer").merge(ly_final_df, on="Alojamiento", how="left")

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

    import io
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        resumen.to_excel(writer, index=False, sheet_name="Resumen")
        ws = writer.sheets["Resumen"]
        for j, col in enumerate(resumen.columns):
            width = int(min(38, max(12, resumen[col].astype(str).str.len().max() if not resumen.empty else 12)))
            ws.set_column(j, j, width)
    st.download_button(
        "📥 Descargar Excel (.xlsx)",
        data=buffer.getvalue(),
        file_name="resumen_comparativo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ... [resto de modos ya existentes] ...
