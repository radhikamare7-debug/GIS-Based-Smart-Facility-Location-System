import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.distance import geodesic
from io import StringIO, BytesIO
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="GIS Smart Facility Site Selection",
    page_icon="🌍",
    layout="wide"
)

st.title("GIS-Based Smart Facility Location and Site Selection System")
st.write(
    "An integrated GIS decision-support application for facility mapping, "
    "heatmap visualization, statistical analysis, buffer analysis, distance "
    "analysis and weighted site suitability assessment."
)

# =========================================================
# SAMPLE DATA
# =========================================================
DEFAULT_DATA = pd.DataFrame({
    "Name": [
        "College A", "Hospital A", "Library A", "School A", "Hospital B"
    ],
    "Type": [
        "College", "Hospital", "Library", "School", "Hospital"
    ],
    "Latitude": [19.0760, 19.0800, 19.0700, 19.0900, 19.0650],
    "Longitude": [72.8777, 72.8850, 72.8700, 72.8900, 72.8600]
})

REQUIRED_COLUMNS = {"Name", "Type", "Latitude", "Longitude"}


# =========================================================
# VALIDATION AND HELPER FUNCTIONS
# =========================================================
def validate_facilities(data):
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(
            "CSV must contain: Name, Type, Latitude, Longitude. "
            f"Missing columns: {', '.join(sorted(missing))}"
        )

    df = data[["Name", "Type", "Latitude", "Longitude"]].copy()
    df["Name"] = df["Name"].astype("string").str.strip()
    df["Type"] = df["Type"].astype("string").str.strip()
    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Name", "Type", "Latitude", "Longitude"])
    df = df[(df["Name"] != "") & (df["Type"] != "")].copy()

    if df.empty:
        raise ValueError("CSV does not contain valid facility records.")
    if not df["Latitude"].between(-90, 90).all():
        raise ValueError("Latitude must be between -90 and 90.")
    if not df["Longitude"].between(-180, 180).all():
        raise ValueError("Longitude must be between -180 and 180.")
    if df["Name"].duplicated().any():
        raise ValueError("Facility Name values must be unique.")
    return df.reset_index(drop=True)


def distance_km(lat1, lon1, lat2, lon2):
    return geodesic((float(lat1), float(lon1)),
                    (float(lat2), float(lon2))).km


def add_facility_markers(map_obj, data):
    palette = {
        "Hospital": "red", "School": "blue", "College": "purple",
        "Library": "green", "Bank": "orange"
    }
    for _, row in data.iterrows():
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            tooltip=str(row["Name"]),
            popup=f'{row["Name"]} | {row["Type"]}',
            icon=folium.Icon(
                color=palette.get(str(row["Type"]), "cadetblue"),
                icon="info-sign"
            )
        ).add_to(map_obj)
    return map_obj


def minmax_score(series, higher_is_better=True):
    values = pd.to_numeric(series, errors="coerce").fillna(0)
    low, high = values.min(), values.max()
    if high == low:
        return pd.Series([100.0] * len(values), index=values.index)
    scaled = (values - low) / (high - low) * 100
    if not higher_is_better:
        scaled = 100 - scaled
    return scaled


def calculate_suitability(candidates, facilities, proposed_type,
                           radius_km, weights):
    results = candidates.copy()
    access_counts = []
    same_type_gaps = []
    average_distances = []
    nearest_names = []
    nearest_distances = []

    for _, candidate in results.iterrows():
        distances = facilities.apply(
            lambda f: distance_km(
                candidate["Latitude"], candidate["Longitude"],
                f["Latitude"], f["Longitude"]
            ),
            axis=1
        )
        within_radius = facilities.loc[distances <= radius_km].copy()
        within_distances = distances[distances <= radius_km]

        # Nearby facilities of other categories indicate access to services.
        access_count = int((within_radius["Type"] != proposed_type).sum())
        access_counts.append(access_count)

        # Prefer a larger gap from an existing facility of the proposed type.
        same_type_distances = distances[
            facilities["Type"].astype(str).str.casefold()
            == str(proposed_type).casefold()
        ]
        same_type_gaps.append(
            float(same_type_distances.min())
            if not same_type_distances.empty else float(radius_km * 2)
        )

        average_distances.append(float(distances.mean()))
        nearest_idx = distances.idxmin()
        nearest_names.append(str(facilities.loc[nearest_idx, "Name"]))
        nearest_distances.append(float(distances.loc[nearest_idx]))

    results["Nearby_Other_Facilities"] = access_counts
    results["Nearest_Same_Type_KM"] = same_type_gaps
    results["Average_Distance_KM"] = average_distances
    results["Nearest_Facility"] = nearest_names
    results["Nearest_Facility_KM"] = nearest_distances

    results["Service_Access_Score"] = minmax_score(
        results["Nearby_Other_Facilities"], True
    )
    results["Facility_Gap_Score"] = minmax_score(
        results["Nearest_Same_Type_KM"], True
    )
    results["Distance_Balance_Score"] = minmax_score(
        results["Average_Distance_KM"], False
    )

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("At least one suitability weight must be greater than zero.")

    results["Suitability_Score"] = (
        results["Service_Access_Score"] * weights["service"] +
        results["Facility_Gap_Score"] * weights["gap"] +
        results["Distance_Balance_Score"] * weights["balance"]
    ) / total_weight

    results = results.sort_values(
        "Suitability_Score", ascending=False
    ).reset_index(drop=True)
    results.insert(0, "Rank", range(1, len(results) + 1))
    return results


def make_pdf_report(facilities, results, proposed_type, radius_km):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=35, leftMargin=35, topMargin=35, bottomMargin=35
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("GIS-Based Smart Facility Location and Site Selection System",
                  styles["Title"]),
        Spacer(1, 10),
        Paragraph(f"Generated: {datetime.now().strftime('%d-%m-%Y %H:%M')}",
                  styles["Normal"]),
        Paragraph(f"Proposed facility type: {proposed_type}", styles["Normal"]),
        Paragraph(f"Service analysis radius: {radius_km:.1f} km", styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Existing Facility Summary", styles["Heading2"])
    ]

    type_counts = facilities["Type"].value_counts().reset_index()
    type_counts.columns = ["Facility Type", "Count"]
    summary_data = [["Facility Type", "Count"]] + type_counts.values.tolist()
    summary_table = Table(summary_data, hAlign="LEFT")
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5)
    ]))
    story += [summary_table, Spacer(1, 14),
              Paragraph("Candidate Site Suitability Ranking", styles["Heading2"])]

    cols = ["Rank", "Candidate", "Score", "Nearby Services",
            "Nearest Same Type (km)"]
    table_data = [cols]
    for _, row in results.iterrows():
        table_data.append([
            int(row["Rank"]), str(row["Name"]),
            f'{row["Suitability_Score"]:.1f}',
            int(row["Nearby_Other_Facilities"]),
            f'{row["Nearest_Same_Type_KM"]:.2f}'
        ])
    result_table = Table(table_data, repeatRows=1, hAlign="LEFT")
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#24476B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("ALIGN", (0, 0), (-1, -1), "CENTER")
    ]))
    story += [result_table, Spacer(1, 12)]
    best = results.iloc[0]
    story.append(Paragraph(
        f"Recommended candidate: <b>{best['Name']}</b> "
        f"with suitability score {best['Suitability_Score']:.1f}/100.",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Method note: Scores are relative to the candidate sites entered and "
        "the selected weights. This is a preliminary GIS decision-support model; "
        "it does not replace field surveys, land-use rules, ownership checks, "
        "road-network travel times, environmental review or population-demand data.",
        styles["Italic"]
    ))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# SIDEBAR: DATA UPLOAD
# =========================================================
st.sidebar.title("GIS Control Panel")
uploaded_file = st.sidebar.file_uploader(
    "Upload Facility CSV File",
    type=["csv"],
    help="Required columns: Name, Type, Latitude, Longitude"
)

if uploaded_file is not None:
    try:
        facilities = validate_facilities(pd.read_csv(uploaded_file))
        st.sidebar.success("CSV uploaded and validated successfully.")
    except Exception as exc:
        st.sidebar.error(f"CSV Error: {exc}")
        st.stop()
else:
    facilities = DEFAULT_DATA.copy()
    st.sidebar.info("Using sample demonstration data.")

st.sidebar.metric("Total Facilities", len(facilities))
st.sidebar.caption("Required CSV columns: Name, Type, Latitude, Longitude")

# =========================================================
# NAVIGATION
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard",
    "Heatmap & Statistics",
    "Buffer Analysis",
    "Distance Analysis",
    "Site Selection"
])

# =========================================================
# TAB 1: DASHBOARD
# =========================================================
with tab1:
    st.header("GIS Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Facilities", len(facilities))
    c2.metric("Facility Categories", facilities["Type"].nunique())
    c3.metric("Analysis Modules", 4)
    c4.metric("Average Latitude", f'{facilities["Latitude"].mean():.4f}')

    st.subheader("Facility Information")
    st.dataframe(facilities, use_container_width=True, hide_index=True)

    st.subheader("Interactive GIS Map")
    center = [facilities["Latitude"].mean(), facilities["Longitude"].mean()]
    dashboard_map = folium.Map(location=center, zoom_start=13, control_scale=True)
    add_facility_markers(dashboard_map, facilities)
    st_folium(dashboard_map, width=None, height=480, key="dashboard_map")

    st.download_button(
        "Download Facility Data (CSV)",
        facilities.to_csv(index=False).encode("utf-8"),
        "gis_facilities_cleaned.csv",
        "text/csv"
    )

# =========================================================
# TAB 2: HEATMAP AND STATISTICS
# =========================================================
with tab2:
    st.header("Facility Heatmap & Statistical Analysis")
    st.write("Explore facility concentration and category-wise distribution.")

    categories = sorted(facilities["Type"].unique().tolist())
    selected_categories = st.multiselect(
        "Filter facility categories",
        categories,
        default=categories
    )
    filtered = facilities[facilities["Type"].isin(selected_categories)].copy()

    if filtered.empty:
        st.warning("Select at least one facility category to display the analysis.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Facilities in Filter", len(filtered))
        m2.metric("Selected Categories", filtered["Type"].nunique())
        m3.metric("Average Latitude", f'{filtered["Latitude"].mean():.4f}')

        st.subheader("Geographic Facility Density Heatmap")
        heat_center = [filtered["Latitude"].mean(), filtered["Longitude"].mean()]
        heat_map = folium.Map(location=heat_center, zoom_start=13, control_scale=True)
        HeatMap(
            filtered[["Latitude", "Longitude"]].values.tolist(),
            radius=28, blur=20, min_opacity=0.35
        ).add_to(heat_map)
        add_facility_markers(heat_map, filtered)
        st_folium(heat_map, width=None, height=480, key="facility_heatmap")
        st.caption(
            "Heatmap intensity represents the number and proximity of points "
            "in the uploaded dataset, not population density or demand."
        )

        st.subheader("Statistical Charts")
        counts = filtered["Type"].value_counts().rename_axis(
            "Facility Type"
        ).reset_index(name="Number of Facilities")
        chart1, chart2 = st.columns(2)
        with chart1:
            st.plotly_chart(
                px.bar(counts, x="Facility Type", y="Number of Facilities",
                       color="Facility Type", text="Number of Facilities",
                       title="Facilities by Category"),
                use_container_width=True
            )
        with chart2:
            st.plotly_chart(
                px.pie(counts, names="Facility Type",
                       values="Number of Facilities", hole=0.4,
                       title="Facility Category Distribution"),
                use_container_width=True
            )

        scatter = px.scatter(
            filtered, x="Longitude", y="Latitude", color="Type",
            hover_name="Name", title="Geographic Distribution of Facilities"
        )
        st.plotly_chart(scatter, use_container_width=True)

        st.subheader("Facility Statistics")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Category Statistics (CSV)",
            counts.to_csv(index=False).encode("utf-8"),
            "facility_category_statistics.csv",
            "text/csv"
        )

# =========================================================
# TAB 3: BUFFER ANALYSIS
# =========================================================
with tab3:
    st.header("Buffer Analysis")
    st.write("Find facilities located within a selected distance from a reference facility.")

    ref_name = st.selectbox(
        "Select Reference Facility", facilities["Name"].tolist(),
        key="buffer_reference"
    )
    radius = st.slider("Select Buffer Radius (KM)", 1, 20, 5, key="buffer_radius")
    reference = facilities[facilities["Name"] == ref_name].iloc[0]
    ref_point = (reference["Latitude"], reference["Longitude"])

    buffer_results = facilities.copy()
    buffer_results["Distance_KM"] = buffer_results.apply(
        lambda row: distance_km(
            ref_point[0], ref_point[1], row["Latitude"], row["Longitude"]
        ), axis=1
    )
    nearby = buffer_results[buffer_results["Distance_KM"] <= radius].copy()
    st.success(f"{len(nearby)} facilities found within {radius} KM.")
    st.dataframe(nearby.sort_values("Distance_KM"),
                 use_container_width=True, hide_index=True)

    buffer_map = folium.Map(location=ref_point, zoom_start=13, control_scale=True)
    folium.Circle(
        ref_point, radius=radius * 1000, color="blue",
        fill=True, fill_opacity=0.15, tooltip=f"{radius} KM Buffer"
    ).add_to(buffer_map)
    folium.Marker(
        ref_point, popup=f"Reference: {ref_name}",
        icon=folium.Icon(color="red", icon="star")
    ).add_to(buffer_map)
    for _, row in nearby.iterrows():
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=6, tooltip=row["Name"],
            popup=f'{row["Name"]}: {row["Distance_KM"]:.2f} KM',
            color="green", fill=True, fill_opacity=0.8
        ).add_to(buffer_map)
    st_folium(buffer_map, width=None, height=450, key="buffer_map")

    st.download_button(
        "Download Buffer Results (CSV)",
        nearby.to_csv(index=False).encode("utf-8"),
        "buffer_analysis_results.csv",
        "text/csv"
    )

# =========================================================
# TAB 4: DISTANCE ANALYSIS
# =========================================================
with tab4:
    st.header("Distance Analysis")
    st.write("Calculate geographical distance between two facilities.")

    names = facilities["Name"].tolist()
    dcol1, dcol2 = st.columns(2)
    with dcol1:
        facility_a_name = st.selectbox("Select First Facility", names, index=0)
    with dcol2:
        second_index = 1 if len(names) > 1 else 0
        facility_b_name = st.selectbox(
            "Select Second Facility", names, index=second_index
        )

    facility_a = facilities[facilities["Name"] == facility_a_name].iloc[0]
    facility_b = facilities[facilities["Name"] == facility_b_name].iloc[0]
    distance = distance_km(
        facility_a["Latitude"], facility_a["Longitude"],
        facility_b["Latitude"], facility_b["Longitude"]
    )
    st.metric("Distance Between Facilities", f"{distance:.3f} KM")

    distance_map = folium.Map(
        location=[
            (facility_a["Latitude"] + facility_b["Latitude"]) / 2,
            (facility_a["Longitude"] + facility_b["Longitude"]) / 2
        ],
        zoom_start=13, control_scale=True
    )
    folium.Marker(
        [facility_a["Latitude"], facility_a["Longitude"]],
        popup=facility_a_name, icon=folium.Icon(color="blue")
    ).add_to(distance_map)
    folium.Marker(
        [facility_b["Latitude"], facility_b["Longitude"]],
        popup=facility_b_name, icon=folium.Icon(color="red")
    ).add_to(distance_map)
    folium.PolyLine(
        [
            [facility_a["Latitude"], facility_a["Longitude"]],
            [facility_b["Latitude"], facility_b["Longitude"]]
        ],
        color="purple", weight=4,
        tooltip=f"Straight-line distance: {distance:.3f} KM"
    ).add_to(distance_map)
    st_folium(distance_map, width=None, height=450, key="distance_map")
    st.caption("Geodesic straight-line distance, not road-network travel distance.")

    nearest_rows = []
    for _, source in facilities.iterrows():
        others = facilities[facilities["Name"] != source["Name"]]
        if not others.empty:
            ds = others.apply(
                lambda r: distance_km(
                    source["Latitude"], source["Longitude"],
                    r["Latitude"], r["Longitude"]
                ), axis=1
            )
            idx = ds.idxmin()
            nearest_rows.append({
                "Facility": source["Name"],
                "Nearest Facility": facilities.loc[idx, "Name"],
                "Distance (KM)": round(float(ds.loc[idx]), 3)
            })
    if nearest_rows:
        st.subheader("Nearest Facility Finder")
        nearest_df = pd.DataFrame(nearest_rows)
        st.dataframe(nearest_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download Nearest Facility Report (CSV)",
            nearest_df.to_csv(index=False).encode("utf-8"),
            "nearest_facility_report.csv",
            "text/csv"
        )

# =========================================================
# TAB 5: ADVANCED SITE SELECTION
# =========================================================
with tab5:
    st.header("Smart Facility Site Selection")
    st.write(
        "Compare candidate sites using nearby service availability, "
        "distance from existing facilities of the proposed type, and "
        "overall distance balance."
    )

    st.subheader("Step 1: Candidate Site Input")
    st.caption("Enter one candidate per line in the format: Name,Latitude,Longitude")
    candidate_text = st.text_area(
        "Candidate Sites",
        value=(
            "Site A,19.0750,72.8750\n"
            "Site B,19.0850,72.8950\n"
            "Site C,19.0650,72.8650"
        ),
        height=125,
        help="Coordinates must be decimal degrees."
    )

    st.subheader("Step 2: Suitability Settings")
    types_available = sorted(facilities["Type"].astype(str).unique().tolist())
    proposed_type = st.selectbox(
        "Proposed New Facility Type",
        types_available + ["Other / New Category"]
    )
    if proposed_type == "Other / New Category":
        proposed_type = st.text_input("Enter new facility category", "Clinic").strip()
        if not proposed_type:
            proposed_type = "New Facility"

    coverage_radius = st.slider(
        "Nearby Service Coverage Radius (KM)", 1, 20, 5
    )

    st.markdown("**Set importance of each criterion (weights are normalized automatically).**")
    w1, w2, w3 = st.columns(3)
    service_weight = w1.slider("Nearby Services Weight", 0, 100, 40)
    gap_weight = w2.slider("Existing Same-Type Gap Weight", 0, 100, 35)
    balance_weight = w3.slider("Distance Balance Weight", 0, 100, 25)

    st.caption(
        "Nearby Services rewards access to other facility categories. "
        "Same-Type Gap rewards locations farther from existing facilities "
        "of the proposed type. Distance Balance rewards a more central location."
    )

    if st.button("Analyze Best Site", type="primary"):
        try:
            candidates = pd.read_csv(
                StringIO("Name,Latitude,Longitude\n" + candidate_text),
                skipinitialspace=True
            )
            required_candidate_cols = {"Name", "Latitude", "Longitude"}
            if not required_candidate_cols.issubset(candidates.columns):
                raise ValueError("Candidate data needs Name, Latitude, Longitude.")
            candidates["Name"] = candidates["Name"].astype("string").str.strip()
            candidates["Latitude"] = pd.to_numeric(candidates["Latitude"], errors="coerce")
            candidates["Longitude"] = pd.to_numeric(candidates["Longitude"], errors="coerce")
            candidates = candidates.dropna(
                subset=["Name", "Latitude", "Longitude"]
            )
            candidates = candidates[candidates["Name"] != ""].copy()

            if candidates.empty:
                raise ValueError("Enter at least one valid candidate site.")
            if candidates["Name"].duplicated().any():
                raise ValueError("Candidate site names must be unique.")
            if not candidates["Latitude"].between(-90, 90).all():
                raise ValueError("Candidate latitude must be between -90 and 90.")
            if not candidates["Longitude"].between(-180, 180).all():
                raise ValueError("Candidate longitude must be between -180 and 180.")

            weights = {
                "service": service_weight,
                "gap": gap_weight,
                "balance": balance_weight
            }
            results = calculate_suitability(
                candidates, facilities, proposed_type,
                coverage_radius, weights
            )
            st.session_state["site_results"] = results
            st.session_state["site_proposed_type"] = proposed_type
            st.session_state["site_radius"] = coverage_radius

        except Exception as exc:
            st.error(f"Please check candidate input: {exc}")

    if "site_results" in st.session_state:
        results = st.session_state["site_results"]
        best = results.iloc[0]

        st.divider()
        st.subheader("Site Selection Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Recommended Site", str(best["Name"]))
        r2.metric("Suitability Score", f'{best["Suitability_Score"]:.1f}/100')
        r3.metric("Candidate Sites Evaluated", len(results))

        st.success(
            f"Recommended Location: {best['Name']} "
            f"(Suitability Score: {best['Suitability_Score']:.1f}/100)"
        )

        display_cols = [
            "Rank", "Name", "Latitude", "Longitude",
            "Suitability_Score", "Service_Access_Score",
            "Facility_Gap_Score", "Distance_Balance_Score",
            "Nearby_Other_Facilities", "Nearest_Same_Type_KM",
            "Nearest_Facility", "Nearest_Facility_KM"
        ]
        display_results = results[display_cols].copy()
        st.dataframe(
            display_results.style.format({
                "Latitude": "{:.5f}", "Longitude": "{:.5f}",
                "Suitability_Score": "{:.1f}",
                "Service_Access_Score": "{:.1f}",
                "Facility_Gap_Score": "{:.1f}",
                "Distance_Balance_Score": "{:.1f}",
                "Nearest_Same_Type_KM": "{:.2f}",
                "Nearest_Facility_KM": "{:.2f}"
            }),
            use_container_width=True, hide_index=True
        )

        st.subheader("Candidate Suitability Comparison")
        score_chart = px.bar(
            results, x="Name", y="Suitability_Score",
            color="Suitability_Score", text_auto=".1f",
            title="Weighted Site Suitability Score",
            labels={"Name": "Candidate Site", "Suitability_Score": "Score / 100"}
        )
        score_chart.update_yaxes(range=[0, 100])
        st.plotly_chart(score_chart, use_container_width=True)

        st.subheader("Site Suitability Map")
        site_map = folium.Map(
            location=[best["Latitude"], best["Longitude"]],
            zoom_start=13, control_scale=True
        )
        add_facility_markers(site_map, facilities)
        for _, row in results.iterrows():
            is_best = int(row["Rank"]) == 1
            folium.Marker(
                [row["Latitude"], row["Longitude"]],
                tooltip=f'Rank {row["Rank"]}: {row["Name"]}',
                popup=(
                    f'{row["Name"]}<br>Rank: {row["Rank"]}'
                    f'<br>Suitability: {row["Suitability_Score"]:.1f}/100'
                ),
                icon=folium.Icon(
                    color="green" if is_best else "orange",
                    icon="star" if is_best else "map-marker"
                )
            ).add_to(site_map)
        st_folium(site_map, width=None, height=480, key="site_suitability_map")

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Site Selection Results (CSV)",
            csv_bytes, "advanced_site_selection_results.csv", "text/csv"
        )

        pdf_bytes = make_pdf_report(
            facilities, results,
            st.session_state["site_proposed_type"],
            st.session_state["site_radius"]
        )
        st.download_button(
            "Download Site Analysis Report (PDF)",
            pdf_bytes, "gis_site_analysis_report.pdf", "application/pdf"
        )

        st.info(
            "Interpretation: the highest score is the best candidate only "
            "among the candidate locations entered and under the selected "
            "weights. This preliminary model does not include land price, "
            "zoning, road travel time, population demand, environmental "
            "constraints or field verification."
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()
st.caption(
    "GIS-Based Smart Facility Location and Site Selection System | "
    "Developed using Python, Streamlit, Pandas, Folium, GeoPy, Plotly and ReportLab"
)
