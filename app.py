import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.optimize import minimize


# ======================================================
# Page setup
# ======================================================

st.set_page_config(
    page_title="Drillhole Trajectory Correction Tool",
    layout="wide"
)

st.title("Drillhole Trajectory Correction Tool")

st.write(
    """
    Upload an offset Excel file, enter target depth / azimuth / dip, 
    and calculate the corrected design azimuth and dip.
    """
)


# ======================================================
# Column handling
# ======================================================

def normalise_column_name(name):
    name = str(name).strip()
    name = name.replace("\n", "")
    name = name.replace("\r", "")
    name = name.replace("\t", "")
    name = re.sub(r"[^A-Za-z0-9]", "", name)
    return name.lower()


def standardise_offset_columns(df):
    """
    Convert uploaded Excel columns to:
    Depth, Dip_Offset, Azi_Offset

    Supported formats:
    Depth / Dip_Offset / Azi_Offset
    Depth / AVG Change Dip / AVG Change Azi
    Depth / Change Dip / Change Azi
    """

    original_columns = list(df.columns)

    normalised_lookup = {}
    for col in original_columns:
        normalised_lookup[normalise_column_name(col)] = col

    column_map = {}

    # Depth
    depth_keys = [
        "depth",
        "depthm",
        "md",
        "measureddepth",
        "holedepth"
    ]

    for key in depth_keys:
        if key in normalised_lookup:
            column_map[normalised_lookup[key]] = "Depth"
            break

    # Dip
    dip_keys = [
        "dipoffset",
        "dipoffset5m",
        "avgchangedip",
        "averagechangedip",
        "avgdipchange",
        "averagedipchange",
        "changedip",
        "dipchange",
        "dipdeviation",
        "dipoff"
    ]

    for key in dip_keys:
        if key in normalised_lookup:
            column_map[normalised_lookup[key]] = "Dip_Offset"
            break

    # Azi
    azi_keys = [
        "azioffset",
        "azioffset5m",
        "avgchangeazi",
        "averagechangeazi",
        "avgazichange",
        "averageazichange",
        "changeazi",
        "azichange",
        "azimuthoffset",
        "azideviation",
        "azioff"
    ]

    for key in azi_keys:
        if key in normalised_lookup:
            column_map[normalised_lookup[key]] = "Azi_Offset"
            break

    df = df.rename(columns=column_map)

    required_columns = ["Depth", "Dip_Offset", "Azi_Offset"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n\n"
            f"Detected columns: {original_columns}\n\n"
            "The Excel file should contain one of these formats:\n"
            "1) Depth, Dip_Offset, Azi_Offset\n"
            "2) Depth, AVG Change Dip, AVG Change Azi\n"
            "3) Depth, Change Dip, Change Azi"
        )

    df_offset = df[required_columns].copy()

    for col in required_columns:
        df_offset[col] = pd.to_numeric(df_offset[col], errors="coerce")

    df_offset = df_offset.dropna()
    df_offset = df_offset.sort_values("Depth").reset_index(drop=True)

    if df_offset.empty:
        raise ValueError("No valid numeric offset data found.")

    if float(df_offset.loc[0, "Depth"]) != 0:
        zero_row = pd.DataFrame({
            "Depth": [0.0],
            "Dip_Offset": [0.0],
            "Azi_Offset": [0.0]
        })
        df_offset = pd.concat([zero_row, df_offset], ignore_index=True)
        df_offset = df_offset.sort_values("Depth").reset_index(drop=True)

    return df_offset, original_columns, column_map


# ======================================================
# Geometry calculation
# ======================================================

def unit_vector_from_azi_dip(azi_deg, dip_deg):
    """
    Azi is clockwise from north.
    Dip is negative downward.
    """
    azi = np.radians(azi_deg)
    dip = np.radians(dip_deg)

    east = np.cos(dip) * np.sin(azi)
    north = np.cos(dip) * np.cos(azi)
    vertical = np.sin(dip)

    return np.array([east, north, vertical])


def straight_line_point(depth, azi_deg, dip_deg):
    return depth * unit_vector_from_azi_dip(azi_deg, dip_deg)


def curved_trajectory_point(df_offset, design_azi, design_dip, depth):
    position = np.array([0.0, 0.0, 0.0])

    cumulative_dip_offset = 0.0
    cumulative_azi_offset = 0.0
    previous_depth = 0.0

    for i in range(1, len(df_offset)):
        current_depth = float(df_offset.loc[i, "Depth"])

        if previous_depth >= depth:
            break

        segment_length = min(current_depth, depth) - previous_depth

        if segment_length <= 0:
            break

        cumulative_dip_offset += float(df_offset.loc[i, "Dip_Offset"])
        cumulative_azi_offset += float(df_offset.loc[i, "Azi_Offset"])

        segment_dip = design_dip + cumulative_dip_offset
        segment_azi = design_azi + cumulative_azi_offset

        position += segment_length * unit_vector_from_azi_dip(segment_azi, segment_dip)

        previous_depth += segment_length

    return position


def straight_trajectory_table(target_azi, target_dip, max_depth, step=5):
    depths = list(np.arange(0, max_depth, step))

    if len(depths) == 0 or depths[-1] != max_depth:
        depths.append(max_depth)

    rows = []

    for depth in depths:
        pos = straight_line_point(depth, target_azi, target_dip)
        horizontal = np.sqrt(pos[0] ** 2 + pos[1] ** 2)

        rows.append({
            "Depth": depth,
            "E": pos[0],
            "N": pos[1],
            "Z": pos[2],
            "Horizontal_Displacement": horizontal,
            "Vertical_Depth": -pos[2]
        })

    return pd.DataFrame(rows)


def curved_trajectory_table(df_offset, design_azi, design_dip, max_depth):
    rows = []

    position = np.array([0.0, 0.0, 0.0])
    cumulative_dip_offset = 0.0
    cumulative_azi_offset = 0.0
    previous_depth = 0.0

    rows.append({
        "Depth": 0.0,
        "Segment_Length": 0.0,
        "Cum_Dip_Offset": 0.0,
        "Cum_Azi_Offset": 0.0,
        "Segment_Dip": design_dip,
        "Segment_Azi": design_azi % 360,
        "E": 0.0,
        "N": 0.0,
        "Z": 0.0,
        "Horizontal_Displacement": 0.0,
        "Vertical_Depth": 0.0
    })

    for i in range(1, len(df_offset)):
        current_depth = float(df_offset.loc[i, "Depth"])

        if previous_depth >= max_depth:
            break

        segment_length = min(current_depth, max_depth) - previous_depth

        if segment_length <= 0:
            break

        cumulative_dip_offset += float(df_offset.loc[i, "Dip_Offset"])
        cumulative_azi_offset += float(df_offset.loc[i, "Azi_Offset"])

        segment_dip = design_dip + cumulative_dip_offset
        segment_azi = design_azi + cumulative_azi_offset

        position += segment_length * unit_vector_from_azi_dip(segment_azi, segment_dip)

        horizontal = np.sqrt(position[0] ** 2 + position[1] ** 2)
        vertical_depth = -position[2]

        rows.append({
            "Depth": previous_depth + segment_length,
            "Segment_Length": segment_length,
            "Cum_Dip_Offset": cumulative_dip_offset,
            "Cum_Azi_Offset": cumulative_azi_offset,
            "Segment_Dip": segment_dip,
            "Segment_Azi": segment_azi % 360,
            "E": position[0],
            "N": position[1],
            "Z": position[2],
            "Horizontal_Displacement": horizontal,
            "Vertical_Depth": vertical_depth
        })

        previous_depth += segment_length

    return pd.DataFrame(rows)


def run_correction(df_offset, target_depth, target_azi, target_dip):
    target_point = straight_line_point(target_depth, target_azi, target_dip)

    def objective(x):
        design_azi = x[0]
        design_dip = x[1]

        actual_point = curved_trajectory_point(
            df_offset=df_offset,
            design_azi=design_azi,
            design_dip=design_dip,
            depth=target_depth
        )

        error_vector = actual_point - target_point
        return np.sum(error_vector ** 2)

    result = minimize(
        objective,
        np.array([target_azi, target_dip]),
        method="Nelder-Mead",
        options={
            "xatol": 1e-10,
            "fatol": 1e-10,
            "maxiter": 20000
        }
    )

    design_azi = result.x[0] % 360
    design_dip = result.x[1]

    actual_point = curved_trajectory_point(df_offset, design_azi, design_dip, target_depth)
    error_vector = actual_point - target_point
    error_distance = np.linalg.norm(error_vector)

    straight_df = straight_trajectory_table(target_azi, target_dip, target_depth)
    curved_df = curved_trajectory_table(df_offset, design_azi, design_dip, target_depth)

    summary = {
        "Target Depth": target_depth,
        "Target Azi": target_azi,
        "Target Dip": target_dip,
        "Design Azi": design_azi,
        "Design Dip": design_dip,
        "Target E": target_point[0],
        "Target N": target_point[1],
        "Target Z": target_point[2],
        "Actual E": actual_point[0],
        "Actual N": actual_point[1],
        "Actual Z": actual_point[2],
        "dE": error_vector[0],
        "dN": error_vector[1],
        "dZ": error_vector[2],
        "3D Error Distance": error_distance,
        "Optimisation Status": result.message
    }

    return summary, straight_df, curved_df


# ======================================================
# Plot and export
# ======================================================

def create_trajectory_plot(straight_df, curved_df, summary):
    fig, ax = plt.subplots(figsize=(8, 7))

    target_azi = summary["Target Azi"]
    target_dip = summary["Target Dip"]
    design_azi = summary["Design Azi"]
    design_dip = summary["Design Dip"]
    error_distance = summary["3D Error Distance"]

    ax.plot(
        straight_df["Horizontal_Displacement"],
        straight_df["Vertical_Depth"],
        marker="o",
        linewidth=2,
        label=f"Target straight hole\nAzi={target_azi:.2f}°, Dip={target_dip:.2f}°"
    )

    ax.plot(
        curved_df["Horizontal_Displacement"],
        curved_df["Vertical_Depth"],
        marker="s",
        linewidth=2,
        label=f"Corrected curved hole\nDesign Azi={design_azi:.2f}°, Design Dip={design_dip:.2f}°"
    )

    ax.scatter(
        straight_df["Horizontal_Displacement"].iloc[-1],
        straight_df["Vertical_Depth"].iloc[-1],
        s=100,
        label="Target point"
    )

    ax.scatter(
        curved_df["Horizontal_Displacement"].iloc[-1],
        curved_df["Vertical_Depth"].iloc[-1],
        s=100,
        label="Corrected endpoint"
    )

    ax.annotate(
        f"Target\nAzi={target_azi:.2f}°\nDip={target_dip:.2f}°",
        xy=(
            straight_df["Horizontal_Displacement"].iloc[-1],
            straight_df["Vertical_Depth"].iloc[-1]
        ),
        xytext=(10, -40),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

    ax.annotate(
        f"Corrected\nAzi={design_azi:.2f}°\nDip={design_dip:.2f}°\nError={error_distance:.3f} m",
        xy=(
            curved_df["Horizontal_Displacement"].iloc[-1],
            curved_df["Vertical_Depth"].iloc[-1]
        ),
        xytext=(10, 25),
        textcoords="offset points",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", linewidth=1)
    )

    ax.set_xlabel("Horizontal displacement from collar (m)")
    ax.set_ylabel("Vertical depth (m)")
    ax.set_title("Drillhole Trajectory: Depth vs Position Change")
    ax.invert_yaxis()
    ax.grid(True)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def make_excel_file(summary, straight_df, curved_df):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        straight_df.to_excel(writer, sheet_name="Target_Straight_Hole", index=False)
        curved_df.to_excel(writer, sheet_name="Corrected_Curved_Hole", index=False)

    output.seek(0)
    return output


def make_png_file(fig):
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=300, bbox_inches="tight")
    output.seek(0)
    return output


# ======================================================
# UI
# ======================================================

with st.sidebar:
    st.header("Input")

    uploaded_file = st.file_uploader(
        "Upload offset Excel",
        type=["xlsx", "xls"]
    )

    target_depth = st.number_input(
        "Target Depth",
        min_value=0.0,
        value=152.0,
        step=1.0
    )

    target_azi = st.number_input(
        "Target Azi",
        min_value=0.0,
        max_value=360.0,
        value=270.0,
        step=1.0
    )

    target_dip = st.number_input(
        "Target Dip",
        min_value=-90.0,
        max_value=90.0,
        value=-60.0,
        step=1.0
    )

    run_button = st.button("Run Calculation", type="primary")


st.subheader("Excel Format")

st.write("The uploaded Excel file can use any of these column formats:")

st.dataframe(
    pd.DataFrame({
        "Depth": [0, 5, 10, 15],
        "Dip_Offset / AVG Change Dip": [0, -0.08, 0.00, 0.09],
        "Azi_Offset / AVG Change Azi": [0, 0.08, 0.71, 0.54]
    }),
    use_container_width=True
)

st.info(
    "The app treats offset values as incremental offset for each depth interval."
)


if uploaded_file is None:
    st.warning("Please upload an offset Excel file.")

else:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_offset, original_columns, column_map = standardise_offset_columns(df_raw)

        with st.expander("Detected Excel columns"):
            st.write("Original columns:")
            st.write(original_columns)
            st.write("Column mapping:")
            st.write(column_map)

        st.subheader("Offset Table Used")
        st.dataframe(df_offset, use_container_width=True)

        max_depth = float(df_offset["Depth"].max())

        if target_depth > max_depth:
            st.warning(
                f"Target depth {target_depth:.2f} m is deeper than maximum offset depth "
                f"{max_depth:.2f} m. Please upload a deeper offset table or reduce target depth."
            )

        if run_button:
            with st.spinner("Calculating..."):
                summary, straight_df, curved_df = run_correction(
                    df_offset=df_offset,
                    target_depth=target_depth,
                    target_azi=target_azi,
                    target_dip=target_dip
                )

                fig = create_trajectory_plot(straight_df, curved_df, summary)

            st.success("Calculation completed.")

            col1, col2, col3 = st.columns(3)

            col1.metric("Design Azi", f"{summary['Design Azi']:.2f}°")
            col2.metric("Design Dip", f"{summary['Design Dip']:.2f}°")
            col3.metric("3D Error", f"{summary['3D Error Distance']:.3f} m")

            st.subheader("Trajectory Plot")
            st.pyplot(fig)

            st.subheader("Calculation Summary")
            st.dataframe(pd.DataFrame([summary]), use_container_width=True)

            excel_file = make_excel_file(summary, straight_df, curved_df)
            png_file = make_png_file(fig)

            st.download_button(
                "Download Result Excel",
                data=excel_file,
                file_name="corrected_trajectory_output.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            st.download_button(
                "Download Trajectory PNG",
                data=png_file,
                file_name="drillhole_depth_position_trajectory.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(str(e))