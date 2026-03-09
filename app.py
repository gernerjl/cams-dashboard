import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output

# -----------------------------
# 1. LOAD PROCESSED DATA
# -----------------------------
df = pd.read_csv("data/processed/cams_long_umap.csv")
sim_df = pd.read_csv("data/processed/participant_similarity_results.csv")

df["text"] = df["text"].fillna("").astype(str)
df["type"] = df["type"].fillna("unknown").astype(str).str.strip().str.lower()
df["row_id"] = df["row_id"].astype(int)
df["id"] = df["id"].astype(str)

sim_df["id"] = sim_df["id"].astype(str)

# Standardize type labels
type_map = {
    "driver": "drivers",
    "drivers": "drivers",
    "rfd": "rfd",
    "reason for dying": "rfd",
    "reasons for dying": "rfd"
}
df["type"] = df["type"].replace(type_map)

# Pretty labels for display
display_map = {
    "drivers": "Drivers",
    "rfd": "RFD"
}
df["type_label"] = df["type"].map(display_map).fillna(df["type"])

# Short hover preview
df["text_preview"] = df["text"].str.slice(0, 80)
df.loc[df["text"].str.len() > 80, "text_preview"] += "..."

# Use row_id as index
df = df.set_index("row_id", drop=False)

# -----------------------------
# 2. GLOBAL AXIS LIMITS
# -----------------------------
x_range = [df["x"].min(), df["x"].max()]
y_range = [df["y"].min(), df["y"].max()]
z_range = [df["z"].min(), df["z"].max()]

# -----------------------------
# 3. HELPER FUNCTIONS
# -----------------------------
def make_umap_figure(filtered_df):
    fig = px.scatter_3d(
        filtered_df,
        x="x",
        y="y",
        z="z",
        color="type",
        color_discrete_map={
            "drivers": "#1f77b4",
            "rfd": "#d62728"
        },
        category_orders={
            "type": ["drivers", "rfd"]
        },
        hover_name="type_label",
        hover_data={
            "id": True,
            "text_preview": True,
            "x": False,
            "y": False,
            "z": False,
            "row_id": False,
            "type": False,
            "type_label": False
        },
        custom_data=["row_id"]
    )

    marker_size = 5 if len(filtered_df) > 20 else 8
    fig.update_traces(marker=dict(size=marker_size))

    fig.for_each_trace(
        lambda trace: trace.update(name=display_map.get(trace.name, trace.name))
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        height=600,
        scene=dict(
            xaxis=dict(range=x_range, title="x", autorange=False),
            yaxis=dict(range=y_range, title="y", autorange=False),
            zaxis=dict(range=z_range, title="z", autorange=False),
            aspectmode="cube"
        ),
        legend=dict(
            title="Type",
            font=dict(size=14),
            title_font=dict(size=16),
            itemsizing="constant"
        ),
        uirevision="fixed"
    )

    return fig


def make_overall_similarity_figure(similarity_df):
    mean_dd = similarity_df["driver_driver_similarity"].dropna().mean()
    mean_rr = similarity_df["rfd_rfd_similarity"].dropna().mean()
    mean_dr = similarity_df["driver_rfd_similarity"].dropna().mean()

    plot_df = pd.DataFrame({
        "comparison": ["Driver–Driver", "RFD–RFD", "Driver–RFD"],
        "similarity": [mean_dd, mean_rr, mean_dr]
    })

    fig = px.bar(
        plot_df,
        x="comparison",
        y="similarity",
        color="comparison",
        color_discrete_map={
            "Driver–Driver": "#1f77b4",
            "RFD–RFD": "#ff7f0e",
            "Driver–RFD": "#2ca02c"
        },
        text=plot_df["similarity"].round(2)
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        title="Overall Sample Mean Semantic Similarity",
        showlegend=False,
        height=240,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Mean Cosine Similarity",
        xaxis_title="",
        yaxis=dict(range=[0, 0.45])
    )

    return fig


def make_participant_similarity_figure(selected_id):
    if selected_id == "All":
        fig = go.Figure()
        fig.add_annotation(
            text="Select a participant to view.",
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Participant-Level Semantic Similarity",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    row = sim_df[sim_df["id"] == selected_id]

    if row.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity summary available for this participant.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(
            title="Participant-Level Semantic Similarity",
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    row = row.iloc[0]

    plot_df = pd.DataFrame({
        "comparison": ["Driver–Driver", "RFD–RFD", "Driver–RFD"],
        "similarity": [
            row["driver_driver_similarity"],
            row["rfd_rfd_similarity"],
            row["driver_rfd_similarity"]
        ]
    })

    fig = px.bar(
        plot_df,
        x="comparison",
        y="similarity",
        color="comparison",
        color_discrete_map={
            "Driver–Driver": "#1f77b4",
            "RFD–RFD": "#ff7f0e",
            "Driver–RFD": "#2ca02c"
        },
        text=plot_df["similarity"].round(2)
    )

    fig.update_traces(textposition="outside")

    fig.update_layout(
        title=f"Participant {selected_id}: Semantic Similarity",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis_title="Cosine Similarity",
        xaxis_title=""
    )

    return fig


# Initial figures
initial_umap_fig = make_umap_figure(df)
initial_overall_fig = make_overall_similarity_figure(sim_df)
initial_participant_fig = make_participant_similarity_figure("All")

# -----------------------------
# 4. DASH APP
# -----------------------------
app = Dash(__name__)
server = app.server

type_options = [{"label": "All", "value": "All"}] + [
    {"label": display_map.get(t, t), "value": t}
    for t in sorted(df["type"].unique())
]

id_options = [{"label": "All participants", "value": "All"}] + [
    {"label": pid, "value": pid}
    for pid in sorted(df["id"].unique())
]

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "16px",
        "backgroundColor": "#fafafa",
        "maxWidth": "1400px",
        "margin": "0 auto"
    },
    children=[
        html.H1(
            "Exploring the Semantic Similarity of CAMS Constructs",
            style={"marginBottom": "6px"}
        ),


        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "260px minmax(700px, 1fr) 460px",
                "gap": "16px",
                "alignItems": "start"
            },
            children=[

                # LEFT PANEL
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "padding": "14px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
                    },
                    children=[
                        html.H3("Controls", style={"marginTop": "0"}),

                        html.Label("Filter by type"),
                        dcc.Dropdown(
                            id="type-filter",
                            options=type_options,
                            value="All",
                            clearable=False
                        ),

                        html.Br(),

                        html.Label("Filter by participant"),
                        dcc.Dropdown(
                            id="id-filter",
                            options=id_options,
                            value="All",
                            clearable=False
                        ),

                        html.Br(),

html.Div(
    children=[
        html.P(
            html.Strong("About This Page"),
            style={"marginBottom": "8px"}
        ),

        html.P(
            "This dashboard explores semantic similarity among CAMS Drivers and Reasons for Dying (RFD) using natural language processing (NLP). "
            "It combines an interactive 3D visualization with summary charts to show both individual text responses and broader similarity patterns.",
            style={"marginBottom": "8px"}
        ),

        html.P(
            html.Strong("How to use it"),
            style={"marginBottom": "6px"}
        ),

        html.Ul([
            html.Li("Use the dropdowns to filter by response type or participant."),
            html.Li("Rotate and zoom the 3D plot to explore how responses cluster in semantic space."),
            html.Li("Click any point to view the full text, participant ID, and UMAP coordinates."),
            html.Li("Select a participant to view their Driver–Driver, RFD–RFD, and Driver–RFD similarity values."),
            html.Li("Use the bottom chart to compare overall mean semantic similarity across the full sample.")
        ], style={"paddingLeft": "20px", "marginTop": "0", "marginBottom": "8px"}),

        html.P(
            html.Strong("How to interpret it"),
            style={"marginBottom": "6px"}
        ),

        html.Ul([
            html.Li("Each point in the 3D plot represents one text response (a Driver or an RFD)."),
            html.Li("Points that appear closer together represent responses that are more similar in meaning."),
            html.Li("The 3D axes themselves do not represent directly interpretable units; they are a visualization of high-dimensional semantic relationships."),
            html.Li("The participant-level and overall bar charts show average cosine similarity, where higher values indicate greater semantic overlap.")
        ], style={"paddingLeft": "20px", "marginTop": "0", "marginBottom": "0"})
    ],
    style={
        "backgroundColor": "#f8f9fb",
        "padding": "12px",
        "borderRadius": "8px",
        "fontSize": "13px",
        "lineHeight": "1.45",
        "border": "1px solid #e3e6eb"
    }
),

                        html.Br(),

                        html.Div(
                            [
                                html.Hr(),
                                html.H4("Coming later... :)"),
                                html.Ul([
                                    html.Li("Color by qualitative code"),
                                    html.Li("Filter by theme")
                                ])
                            ],
                            style={"color": "#555", "fontSize": "14px"}
                        )
                    ]
                ),

                # CENTER PANEL
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px"
                    },
                    children=[
                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
                            },
                            children=[
                                html.H3("3D UMAP Visualization", style={"marginTop": "0"}),
                                dcc.Graph(
                                    id="umap-graph",
                                    figure=initial_umap_fig,
                                    style={"height": "65vh"}
                                )
                            ]
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
                            },
                            children=[
                                dcc.Graph(
                                    id="overall-similarity-graph",
                                    figure=initial_overall_fig
                                )
                            ]
                        )
                    ]
                ),

                # RIGHT PANEL
                html.Div(
                    style={
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "16px"
                    },
                    children=[
                        html.Div(
                            id="text-panel",
                            style={
                                "backgroundColor": "white",
                                "padding": "14px",
                                "borderRadius": "10px",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
                                "minHeight": "300px"
                            },
                            children=[
                                html.H3("Selected Point", style={"marginTop": "0"}),
                                html.P("Click a point in the plot to view its full text and metadata.")
                            ]
                        ),

                        html.Div(
                            style={
                                "backgroundColor": "white",
                                "padding": "10px",
                                "borderRadius": "10px",
                                "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
                            },
                            children=[
                                dcc.Graph(
                                    id="participant-similarity-graph",
                                    figure=initial_participant_fig
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# -----------------------------
# 5. CALLBACK: UPDATE UMAP
# -----------------------------
@app.callback(
    Output("umap-graph", "figure"),
    Input("type-filter", "value"),
    Input("id-filter", "value")
)
def update_graph(selected_type, selected_id):
    filtered_df = df.copy()

    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["type"] == selected_type]

    if selected_id != "All":
        filtered_df = filtered_df[filtered_df["id"] == selected_id]

    return make_umap_figure(filtered_df)

# -----------------------------
# 6. CALLBACK: UPDATE PARTICIPANT SIMILARITY CHART
# -----------------------------
@app.callback(
    Output("participant-similarity-graph", "figure"),
    Input("id-filter", "value")
)
def update_participant_similarity(selected_id):
    return make_participant_similarity_figure(selected_id)

# -----------------------------
# 7. CALLBACK: SHOW CLICKED TEXT
# -----------------------------
@app.callback(
    Output("text-panel", "children"),
    Input("umap-graph", "clickData")
)
def display_click_data(clickData):
    if clickData is None:
        return [
            html.H3("Selected Point", style={"marginTop": "0"}),
            html.P("Click a point in the plot to view its full text and metadata.")
        ]

    row_id = clickData["points"][0]["customdata"][0]
    row = df.loc[row_id]

    panel_children = [
        html.H3("Selected Point", style={"marginTop": "0"}),
        html.P([
            html.Strong("Participant ID: "),
            str(row["id"])
        ]),
        html.P([
            html.Strong("Type: "),
            row["type_label"]
        ]),
        html.P([
            html.Strong("Row ID: "),
            str(row["row_id"])
        ]),
        html.P([
            html.Strong("UMAP Coordinates: "),
            f"({row['x']:.3f}, {row['y']:.3f}, {row['z']:.3f})"
        ]),
        html.Hr(),
        html.H4("Full Text"),
        html.Div(
            row["text"],
            style={
                "whiteSpace": "pre-wrap",
                "lineHeight": "1.5",
                "backgroundColor": "#f7f7f7",
                "padding": "10px",
                "borderRadius": "8px",
                "border": "1px solid #e0e0e0"
            }
        )
    ]

    optional_cols = [
        "driver_code",
        "driver_theme",
        "rfd_code",
        "rfd_theme"
    ]

    available_optional = [col for col in optional_cols if col in df.columns]

    if available_optional:
        panel_children.append(html.Hr())
        panel_children.append(html.H4("Additional Metadata"))

        for col in available_optional:
            val = row[col]
            panel_children.append(
                html.P([
                    html.Strong(f"{col}: "),
                    str(val)
                ])
            )

    return panel_children

# -----------------------------
# 8. RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)