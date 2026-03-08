import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# -----------------------------
# 1. LOAD PROCESSED DATA
# -----------------------------
df = pd.read_csv("data/processed/cams_long_umap.csv")

df["text"] = df["text"].fillna("").astype(str)
df["type"] = df["type"].fillna("unknown").astype(str).str.strip().str.lower()
df["row_id"] = df["row_id"].astype(int)
df["id"] = df["id"].astype(str)

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
# 3. HELPER FUNCTION TO BUILD FIGURE
# -----------------------------
def make_figure(filtered_df):
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
        margin=dict(l=0, r=0, t=50, b=0),
        height=850,
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


# Initial figure
initial_fig = make_figure(df)

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
        "backgroundColor": "#fafafa"
    },
    children=[

        html.H1(
            "Semantic Similarity of CAMS Constructs: Dataset Exploration Dashboard",
            style={"marginBottom": "6px"}
        ),

        html.Div(
            "Explore semantic clustering of text responses. Filter by type or participant, then click any point to view the text.",
            style={
                "marginBottom": "16px",
                "color": "#444"
            }
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "300px 1fr 360px",
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
            html.Strong("About the Visualization"),
            style={"marginBottom": "6px"}
    ),
        html.P(
            "Each point represents a single text response from the CAMS dataset "
            "(either a Driver or a Reason for Dying [RFD]).",
            style={"marginBottom": "6px"}
        ),

        html.P(
            "To compare responses, Natural Language Processing (NLP) was used to convert "
            "each piece of text into a numerical representation called an embedding.",
            style={"marginBottom": "6px"}
        ),

        html.P(
            "Embeddings capture aspects of the semantic meaning of text responses, "
            "allowing responses with similar themes or language to be compared.",
            style={"marginBottom": "6px"}
        ),

        html.P(
            "Because these embeddings exist in a very high-dimensional space "
            "(often hundreds of dimensions), UMAP (Uniform Manifold Approximation "
            "and Projection) was used to reduce them to three dimensions for visualization.",
            style={"marginBottom": "6px"}
        ),

        html.P(
            "Points that appear closer together represent responses that are more "
            "similar in meaning, while points farther apart represent responses "
            "that are less similar.",
            style={"marginBottom": "0"}
        ),
    ],
    style={
        "backgroundColor": "#f8f9fb",
        "padding": "12px",
        "borderRadius": "8px",
        "fontSize": "13px",
        "lineHeight": "1.4",
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
                        "backgroundColor": "white",
                        "padding": "10px",
                        "borderRadius": "10px",
                        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
                    },
                    children=[
                        dcc.Graph(
                            id="umap-graph",
                            figure=initial_fig,
                            style={"height": "85vh"}
                        )
                    ]
                ),

                # RIGHT PANEL
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
                )
            ]
        )
    ]
)

# -----------------------------
# 5. CALLBACK: UPDATE PLOT
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

    return make_figure(filtered_df)

# -----------------------------
# 6. CALLBACK: SHOW CLICKED TEXT
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
# 7. RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)