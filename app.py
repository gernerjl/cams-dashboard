import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, Input, Output

# -----------------------------
# 1. LOAD PROCESSED DATA
# -----------------------------
df = pd.read_csv("data/processed/cams_long_umap.csv")

df["text"] = df["text"].fillna("").astype(str)
df["type"] = df["type"].fillna("Unknown").astype(str)
df["row_id"] = df["row_id"].astype(int)


# -----------------------------
# 3. HELPER FUNCTION TO BUILD FIGURE
# -----------------------------
def make_figure(filtered_df, color_var="type"):
    fig = px.scatter_3d(
        filtered_df,
        x="x",
        y="y",
        z="z",
        color=color_var,
        hover_name="type",
        hover_data={
            "text": True,
            "x": False,
            "y": False,
            "z": False,
            "row_id": False
        },
        custom_data=["row_id", "text", "type"]
    )

    fig.update_traces(
        marker=dict(size=5),
        selector=dict(mode="markers")
    )

    fig.update_layout(
        title={
            "text": "3D UMAP of Drivers and reasons for Dying",
            "x": 0.03,
            "xanchor": "left",
            "y": 0.98,
            "yanchor": "top"
        },
        margin=dict(l=0, r=0, t=50, b=0),
        height=850,
        legend=dict(
            title="Type",
            font=dict(size=14),
            title_font=dict(size=16),
            itemsizing="constant"
        )
    )

    return fig


# -----------------------------
# 4. DASH APP
# -----------------------------
app = Dash(__name__)

type_options = [{"label": "All", "value": "All"}] + [
    {"label": t, "value": t} for t in sorted(df["type"].unique())
]

app.layout = html.Div(
    style={
        "fontFamily": "Arial, sans-serif",
        "padding": "16px",
        "backgroundColor": "#fafafa"
    },
    children=[

        html.H1(
            "Semantic Similarity of CAMS Constructs: Exploration Dashboard",
            style={
                "marginBottom": "6px"
            }
        ),

        html.Div(
            "Explore semantic clustering of text responses. Click any point to view the full text.",
            style={
                "marginBottom": "16px",
                "color": "#444"
            }
        ),

        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "260px 1fr 360px",
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

                        html.Div(
                            [
                                html.Hr(),
                                html.H4("Coming later... :)"),
                                html.Ul([
                                    html.Li("Color by qualitative code"),
                                    html.Li("Filter by theme"),
                                    html.Li("Driver vs RFD overlays"),
                                    html.Li("Participant-level exploration")
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
                            figure=make_figure(df),
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
    Input("type-filter", "value")
)
def update_graph(selected_type):
    if selected_type == "All":
        filtered_df = df.copy()
    else:
        filtered_df = df[df["type"] == selected_type].copy()

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

    point = clickData["points"][0]
    row_id = point["customdata"][0]

    row = df.loc[df["row_id"] == row_id].iloc[0]

    panel_children = [
        html.H3("Selected Point", style={"marginTop": "0"}),
        html.P([
            html.Strong("Type: "),
            row["type"]
        ]),
        html.P([
            html.Strong("Row ID: "),
            str(row["row_id"])
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

    # Add optional columns later if they exist
    optional_cols = [
        "participant_id",
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
server = app.server

if __name__ == "__main__":
    app.run(debug=True)