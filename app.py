import pandas as pd
import dash
from dash import html, dcc, Input, Output
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import numpy as np

# ── Colour palette ───────────────────────────────────────────────────────────
COLORS = {
    'bg_primary':   '#0d0d14',
    'bg_secondary': '#16161f',
    'bg_card':      '#1e1e2d',
    'text_primary': '#e8e8f0',
    'text_muted':   '#8888a8',
    'accent':       '#5c9eff',
    'border':       '#2a2a3d',
}

# ── Data ─────────────────────────────────────────────────────────────────────
# Loaded once when the app starts. Stats are computed here and reused in the layout.
df = pd.read_csv('data/SDSS_DR18.csv')

total_objects  = len(df)
class_counts   = df['class'].value_counts()
redshift_min   = df['redshift'].min()
redshift_max   = df['redshift'].max()
redshift_mean  = df['redshift'].mean()

# ── Model training ────────────────────────────────────────────────────────────
# Separate copy so the Overview stats above keep using the full unfiltered data.
df_model = df.copy()
df_model['u - g'] = df_model['u'] - df_model['g']
df_model['g - r'] = df_model['g'] - df_model['r']
df_model['r - i'] = df_model['r'] - df_model['i']
df_model['i - z'] = df_model['i'] - df_model['z']

df_model = df_model[(df_model['r - i'] > -5) & (df_model['r - i'] < 5)]
df_model = df_model[(df_model['i - z'] > -5) & (df_model['i - z'] < 5)]
df_model = df_model[df_model['redshift'] >= 0]

X = df_model[['u - g', 'g - r', 'r - i', 'i - z']]
y = df_model['redshift']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred  = lr_model.predict(X_test)

lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2  = r2_score(y_test, lr_pred)

# ── Linear Regression: True vs Predicted figure ───────────────────────────────
max_redshift = float(y_test.max())

fig_lr_scatter = go.Figure()

fig_lr_scatter.add_trace(go.Scattergl(
    x=y_test,
    y=lr_pred,
    mode='markers',
    marker=dict(color='#5c9eff', size=3, opacity=0.35),
    name='Predictions',
))

fig_lr_scatter.add_trace(go.Scatter(
    x=[0, max_redshift],
    y=[0, max_redshift],
    mode='lines',
    line=dict(color='#ff6b6b', dash='dash', width=1.5),
    name='Perfect Prediction',
))

fig_lr_scatter.update_layout(
    xaxis_title='True Redshift',
    yaxis_title='Predicted Redshift',
    paper_bgcolor='#1e1e2d',
    plot_bgcolor='#1e1e2d',
    font=dict(color='#8888a8', size=12),
    xaxis=dict(gridcolor='#2a2a3d', zerolinecolor='#2a2a3d'),
    yaxis=dict(gridcolor='#2a2a3d', zerolinecolor='#2a2a3d'),
    legend=dict(bgcolor='#16161f', bordercolor='#2a2a3d', borderwidth=1),
    margin=dict(l=60, r=30, t=30, b=60),
    height=420,
)

# ── Helper: stat card ─────────────────────────────────────────────────────────
# A small reusable function. Pass it a value and a label, it returns a styled card.
def stat_card(value, label):
    return html.Div([
        html.Div(str(value), style={
            'fontSize':   '28px',
            'fontWeight': '700',
            'color':      COLORS['accent'],
        }),
        html.Div(label, style={
            'fontSize':  '12px',
            'color':     COLORS['text_muted'],
            'marginTop': '6px',
            'letterSpacing': '0.05em',
        }),
    ], style={
        'backgroundColor': COLORS['bg_card'],
        'border':          f"1px solid {COLORS['border']}",
        'borderRadius':    '4px',
        'padding':         '20px 24px',
        'flex':            '1',
        'minWidth':        '150px',
    })

# ── Tab styles ────────────────────────────────────────────────────────────────
tab_style = {
    'backgroundColor': COLORS['bg_secondary'],
    'color':           COLORS['text_muted'],
    'border':          f"1px solid {COLORS['border']}",
    'padding':         '10px 28px',
    'fontSize':        '14px',
    'letterSpacing':   '0.05em',
}

tab_selected_style = {
    'backgroundColor': COLORS['bg_card'],
    'color':           COLORS['text_primary'],
    'borderTop':       f"2px solid {COLORS['accent']}",
    'borderLeft':      f"1px solid {COLORS['border']}",
    'borderRight':     f"1px solid {COLORS['border']}",
    'borderBottom':    'none',
    'padding':         '10px 28px',
    'fontSize':        '14px',
    'letterSpacing':   '0.05em',
}

# ── Shared text styles ────────────────────────────────────────────────────────
section_heading = {
    'color':        COLORS['text_primary'],
    'fontSize':     '20px',
    'fontWeight':   '600',
    'marginBottom': '12px',
}

subsection_heading = {
    'color':        COLORS['text_primary'],
    'fontSize':     '17px',
    'fontWeight':   '600',
    'marginBottom': '10px',
}

body_text = {
    'color':      COLORS['text_muted'],
    'fontSize':   '15px',
    'lineHeight': '1.7',
    'maxWidth':   '760px',
}

divider = {
    'border':     'none',
    'borderTop':  f"1px solid {COLORS['border']}",
    'margin':     '36px 0',
}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Star-Shiftmetrics",
                style={'color': COLORS['text_primary'], 'margin': '0', 'fontSize': '28px'}),
        html.P("Photometric redshift prediction using SDSS DR18 data.",
               style={'color': COLORS['text_muted'], 'margin': '6px 0 0 0', 'fontSize': '14px'}),
    ], style={
        'backgroundColor': COLORS['bg_secondary'],
        'padding':         '24px 32px',
        'borderBottom':    f"1px solid {COLORS['border']}",
    }),

    # Navigation tabs
    dcc.Tabs(id='tabs', value='overview', children=[

        # ── Overview ──────────────────────────────────────────────────────────
        dcc.Tab(label='Overview', value='overview',
                style=tab_style, selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Div([

                            # About
                            html.H2("About This Project", style=section_heading),

                            html.Hr(style=divider),

                            # What is SDSS
                            html.H2("What is SDSS?", style=section_heading),

                            html.Hr(style=divider),

                            # Dataset stats
                            html.H2("Dataset at a Glance", style=section_heading),
                            html.Div([
                                stat_card(f"{total_objects:,}",                                  "Total Objects"),
                                stat_card(f"{redshift_min:.2f} – {redshift_max:.2f}",            "Redshift Range"),
                                stat_card(f"{redshift_mean:.3f}",                                "Mean Redshift"),
                                stat_card(f"{class_counts.get('GALAXY', 0):,}",                  "Galaxies"),
                                stat_card(f"{class_counts.get('STAR', 0):,}",                    "Stars"),
                                stat_card(f"{class_counts.get('QSO', 0):,}",                     "Quasars"),
                            ], style={
                                'display':   'flex',
                                'gap':       '16px',
                                'flexWrap':  'wrap',
                                'marginTop': '20px',
                            }),

                        ], style={'maxWidth': '960px', 'margin': '0 auto'}),
                    ], style={'padding': '40px 40px'}),
                ]),

        # ── Predictive Modelling ──────────────────────────────────────────────
        dcc.Tab(label='Predictive Modelling', value='predictive',
                style=tab_style, selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Div([

                            html.H2("Predictive Modelling", style=section_heading),

                            html.Hr(style=divider),

                            # Linear Regression subsection
                            html.H3("Linear Regression", style=subsection_heading),
                            html.Ul([
                                html.Li("What Linear Regression is and how it works."),
                                html.Li("Why it was chosen as the baseline model here."),
                            ], style={**body_text, 'paddingLeft': '20px'}),

                            # Metrics
                            html.P("Model Performance",
                                   style={'color': COLORS['text_muted'], 'fontSize': '13px',
                                          'letterSpacing': '0.08em', 'marginTop': '28px',
                                          'marginBottom': '0'}),
                            html.Div([
                                stat_card(f"{lr_mse:.4f}", "Mean Squared Error"),
                                stat_card(f"{lr_r2:.4f}",  "R² Score"),
                            ], style={'display': 'flex', 'gap': '16px', 'marginTop': '12px'}),

                            # True vs Predicted scatter plot
                            html.P("True vs Predicted Redshift",
                                   style={'color': COLORS['text_muted'], 'fontSize': '13px',
                                          'letterSpacing': '0.08em', 'marginTop': '36px',
                                          'marginBottom': '0'}),
                            dcc.Graph(figure=fig_lr_scatter,
                                      style={'marginTop': '12px'}),

                            html.Hr(style=divider),

                            # Colour index exploration
                            html.P("Colour Index Exploration",
                                   style={'color': COLORS['text_muted'], 'fontSize': '13px',
                                          'letterSpacing': '0.08em', 'marginBottom': '12px'}),
                            dcc.Dropdown(
                                id='colour-index-dropdown',
                                options=[
                                    {'label': 'u-g  vs  g-r', 'value': 'u-g_g-r'},
                                    {'label': 'g-r  vs  r-i', 'value': 'g-r_r-i'},
                                    {'label': 'r-i  vs  i-z', 'value': 'r-i_i-z'},
                                    {'label': 'u-g  vs  i-z', 'value': 'u-g_i-z'},
                                ],
                                value='u-g_g-r',
                                clearable=False,
                                style={
                                    'backgroundColor': COLORS['bg_card'],
                                    'color':           COLORS['text_primary'],
                                    'border':          f"1px solid {COLORS['border']}",
                                    'width':           '220px',
                                },
                            ),
                            html.Ul(id='colour-description',
                                    style={**body_text, 'paddingLeft': '20px', 'marginTop': '14px'}),
                            html.Div([
                                dcc.Graph(
                                    id='colour-index-graph',
                                    style={'flex': '1', 'minWidth': '0'},
                                ),
                                html.Div(
                                    id='click-info-panel',
                                    children=html.P(
                                        "Click any point to identify it",
                                        style={
                                            'color':     COLORS['text_muted'],
                                            'fontSize':  '13px',
                                            'textAlign': 'center',
                                            'marginTop': '60px',
                                        }
                                    ),
                                    style={
                                        'width':           '260px',
                                        'flexShrink':      '0',
                                        'backgroundColor': COLORS['bg_card'],
                                        'border':          f"1px solid {COLORS['border']}",
                                        'borderRadius':    '4px',
                                        'padding':         '20px',
                                    }
                                ),
                            ], style={
                                'display':        'flex',
                                'gap':            '20px',
                                'alignItems':     'flex-start',
                                'marginTop':      '16px',
                            }),

                        ], style={'maxWidth': '960px', 'margin': '0 auto'}),
                    ], style={'padding': '40px'}),
                ]),

        # ── Calculations ──────────────────────────────────────────────────────
        dcc.Tab(label='Calculations', value='calculations',
                style=tab_style, selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Div([

                            html.H2("Calculations", style=section_heading),

                            html.Hr(style=divider),

                        ], style={'maxWidth': '960px', 'margin': '0 auto'}),
                    ], style={'padding': '40px'}),
                ]),

        # ── FAQ ───────────────────────────────────────────────────────────────
        dcc.Tab(label='FAQ', value='faq',
                style=tab_style, selected_style=tab_selected_style,
                children=[
                    html.Div([
                        html.Div([

                            html.H2("FAQ", style=section_heading),

                            html.Hr(style=divider),

                        ], style={'maxWidth': '960px', 'margin': '0 auto'}),
                    ], style={'padding': '40px'}),
                ]),

    ], style={
        'backgroundColor': COLORS['bg_secondary'],
        'borderBottom':    f"1px solid {COLORS['border']}",
    }),

], style={
    'backgroundColor': COLORS['bg_primary'],
    'minHeight':       '100vh',
    'fontFamily':      "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    'margin':          '0',
})


# ── Callbacks ────────────────────────────────────────────────────────────────

# Colour index pair -> (x column, y column, x axis range, y axis range)
COLOUR_PAIRS = {
    'u-g_g-r': ('u - g', 'g - r', (-0.5, 3.0), (-0.5, 1.5)),
    'g-r_r-i': ('g - r', 'r - i', (-0.5, 1.5), (-0.5, 0.8)),
    'r-i_i-z': ('r - i', 'i - z', (-0.5, 0.8), (-0.5, 0.8)),
    'u-g_i-z': ('u - g', 'i - z', (-0.5, 3.0), (-0.5, 1.0)),
}

# Sample once - 15,000 points is enough to show the pattern clearly
df_plot = df_model.sample(n=15000, random_state=42)

# Lookback time lookup table — Planck 2018 flat ΛCDM (H0=67.4, Om=0.315, OL=0.685)
# Hubble time in Gyr: 977.8 / H0
_tH     = 977.8 / 67.4
_z_lt   = np.linspace(0, 7, 10000)
_dz_lt  = _z_lt[1] - _z_lt[0]
_E_lt   = np.sqrt(0.315 * (1 + _z_lt)**3 + 0.685)
_f_lt   = 1.0 / ((1 + _z_lt) * _E_lt)
_lt_arr = np.concatenate([[0.0], np.cumsum(0.5 * (_f_lt[:-1] + _f_lt[1:]) * _dz_lt)]) * _tH

def lookback_gyr(z):
    return float(np.interp(z, _z_lt, _lt_arr))

COLOUR_DESCRIPTIONS = {
    'u-g_g-r': [
        "u-g measures the brightness difference between ultraviolet and green light. Higher values indicate redder objects.",
        "g-r captures the green-to-red shift. Together these two indices describe the broad optical colour of each object.",
    ],
    'g-r_r-i': [
        "Both indices sit in the visible spectrum, tracing how light output changes across optical wavelengths.",
        "Galaxies at moderate redshift cluster toward higher values on both axes as their light shifts redward.",
    ],
    'r-i_i-z': [
        "r-i and i-z compare adjacent near-infrared bands, making this plot sensitive to higher-redshift objects.",
        "As redshift increases, visible light shifts into these infrared channels, seen as tighter clustering at higher values.",
    ],
    'u-g_i-z': [
        "The broadest colour comparison available, spanning ultraviolet to infrared, producing the strongest correlation with redshift.",
        "Yellow points (high redshift) consistently cluster toward higher values on both axes.",
    ],
}

@app.callback(
    Output('colour-index-graph', 'figure'),
    Output('colour-description', 'children'),
    Input('colour-index-dropdown', 'value')
)
def update_colour_scatter(pair):
    x_col, y_col, x_range, y_range = COLOUR_PAIRS[pair]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=df_plot[x_col],
        y=df_plot[y_col],
        mode='markers',
        marker=dict(
            color=df_plot['redshift'],
            colorscale='Plasma',
            size=3,
            opacity=0.6,
            colorbar=dict(
                title=dict(text='Redshift', font=dict(color='#8888a8')),
                tickfont=dict(color='#8888a8'),
                bgcolor='#1e1e2d',
                bordercolor='#2a2a3d',
            ),
        ),
        customdata=df_plot[['ra', 'dec', 'class', 'redshift']].values,
        hovertemplate='<b>%{customdata[2]}</b><br>Redshift: %{customdata[3]:.3f}<extra></extra>',
        showlegend=False,
    ))

    fig.update_layout(
        xaxis=dict(title=x_col, range=x_range, gridcolor='#2a2a3d', zerolinecolor='#2a2a3d'),
        yaxis=dict(title=y_col, range=y_range, gridcolor='#2a2a3d', zerolinecolor='#2a2a3d'),
        paper_bgcolor='#1e1e2d',
        plot_bgcolor='#1e1e2d',
        font=dict(color='#8888a8', size=12),
        margin=dict(l=60, r=30, t=30, b=60),
        height=420,
    )

    bullets = [html.Li(line) for line in COLOUR_DESCRIPTIONS[pair]]

    return fig, bullets


@app.callback(
    Output('click-info-panel', 'children'),
    Input('colour-index-graph', 'clickData'),
    prevent_initial_call=True,
)
def identify_point(clickData):
    if clickData is None:
        return html.P(
            "Click any point to identify it",
            style={'color': COLORS['text_muted'], 'fontSize': '13px',
                   'textAlign': 'center', 'marginTop': '60px'},
        )

    point     = clickData['points'][0]
    ra        = point['customdata'][0]
    dec       = point['customdata'][1]
    obj_class = point['customdata'][2]
    redshift  = point['customdata'][3]

    sdss_url = (
        f"https://skyserver.sdss.org/dr18/en/tools/explore/summary.aspx"
        f"?ra={ra}&dec={dec}"
    )

    row_s   = {'display': 'flex', 'justifyContent': 'space-between',
               'marginBottom': '10px', 'fontSize': '13px'}
    label_s = {'color': COLORS['text_muted']}
    value_s = {'color': COLORS['text_primary'], 'fontWeight': '600'}

    rows = [
        html.P("Object Identified", style={
            'color':        COLORS['text_primary'],
            'fontSize':     '14px',
            'fontWeight':   '600',
            'marginBottom': '16px',
        }),
        html.Div([html.Span("Class",    style=label_s),
                  html.Span(obj_class,  style=value_s)], style=row_s),
        html.Div([html.Span("Redshift", style=label_s),
                  html.Span(f"{redshift:.4f}", style=value_s)], style=row_s),
        html.Div([html.Span("RA",  style=label_s),
                  html.Span(f"{ra:.5f}", style=value_s)], style=row_s),
        html.Div([html.Span("Dec", style=label_s),
                  html.Span(f"{dec:.5f}", style=value_s)], style=row_s),
    ]

    if obj_class in ('GALAXY', 'QSO'):
        lt = lookback_gyr(redshift)
        rows.append(html.Div([
            html.Span("Lookback Time", style=label_s),
            html.Span(f"{lt:.2f} Gyr",  style=value_s),
        ], style=row_s))

    rows += [
        html.Hr(style={'border': 'none',
                       'borderTop': f"1px solid {COLORS['border']}",
                       'margin': '16px 0'}),
        html.A(
            "Open in SDSS Explorer",
            href=sdss_url,
            target="_blank",
            style={
                'color':          COLORS['accent'],
                'fontSize':       '13px',
                'textDecoration': 'none',
                'display':        'block',
                'textAlign':      'center',
            },
        ),
    ]

    return rows


if __name__ == '__main__':
    app.run(debug=True)
