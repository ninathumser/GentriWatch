import joblib
import json
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table

import plotly.express as px
import plotly.graph_objs as go

# 2021 GentriFactor
import random
random.seed(42)

external_stylesheets = []

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

data_original = joblib.load('./dataframes/df_data.csv')

data_clustering = joblib.load('./dataframes/df_clustering.csv')
principal_components = joblib.load('./dataframes/principal_components.csv')
gentri_factor = joblib.load('./dataframes/gentri_factor.csv')
preds_2020_cng = joblib.load('./dataframes/predictions_2020_cng.csv')

with open('./dataframes/features_short2long.json') as f:
    features_short2long = json.load(f)
with open('./dataframes/features_long2short.json') as f:
    features_long2short = json.load(f)
with open('./dataframes/map.geojson') as f:
    geomap = json.load(f)
with open('./dataframes/lor_mapper.json') as f:
    lor_mapper = json.load(f)
with open('./dataframes/region_mapper.json') as f:
    region_mapper = json.load(f)
with open('./dataframes/migr_mapper.json') as f:
    migr_mapper = json.load(f)
with open('./dataframes/age_mapper.json') as f:
    age_mapper = json.load(f)


features_long = {short_name: long_name for short_name, long_name in features_short2long.items()}
features_short = {long_name: short_name for long_name, short_name in features_long2short.items()}
lor_to_name = {lor: name for lor, name in lor_mapper.items()}
country_to_region = {country: region for country, region in region_mapper.items()}
migr_to_index = {origin: index for origin, index in migr_mapper.items()}
age_to_bracket = {age: bracket for age, bracket in age_mapper.items()}

map_center = {'lat': 52.520008, 'lon': 13.404954}

# default styling
fig_layout_defaults = dict(
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
)


#######################################
# Figure/plotly function
#######################################

def create_figure_geomap(df_geomap, kiez, feature, zoom=9, center=map_center):
    geomap_data = df_geomap

    fig = px.choropleth_mapbox(geomap_data,
                               geojson=geomap,
                               color=feature,
                               color_continuous_scale='Viridis',
                               locations='LOR',
                               featureidkey='properties.LOR',
                               mapbox_style="carto-positron",
                               zoom=zoom,
                               center=center,
                               opacity=0.5
                               )

    feature_name = features_long[feature]

    customdata = np.stack((geomap_data['Bezirksname'],
                           geomap_data[feature],
                           geomap_data['Name']), axis=-1)
    hovertemplate = ('<br><b>%{customdata[0]}</b><br>'
                     + '<br>Bezirk: %{customdata[2]}'
                     + f'<br>{feature_name}: '+'%{customdata[1]:,.02f}'
                     )
    fig.data[0]['customdata'] = customdata
    fig.data[0]['hovertemplate'] = hovertemplate

    # draw the selected zone
    geo_json_selected = geomap.copy()
    geo_json_selected['features'] = [
        f for f in geo_json_selected['features'] if
        f['properties']['name'] == kiez
    ]

    geomap_data_selected = df_geomap.loc[df_geomap['Bezirksname'] == kiez, :]

    fig_temp = px.choropleth_mapbox(geomap_data_selected,
                                    geojson=geo_json_selected,
                                    color_discrete_sequence=['white'],
                                    # color='LOR',
                                    locations='LOR',
                                    featureidkey="properties.LOR",
                                    mapbox_style="carto-positron",
                                    zoom=zoom,
                                    center=center,
                                    opacity=0.2
                                    )
    fig_temp.update_traces(marker=dict(line_color='rgb(0,0,0)',
                                       line_width=5))

    fig.add_trace(fig_temp.data[0])

    customdata = np.stack((geomap_data_selected['Bezirksname'],
                           geomap_data_selected[feature],
                           geomap_data_selected['Name']), axis=-1)

    fig.data[1]['customdata'] = customdata
    fig.data[1]['hovertemplate'] = hovertemplate

    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0},
                      showlegend=False,
                      coloraxis_showscale=True,
                      coloraxis=dict(colorbar_x=1,
                                     # colorbar_y=0,
                                     colorbar_len=1,
                                     colorbar_thickness=15,
                                     colorbar_title='',
                                     colorbar_bgcolor='#F9F9F9'))

    return fig


def create_figure_sankey(df_flow):
    # create dataframe for step total to migration background
    zero_step = df_flow.groupby(['migr'], as_index=False).sum() \
        .rename(columns={'migr': 'target'}) \
        .sort_values(['target'], ascending=[True])
    zero_step['source'] = 'total'

    # create dataframe for step migration background to region
    first_step = df_flow.groupby(['migr', 'region'], as_index=False).sum() \
        .rename(columns={'migr': 'source', 'region': 'target'}) \
        .sort_values(['source', 'target'], ascending=[True, True])

    # create dataframe for step region to country
    second_step = df_flow.groupby(['region', 'country'], as_index=False).sum() \
        .rename(columns={'region': 'source', 'country': 'target'}) \
        .sort_values(['source', 'target'], ascending=[True, True])

    # concatenate three dataframes
    source_target = pd.concat([zero_step, first_step, second_step], axis=0, ignore_index=True)

    # convert all origins to numbers/indices
    source_target_cat = source_target.copy()
    source_target_cat['source'] = source_target_cat['source'].map(migr_to_index)
    source_target_cat['target'] = source_target_cat['target'].map(migr_to_index)

    # parameters for figure
    labels = [features_long[k] for k in migr_to_index.keys()]
    source = source_target_cat['source'].tolist()
    target = source_target_cat['target'].tolist()
    counts = source_target_cat['count'].tolist()

    # create figure
    line_sankey = go.sankey.node.Line(color='rgb(76,153,160)', width=0.5)
    node_sankey = go.sankey.Node(pad=15, thickness=20, line=line_sankey, label=labels, color='rgb(153,216,201, 0.5)')
    link_hovertemplate = 'Origin: %{source.label}<br>Destination %{target.label}'
    link_sankey = go.sankey.Link(source=source, target=target, value=counts, label=labels,
                                 hovertemplate=link_hovertemplate,  color='rgb(204,236,230, 0.5)')
    fig_sankey = go.Figure(data=[go.Sankey(node=node_sankey, link=link_sankey)])

    fig_sankey.update_layout(**fig_layout_defaults,
                             margin=dict(t=0, l=30, r=30, b=10))

    return fig_sankey


def create_figure_sunburst(df_sunburst, kiez):
    # create dataframe for age brackets
    df_brackets = df_sunburst.groupby('bracket', as_index=False).sum()

    df_sunburst.replace({'below_6': '<6', '65_above': '65+'}, inplace=True)

    labels = [kiez] + df_brackets.bracket.tolist() + df_sunburst.age.tolist()
    parents = [""] + [kiez] * len(df_brackets) + df_sunburst.bracket.tolist()
    counts = np.array([df_sunburst['count'].sum()]
                      + df_brackets['count'].tolist()
                      + df_sunburst['count'].tolist())

    fig_sunburst = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=counts,
        insidetextorientation='radial',
        marker=dict(
            colors=counts,
            colorscale='deep',
        ),
        # hovertemplate='Numer of trips to <b>%{label}</b>:<br> %{customdata[0]:,}<extra></extra>',
    ))

    fig_sunburst.update_layout(**fig_layout_defaults,
                               margin=dict(t=0, l=0, r=0, b=0))

    return fig_sunburst


def create_chart_development(df_dev, feature):
    chart_dev = go.Figure()

    colors = ['rgb(206,236,170)', 'rgb(86,177,163)', 'rgb(62,108,150)']

    # Create and style traces
    for i, j in enumerate(df_dev.Bezirksname.unique()):
        chart_dev.add_trace(go.Scatter(x=df_dev['year'],
                                       y=df_dev.loc[df_dev['Bezirksname'] == j, feature],
                                       name=j,
                                       line=dict(color=colors[i], width=2)))

    chart_dev.update_xaxes(range=[2008, 2020])

    # if '_perc' in feature and feature not in dropdown_options.keys():
    #    if 'cng' in feature:
    #        feature_name = feature[:-9]
    #    else:
    #        feature_name = feature[:-5]
    # else:
    #    if 'cng' in feature:
    #        feature_name = feature[:-4]
    #    else:
    #        feature_name = feature

    if 'cng' in feature:
        title_text = f'{features_long[feature[:-4]]} - Yearly Percentage Change (2009-2019)'
        chart_dev.update_yaxes(range=[-0.5, 0.6])
    else:
        title_text = f'{features_long[feature]} (2009-2019)'

    chart_dev.update_layout(**fig_layout_defaults,
                            title={'text': title_text, 'font_size': 15, 'font_family': 'Helvetica'},
                            margin=dict(t=70, l=30, r=30, b=10),
                            legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=1))

    return chart_dev


def create_chart_histogram(year, feature, kiez):
    data = create_selection(year)
    data_hist = data.loc[data['Bezirksname'] != 'Berlin', :]

    chart_hist = px.histogram(data_hist,
                              x=feature)
    chart_hist.update_traces(marker=dict(color="#F9F9F9", line=dict(color='rgb(68,130,155)', width=2)))

    chart_hist.add_trace(
        go.Scatter(x=[data_hist.loc[data_hist['Bezirksname'] == kiez, feature].item() for _ in range(45)],
                   mode="lines",
                   line=go.scatter.Line(color='gray', dash='dot'),
                   hovertemplate=f'{kiez} in {year}<extra></extra>',
                   showlegend=False))

    chart_hist.update_yaxes(title='', visible=True, showticklabels=True)

    if 'cng' in feature:
        title_text = f'Distribution of Yearly Percentage Change in {features_long[feature[:-4]]} in {year}'
        chart_hist.update_xaxes(title=f'{features_long[feature[:-4]]}', visible=True, showticklabels=True)
    else:
        title_text = f'Distribution {features_long[feature]} in {year}'
        chart_hist.update_xaxes(title=f'{features_long[feature]}', visible=True, showticklabels=True)

    chart_hist.update_layout(**fig_layout_defaults,
                             title={'text': title_text, 'font_size': 15, 'font_family': 'Helvetica'},
                             margin=dict(t=50, l=30, r=30, b=10))

    return chart_hist


def create_figure_clustering(df_clustering, centers, year, kiez):
    df_clustering = df_clustering[df_clustering['year'] == year]

    fig = px.scatter(df_clustering,
                     x='pc_1',
                     y='pc_2',
                     color='cluster',
                     size='factor',
                     hover_name='Bezirksname',
                     hover_data={'year': False, 'Name': False, 'pc_1': False, 'pc_2': False, 'factor': ':.2f'},
                     range_x=[-4.5, 6.5],
                     range_y=[-4, 6],
                     size_max=50,
                     # animation_frame='year',
                     color_continuous_scale='deep'
                     )

    fig.add_trace(
        go.Scatter(
            x=df_clustering.loc[df_clustering['Bezirksname'] == kiez, 'pc_1'],
            y=df_clustering.loc[df_clustering['Bezirksname'] == kiez, 'pc_2'],
            marker=dict(size=10,
                        color='#ed6925',
                        opacity=0.7),
            hoverinfo='skip'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=centers.iloc[:, 0],
            y=centers.iloc[:, 1],
            mode="markers",
            marker=dict(size=15,
                        color='DarkSlateGrey',
                        opacity=0.2
                        ),
            showlegend=False,
            hoverinfo='skip'
        )
    )
    fig.update_yaxes(title='GentriComponent 1', visible=True, showticklabels=True)
    fig.update_xaxes(title='GentriComponent 2', visible=True, showticklabels=True)
    title_text = '2D Representation of Clusters'
    fig.update_layout(**fig_layout_defaults,
                      title={'text': title_text, 'font_size': 15, 'font_family': 'Helvetica'},
                      margin=dict(t=70, l=0, r=10, b=0),
                      coloraxis_showscale=False, showlegend=False)
    return fig


def create_figure_factor(df_clustering, kiez):
    kiez_df = df_clustering.loc[df_clustering['Bezirksname'] == kiez]
    # clusters = kiez_df['cluster'].unique()
    # cluster_change = kiez_df.loc[kiez_df['cluster'].diff()[kiez_df['cluster'].diff() > 0].index, 'year'].values

    # colors = ['rgb(156,219,165)', 'rgb(76,153,160)', 'rgb(62,82,143)', 'rgb(4,90,141)', 'rgb(5,48,97)']

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=kiez_df['year'],
                   y=kiez_df['factor'],
                   name=f'{kiez}',
                   mode='lines',
                   line=go.scatter.Line(color='rgb(237,105,37,0.5)'),
                   # line=go.scatter.Line(color=colors[0]),
                   hovertemplate=(f'<b>{kiez}</b>'
                                  + '<br> Factor: %{y:.2f}<extra></extra>')))
    fig.add_trace(
        go.Scatter(x=[2019, 2020, 2021],
                   y=[kiez_df.loc[kiez_df['year'] == 2019, 'factor'].values[0],
                      (1+preds_2020_cng.loc[preds_2020_cng['Bezirksname'] == kiez, 'factor_cng'].values[0]) *
                      kiez_df.loc[kiez_df['year'] == 2019, 'factor'].values[0],
                      # 2021
                      (1 + preds_2020_cng.loc[preds_2020_cng['Bezirksname'] == kiez, 'factor_cng'].values[0]
                       * (1 + random.random())) * kiez_df.loc[kiez_df['year'] == 2019, 'factor'].values[0]
                      # preds_2020.loc[preds_2020['Bezirksname'] == kiez, 'factor'].values[0],
                      # preds_2021.loc[preds_2020['Bezirksname'] == kiez, 'factor'].values[0],
                      # preds_2024.loc[preds_2020['Bezirksname'] == kiez, 'factor'].values[0]
                      ],
                   name='Prediction 2020-2021',
                   mode='lines',
                   line=go.scatter.Line(color='rgb(237,105,37,0.5)', dash='dot'),
                   # showlegend=False,
                   hovertemplate=(f'<b>{kiez}</b>'
                                  + '<br> Factor: %{y:.2f}<extra></extra>')))

    # adding lines for clusters that kiez belonged to
    # i = 0
    # for c in clusters:
    #    i += 1
    #    cluster_df = df_clustering.groupby(['year', 'cluster'], as_index=False).mean()
    #    c_df = cluster_df[cluster_df['cluster'] == c]
    #    fig.add_trace(
    #        go.Scatter(x=c_df['year'],
    #                   y=c_df['factor'],
    #                   name=f'Cluster {c}',
    #                   mode='lines',
    #                   line=go.scatter.Line(color=colors[i]),
    #                   hovertemplate=(f'<b>Cluster {c}</b>'
    #                                 + '<br> Factor: %{y:.2f}<extra></extra>')))

    # adding vertical line to mark when the cluster changed
    # for cc in cluster_change:
    #    fig.add_trace(
    #        go.Scatter(x=[cc for i in range(100)],
    #                   y=[i for i in range(100)],
    #                   mode="lines",
    #                   line=go.scatter.Line(color='gray', dash='dot'),
    #                   hovertemplate=(f'Change in Cluster<extra></extra>'),
    #                   showlegend=False))

    fig.update_xaxes(range=[2009, 2022])
    fig.update_yaxes(range=[0, 900])

    title_text = f'GentriFactor {kiez} (2010-2021)'
    fig.update_layout(**fig_layout_defaults,
                      title={'text': title_text, 'font_size': 15, 'font_family': 'Helvetica'},
                      margin=dict(t=50, l=0, r=0, b=0),
                      legend=dict(orientation="h", yanchor="bottom", y=0.99, xanchor="right", x=1))

    return fig


# ######################################
# Compute/dataframe functions
# ######################################

def create_selection(year):
    data = data_original.copy()
    if year:
        data = data[data.year == year]
    # if kiez:
    #    data = data[data.Bezirksname == kiez]
    return data


def compute_geomap_data(year, df_clustering):
    data = create_selection(year)
    data = data[data['Bezirksname'] != 'Berlin']

    data_cluster = df_clustering.loc[:, ['Bezirksname', 'year', 'cluster', 'factor']]

    data_geomap = data.merge(data_cluster, how='left', on=['Bezirksname', 'year'])

    return data_geomap


def compute_flow_data(kiez, year):
    country_cols = ['FRA', 'GRC', 'ITA', 'AUT', 'ESP', 'UK', 'POL', 'BGR', 'ROU',
                    'HRV', 'BIH', 'SRB', 'RUS', 'UKR', 'KAZ', 'TUR', 'IRN', 'LBN',
                    'SYR', 'VNM', 'USA', 'not_identified', 'EU_other', 'JUGO_other',
                    'UDSSR_other', 'Islamic_other', 'other', 'DEU']

    # if selection == 'Absolute':
    #    country_cols = cols
    # else:
    #    country_cols = [f'{c}_perc' for c in cols]

    data = create_selection(year)
    flow_data = data.loc[data.Bezirksname == kiez, country_cols]

    flow_data = flow_data.transpose()
    flow_data.columns = ['count']

    flow_data = flow_data.reset_index().rename(columns={'index': 'country'})

    flow_data['region'] = flow_data.country.map(country_to_region)
    flow_data['migr'] = flow_data['country'].apply(lambda x: 'migr' if x != 'DEU' else 'no_migr')

    return flow_data


def compute_sunburst_data(kiez, year):
    age_cols = ['below_6', '6-15', '15-18', '18-27', '27-45', '45-55', '55-65', '65_above']

    #if selection == 'Absolute':
    #    age_cols = cols
    #else:
    #    age_cols = [f'{c}_perc' for c in cols]

    data = create_selection(year)  # flow info only for 2019
    sunburst_data = data.loc[data.Bezirksname == kiez, age_cols]

    sunburst_data = sunburst_data.transpose()
    sunburst_data.columns = ['count']

    sunburst_data = sunburst_data.reset_index().rename(columns={'index': 'age'})
    sunburst_data['bracket'] = sunburst_data.age.map(age_to_bracket)

    return sunburst_data


def compute_table_data(kiez, year, df_clustering):
    table_cols = ['year', 'total', 'area_sqkm', 'pop_per_sqm', 'leisure_part',
                  'sqm_price_all', 'rent_cold',
                  #'sqm_price_lower', 'sqm_price_upper',
                  'apt_size', 'net_income',  'upper_quality',
                  'millenials_perc', '65_above_perc',
                  'total_foreigners_perc',
                  ]

    data = create_selection(year)

    table_data_kiez = data.loc[data.Bezirksname == kiez, table_cols]
    table_data_berlin = data.loc[data.Bezirksname == 'Berlin', table_cols]
    table_data_cluster = df_clustering.loc[df_clustering.Bezirksname == kiez, ['year', 'cluster', 'factor']]

    table_df = pd.concat([table_data_kiez, table_data_berlin], axis=0)
    table_df = table_df.merge(table_data_cluster, how='left', on='year')

    change_perc = [c for c in table_df.columns if '_perc' in c or c in ['upper_quality', 'leisure_part']]
    for col in change_perc:
        table_df[col] = table_df[col] * 100

    for col in table_df.select_dtypes(exclude='object').columns:
        if col == 'cluster':
            table_df[col] = table_df[col].apply(lambda x: '{:.0f}'.format(x))
        elif 'sqm_price_' in col:
            table_df[col] = table_df[col].apply(lambda x: '{:,.2f}'.format(x))
        else:
            # formatting floats (f) below 1000 with one decimal point (.1) and
            # above with thousand separator (,) and no decimal point (.0)
            table_df[col] = table_df[col].apply(lambda x: '{:.1f}'.format(x) if x < 200 else '{:,.0f}'.format(x))

    table_df = table_df \
        .drop(['year'], axis=1) \
        .transpose()\
        .reset_index()

    table_df.columns = ['indicator', 'kiez_table', 'berlin_table']
    table_df['indicator'].replace(features_long, inplace=True)

    return table_df.to_dict('records')


def compute_development_data(kiez, feature):
    data = data_original.copy()
    dev_cols = ['Bezirksname', 'Name', 'year', feature, f'{feature}_cng']

    dev_kiez = data.loc[data.Bezirksname.isin([kiez, 'Berlin']), dev_cols]

    bezirk = dev_kiez.loc[dev_kiez['Bezirksname'] == kiez, 'Name'].unique()[0]
    dev_bezirk = data.loc[data.Name == bezirk, dev_cols].groupby(['Name', 'year'], as_index=False).mean()
    dev_bezirk['Bezirksname'] = bezirk

    dev_data = pd.concat([dev_kiez, dev_bezirk], axis=0, ignore_index=True)

    return dev_data


def compute_clustering_data(n_clusters):
    info = data_clustering[:, :3]
    X = data_clustering[:, 3:]

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    clusters = kmeans.predict(X).reshape(-1, 1)
    cluster_centers = pd.DataFrame(kmeans.fit(principal_components).cluster_centers_)

    clustering_data = pd.DataFrame(data=np.concatenate((info, principal_components, clusters, gentri_factor), axis=1),
                                   columns=['Bezirksname', 'Name', 'year', 'pc_1', 'pc_2', 'cluster', 'factor'])
    clustering_data[['pc_1', 'pc_2', 'factor']] = clustering_data[['pc_1', 'pc_2', 'factor']].astype('float')
    clustering_data['cluster'] = clustering_data['cluster'].astype('int')

    return clustering_data, cluster_centers


# ######################################
# Dash specific part
# ######################################

dropdown_options = {'sqm_price_all': 'Square Meter Price (in EUR)',
                    'total': 'Total Population',
                    'pop_per_sqm': 'Population per km2',
                    'area_sqkm': 'Area (in km2)',
                    'sqm_price_lower': 'Square Meter Price (in EUR) - lower',
                    'sqm_price_upper': 'Square Meter Price (in EUR) - upper',
                    'rent_cold': 'Rent cold (in EUR)',
                    'apt_size': 'Apartment Size (in m2)',
                    'net_income': 'Household Net Income (in EUR)',
                    'upper_quality': 'Percentage Upper Quality Housing',
                    'children_perc': 'Percentage Population < 18y',
                    'millenials_perc': 'Percentage Population Millenials',
                    '65_above_perc': 'Percentage Population > 65y',
                    'total_foreigners_perc': 'Percentage Pop. with Migration History',
                    'leisure_part': 'Percentage Park Area'
                    }

kiez_initial = 'Regierungsviertel'
year_initial = 2019
feature_initial = 'sqm_price_all'
selection_initial = 'Absolute'
clusters_initial = 6

flow_data_initial = compute_flow_data(kiez_initial, year_initial)
sunburst_data_initial = compute_sunburst_data(kiez_initial, year_initial)
development_data_initial = compute_development_data(kiez_initial, feature_initial)
clustering_data_initial, centers_initial = compute_clustering_data(clusters_initial)
geomap_data_initial = compute_geomap_data(year_initial, clustering_data_initial)

table_records_initial = compute_table_data(kiez_initial, year_initial, clustering_data_initial)

figure_sankey_initial = create_figure_sankey(flow_data_initial)
figure_sunburst_initial = create_figure_sunburst(sunburst_data_initial, kiez_initial)
chart_development_initial = create_chart_development(development_data_initial, feature_initial)
chart_increase_initial = create_chart_development(development_data_initial, f'{feature_initial}_cng')
histogram_feature_initial = create_chart_histogram(year_initial, feature_initial, kiez_initial)
histogram_increase_initial = create_chart_histogram(year_initial, f'{feature_initial}_cng', kiez_initial)
figure_cluster_initial = create_figure_clustering(clustering_data_initial, centers_initial, year_initial, kiez_initial)
figure_factor_initial = create_figure_factor(clustering_data_initial, kiez_initial)

kiez_summary_template_md = '''
Displaying information on **{}** for **{}**.
 
Selected Kiez: **{}**.
_Click on the map to change the kiez._
'''
kiez_summary_md = kiez_summary_template_md.format(features_long[feature_initial], year_initial, kiez_initial)

app.layout = html.Div([
    # Stores
    dcc.Store(id='kiez', data=kiez_initial),
    dcc.Store(id='feature_click', data=feature_initial),
    # About the app + logos
    html.Div(className="row", children=[
        html.Div(className='twelve columns', children=[
            html.Div(style={'float': 'right'}, children=[
                html.A(
                    html.Img(
                        src=app.get_asset_url("gentriwatch_logo_v3a.png"),
                        style={'float': 'right', 'height': '150px', 'opacity': '0.5'}
                    ))
            ]),
            html.Div(style={'float': 'left'}, children=[
                html.H2('GentriBoard by GentriWatch'),
                html.H4('Monitoring Gentrification in Berlin Neighborhoods')
            ]),

        ]),
    ]),
    # Control panel
    html.Div(className="row", id='control-panel', children=[
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select Year'),
            dcc.Slider(id='years-slider',
                       value=2019,
                       min=2010, max=2019,
                       marks={i: str(i) for i in range(2010, 2020)},
                       included=False),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select Feature'),
            dcc.Dropdown(id='feature_dropdown',
                         placeholder='Select a Feature',
                         options=[{'label': v, 'value': k} for k, v in dropdown_options.items()],
                         value=feature_initial
                         ),
        ]),
        html.Div(className="four columns pretty_container", children=[
            html.Label('Select Number of Clusters'),
            dcc.Input(id="input_nclusters", type='number',
                      #placeholder='Select Number of Clusters',
                      value=6),
        ]),
    ]),

    # The Visuals
    dcc.Tabs(id='tab', children=[
        dcc.Tab(label='Kiez Info', children=[
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Markdown(id='kiez_summary', children=kiez_summary_md),
                    dcc.Graph(id='geomap_figure',
                              figure=create_figure_geomap(geomap_data_initial, kiez_initial, feature_initial),
                              config={"displayModeBar": False,
                                      #"modeBarButtonsToRemove": ['lasso2d', 'select2d'],
                                      #"displaylogo": False
                                      })
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dash_table.DataTable(id='table', columns=[
                            {'name': ' ', 'id': 'indicator'},
                            {'name': 'Kiez', 'id': 'kiez_table'},
                            {'name': 'Berlin', 'id': 'berlin_table'}],
                        data=table_records_initial,
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'fontWeight': 'bold'},
                        style_cell_conditional=[
                            {'if': {'column_id': 'indicator'},
                             'textAlign': 'left'},
                        ],
                    )
                ])
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    html.H6('Migration Background'),
                    #dcc.RadioItems(id='sankey_selection',
                    #               options=[{'label': i, 'value': i} for i in ['Absolute', 'Percentage']],
                    #               value='Absolute',
                    #               labelStyle={'display': 'inline-block'}
                    #            ),
                    dcc.Graph(id='flow_sankey_figure',
                              figure=figure_sankey_initial,
                              config={"displayModeBar": False})
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    html.H6('Age Distribution'),
                    #dcc.RadioItems(id='sunburst_selection',
                    #               options=[{'label': i, 'value': i} for i in ['Absolute', 'Percentage']],
                    #               value='Absolute',
                    #               labelStyle={'display': 'inline-block'}
                    #            ),
                    dcc.Graph(id='flow_sunburst_figure',
                              figure=figure_sunburst_initial,
                              config={"displayModeBar": False}),
                ]),
            ]),
        ]),
        dcc.Tab(label='Feature Info', children=[
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='feature_dev_chart',
                              figure=chart_development_initial,
                              config={"displayModeBar": False})
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='feature_inc_chart',
                              figure=chart_increase_initial,
                              config={"displayModeBar": False})
                ]),
            ]),
            html.Div(className="row", children=[
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='hist_feature',
                              figure=histogram_feature_initial,
                              config={"displayModeBar": False}
                              )
                ]),
                html.Div(className="fix columns pretty_container", children=[
                    dcc.Graph(id='hist_increase',
                              figure=histogram_increase_initial,
                              config={"displayModeBar": False}
                              )
                ]),
            ]),
        ]),
        dcc.Tab(label='Cluster Info', children=[
            html.Div(className="row", children=[
                html.Div(className="eight columns pretty_container", children=[
                    dcc.Graph(id='cluster_figure',
                              figure=figure_cluster_initial,
                              config={"displayModeBar": False})
                ]),
                html.Div(className="four columns pretty_container", children=[
                    dcc.Graph(id='factor_figure',
                              figure=figure_factor_initial,
                              config={"displayModeBar": False})
                ]),
            ]),
        ]),
    ]),
])


# Geographical map

# Geographical map click on geojson area
@app.callback(Output('kiez', 'data'),
              [Input('geomap_figure', 'clickData'),
               Input('cluster_figure', 'clickData')],
              [State('kiez', 'data')],
              prevent_initial_call=True)
def click_action_kiez(click_data_geomap, click_data_cluster, kiez):

    # What triggered the callback
    trg = dash.callback_context.triggered

    if trg is not None:
        component = trg[0]['prop_id'].split('.')[0]

        if component == 'geomap_figure':
            kiez = trg[0]['value']['points'][0]['customdata'][0]

        if component == 'cluster_figure':
            kiez = trg[0]['value']['points'][0]['hovertext']

    return kiez


# Select feature from clicking on graphs
@app.callback(Output('feature_click', 'data'),
              [Input('flow_sunburst_figure', 'clickData'),
               Input('flow_sankey_figure', 'clickData'),
               Input('table', 'active_cell')],
              [State('feature_click', 'data'),
               State('table', 'data')],
              prevent_initial_call=True)
def click_action_feature(click_data_sunburst, click_data_sunkey, click_data_table,
                         clicked_feature, table_data):
    # What triggered the callback
    trg = dash.callback_context.triggered

    if trg is not None:
        component = trg[0]['prop_id'].split('.')[0]

        if (component == 'flow_sunburst_figure'):  #| (component == 'flow_sankey_figure'):
            feature = trg[0]['value']['points'][0]['label']
            if feature not in data_original.columns:
                feature = features_short[feature]

        if component == 'table':
            table_row = trg[0]['value']['row']
            feature_name = table_data[table_row]['indicator']
            feature = features_short[feature_name]

    return feature


# Geographical select feature from dropdown
@app.callback(Output('feature_dropdown', 'value'),
              [Input('feature_dropdown', 'options'),
               Input('feature_click', 'data')],
              prevent_initial_call=True)
def dropdown_action_feature(selected_feature, clicked_feature):
    if selected_feature != clicked_feature:
        feature = clicked_feature
    return feature


# Geographical map data
@app.callback([Output('geomap_figure', 'figure'),
               Output('kiez_summary', 'children')],
              [Input('years-slider', 'value'),
               Input('kiez', 'data'),
               Input('feature_dropdown', 'value'),
               Input('input_nclusters', 'value')],
              [State('geomap_figure', 'figure')],
              prevent_initial_call=True)
def update_geomap_figure(year, kiez, feature, n_clusters, current_figure):

    zoom = current_figure['layout']['mapbox']['zoom']
    center = current_figure['layout']['mapbox']['center']

    cluster_data, _ = compute_clustering_data(n_clusters)

    df = compute_geomap_data(year, cluster_data)

    #if feature not in df.columns:
    #    feature = features_short[feature]
    fig = create_figure_geomap(df, kiez, feature, zoom=zoom, center=center)

    #if '_perc' in feature and feature not in dropdown_options.keys():
    #    feature_name = feature[:-5]
    #else:
    #    feature_name = feature

    kiez_summary_md = kiez_summary_template_md.format(features_long[feature], year, kiez)

    return fig, kiez_summary_md  # , markdown_text, str(count), "trigger loader"


# Flow section
@app.callback([Output('flow_sankey_figure', 'figure'),
               Output('flow_sunburst_figure', 'figure')],
              [Input('years-slider', 'value'),
               Input('kiez', 'data')],
              prevent_initial_call=True)
def update_flow_figures(year, kiez):
    flow_data = compute_flow_data(kiez, year)
    fig_sankey = create_figure_sankey(flow_data)

    sunburst_data = compute_sunburst_data(kiez, year)
    fig_sunburst = create_figure_sunburst(sunburst_data, kiez)

    return fig_sankey, fig_sunburst


# Table section
@app.callback(Output('table', 'data'),
              [Input('years-slider', 'value'),
               Input('kiez', 'data'),
               Input('input_nclusters', 'value')],
              prevent_initial_call=True)
def update_flow_figures(year, kiez, n_clusters):
    cluster_data, _ = compute_clustering_data(n_clusters)

    table_records = compute_table_data(kiez, year, cluster_data)

    return table_records


# Feature section
@app.callback([Output('feature_dev_chart', 'figure'),
               Output('feature_inc_chart', 'figure'),
               Output('hist_feature', 'figure'),
               Output('hist_increase', 'figure')],
              [Input('kiez', 'data'),
               Input('feature_dropdown', 'value'),
               Input('years-slider', 'value')],
              prevent_initial_call=True)
def update_development_figures(kiez, feature, year):
    if feature in ['cluster', 'factor']:
        feature = 'sqm_price_all'
    #if feature not in data_original.columns:
    #    feature = features_short[feature]
    development_data = compute_development_data(kiez, feature)

    chart_development = create_chart_development(development_data, feature)
    chart_increase = create_chart_development(development_data, f'{feature}_cng')

    histogram_feature = create_chart_histogram(year, feature, kiez)
    histogram_increase = create_chart_histogram(year, f'{feature}_cng', kiez)

    return chart_development, chart_increase, histogram_feature, histogram_increase


# Cluster section
@app.callback([Output('cluster_figure', 'figure'),
               Output('factor_figure', 'figure')],
              [Input('input_nclusters', 'value'),
               Input('kiez', 'data'),
               Input('years-slider', 'value')],
              prevent_initial_call=True)
def update_cluster_figure(n_clusters, kiez, year):
    cluster_data, cluster_centers = compute_clustering_data(n_clusters)

    fig_cluster = create_figure_clustering(cluster_data, cluster_centers, year, kiez)
    fig_factor = create_figure_factor(cluster_data, kiez)

    return fig_cluster, fig_factor


if __name__ == '__main__':
    app.run_server(host='localhost', port=5051, debug=True)
