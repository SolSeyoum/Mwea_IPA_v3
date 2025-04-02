import streamlit as st
import pandas as pd
import altair as alt
import json
from PIL import Image
import json
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import geopandas as gpd

import folium
from folium.features import GeoJsonTooltip
from folium.plugins import Fullscreen
from pyproj import Transformer
from folium.raster_layers import ImageOverlay

import branca
from branca.colormap import LinearColormap, StepColormap

import numpy as np
import xarray as xr
import rasterio
import base64
from PIL import Image
import matplotlib.colors as mcolors
import io



#######################
# dfm.columns = [x.replace('_', ' ') for x in dfm.columns]
logo_wide = r'data/logo_wide.png'
logo_small = r'data/logo_small.png'

IPA_description = {
    "beneficial fraction": ":blue[Beneficial fraction (BF)] is the ratio of the water that is consumed as transpiration\
         compared to overall field water consumption (ETa). ${\\footnotesize BF = T_a/ET_a}$. \
         It is a measure of the efficiency of on farm water and agronomic practices in use of water for crop growth.",
    "crop water deficit": ":blue[crop water deficit (CWD)] is measure of adequacy and calculated as the ration of seasonal\
        evapotranspiration to potential or reference evapotranspiration ${\\footnotesize CWD= ET_a/ET_p}$",
    "relative water deficit": ":blue[relative water deficit (RWD)] is also a measure of adequacy which is 1 minus crop water\
          deficit ${\\footnotesize RWD= 1-ET_a/ET_p}$",
    "total seasonal biomass production": ":blue[total seasonal biomass production (TBP)] is total biomass produced in tons. \
        ${\\footnotesize TBP = (NPP * 22.222) / 1000}$",
    "seasonal yield": ":blue[seasonal yield] is the yield in a season which is crop specific and calculated using \
        the TBP and yield factors such as moisture content, harvest index, light use efficiency correction \
            factor and above ground over total biomass production ratio (AOT) \
                ${\\footnotesize Yiled = TBP*HI*AOT*f_c/(1-MC)}$",
    "crop water productivity": ":blue[crop water productivity (CWP)] is the seasonal yield per the amount of water \
        consumed in ${kg/m^3}$"
}

stat_dict = {'Standard deviation':'std', 'Minimum': 'min', 'Maximum':'max', 'Average':'mean', 'Median':'meadian'}

units = {'beneficial fraction':'-', 'crop water deficit': '-',
    'relative water deficit': '-', 'total seasonal biomass production': 'ton',
    'seasonal yield': 'ton/ha', 'crop water productivity': 'kg/m³'}

# @st.cache_data(ttl=300)
def load_image(image_name: str) -> Image:
    """Displays an image.

    Parameters
    ----------
    image_name : str
        Local path of the image.

    Returns
    -------
    Image
        Image to be displayed.
    """
    return Image.open(image_name)


def logos():
    img_small = load_image(logo_small)
    img_wide = load_image(logo_wide)

    return  img_small, img_wide


#######################
BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        max_width: int = 1100, max_width_100_percent: bool = False,
        padding_top: int = 1, padding_right: int = 1, padding_left: int = 1, padding_bottom: int = 10,
        color: str = COLOR, background_color: str = BACKGROUND_COLOR,
    ):
        # Page configuration
        st.set_page_config(
            page_title="Mwea Irrigation Scheme Irrigation Performance Indicators by Sections Dashboard",
            page_icon="📈🌿",
            layout="wide",
            initial_sidebar_state="expanded")

        alt.themes.enable("dark")

        st.markdown("""
            <style>
            header.stAppHeader {
                background-color: transparent;
            }
            section.stMain .block-container {
                padding-top: 0rem;
                z-index: 1;
            }
            </style>""", unsafe_allow_html=True)

        hide_github_icon = """
            <style>
                .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob, .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137, .viewerBadge_text__1JaDK{ display: none; } #MainMenu{ visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
            </style>
        """
        st.markdown(hide_github_icon, unsafe_allow_html=True)

        #######################

# Load data
@st.cache_data(ttl=300)
def read_df_and_geo():
    dfm = pd.read_csv(r'data/Mwea_IPA_statistics3.csv')
    with open(r'data/Mwea_blocks2.json') as response:
        geo = json.load(response)

    return dfm, geo


def indicator_title(indicator, stat_dict):
    # stat_dict = {'std':'Standard deviation', 'min':'Minimum', 'max':'Maximum', 'mean':'Average', 'median':'Median'}
    lst = indicator.split('_')
    t1 = ' '.join(lst[1:])
    t2 = f"{[k for k, v in stat_dict.items() if v == lst[0]][0]} {t1}" 
    return t1,t2

# merge block polygons to sections
def merge_blocks_to_sections(geo, df_section):
    
    new_features = []
    for i, name in enumerate(df_section.section_name):
        polygons = []
        to_combine = [f for f in geo["features"] if f["properties"]["section_name"]==name]
        # print(name)

        for feat in to_combine:
            lst = feat['geometry']['coordinates'][0]
            if isinstance(lst[0][0], list): # check if the geometry is 2d or 3d list
                lst = [e for sl in lst for e in sl]
            polygon = Polygon([ (coor[0], coor[1]) for coor in  lst ])
            polygons.append(polygon)

        new_geometry = mapping(unary_union(polygons)) # This line merges the polygones
        new_feature = dict(type='Feature', id=i, properties=dict(section_name=name),
                        geometry=dict(type=new_geometry['type'], 
                                        coordinates=new_geometry['coordinates']))
        new_features.append(new_feature)
    sections = dict(type='FeatureCollection', 
                    crs= dict(type='name', properties=dict(name='urn:ogc:def:crs:OGC:1.3:CRS84')), 
                    features=new_features)
    return sections


def make_folium_choropleth(geo, indicator, df, col_name):
    df = df.round(2)
    ylable, text = indicator_title(indicator, stat_dict)
    
    # Convert DataFrame to dictionary for mapping
    data_df = df.set_index(col_name)[indicator]

    # Convert DataFrame to dictionary for fast lookup
    data_dict = data_df.to_dict()

      # Add ESRI aerial imagery tile layer
    esri = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Aerial Imagery",
        overlay=False,
        control=True
    )#.add_to(m)
    

    m = folium.Map(location=[-0.69306, 37.35908], 
                   zoom_start=12, height=300, width=400,
                    tiles=esri,  # Add ESRI arial imagery as default tile layer 
  
    )

    # Add OSM map
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    # # bounds = get_bounds(geo)
    minv = df[indicator].min()
    maxv = df[indicator].max()

    # Define a custom color scale
    colormap = StepColormap(
        ["#ff0000", "#ff4500", "#ff7f50", "#ffb347", "#ffdd44", 
        "#ccff33", "#99ff33", "#66ff33", "#33cc33", "#009933"], 
        vmin=minv, vmax=maxv, caption="colormap"
    )
    # Update geojson for tooltip
    for feature in geo["features"]:
        region_id = feature["properties"].get(col_name)  # Get region ID from GeoJSON
        if region_id in data_dict:  # Check if ID exists in CSV
            feature["properties"][indicator] = data_dict[region_id]
        else:
            feature["properties"][indicator] = None  # Assign None if not found

    if 'block' in df.columns:
        fields = ['section_name', col_name, indicator]
        aliases = ["Section:", "Block:", f"{ylable}:"]
        geo["ch_name"] = "Mwea Blocks"	
    else:
        fields = [col_name, indicator]
        aliases = ["Section:", f"{ylable}:"]
        geo["ch_name"] = "Mwea Sections"
    
    # Add Choropleth layer
    tooltip=folium.GeoJsonTooltip(
            fields=fields,#[col_name, indicator],
            aliases=aliases,#["Section:", f"{ylable}:"],
            localize=True,
            sticky=False,
            labels=True,
            style="""
                background-color: #F0EFEF;
                border: 1px solid black;
                border-radius: 3px;
                box-shadow: 3px;
                font-size: 12rem;
                font-weight: normal;
            """,
            max_width=750,
            html=True  # Enables HTML in the tooltip
        )
    
    choropleth = folium.GeoJson(
        geo,
        name=geo["ch_name"],
        tooltip=tooltip,
        style_function=lambda feature: {
            "fillColor": colormap(data_dict[feature["properties"].get(col_name)]),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        },
    ).add_to(m)

    # Add Click event
    click_marker = folium.Marker(
        location=[0, 0],  # Default position (hidden initially)
        popup="Click on the map",
        icon=folium.Icon(color="red")
    )
    m.add_child(click_marker)
           
    folium.LayerControl().add_to(m)
    folium.plugins.Fullscreen().add_to(m)   
    bounds = choropleth.get_bounds()  # Automatically calculates min/max lat/lon

    # Fit map to bounds
    m.fit_bounds(bounds)
    return m

# histogram plot
def make_alt_chart(df,indicator):
    ylable, text = indicator_title(indicator, stat_dict)
    title = alt.TitleParams(f'Yearly {text} by section', anchor='middle')
    barchart = alt.Chart(df, title=title).mark_bar().encode(
        x=alt.X('section_name:N', axis=None),
        y=alt.Y(f'{indicator}:Q', title=ylable),
        color='section_name:N',
        column='year:N'
    ).properties(width=80, height=120).configure_legend(
        orient='bottom'
    )
    return barchart

def format_number(num):
    return f"{num:.2f}"

# Calculation year-over-year difference in metrix
def calculate_indicator_difference(input_df, indicator, input_year):
  selected_year_data = input_df[input_df['year'] == input_year].reset_index()
  previous_year_data = input_df[input_df['year'] == input_year - 1].reset_index()
  selected_year_data['indicator_difference'] = selected_year_data[indicator].sub(previous_year_data[indicator], fill_value=0)
  return pd.concat([selected_year_data['section_name'], selected_year_data[indicator], selected_year_data.indicator_difference], axis=1).sort_values(by="indicator_difference", ascending=False)


def history_df(df1, df2, idx_col, selected_indicator):
    d2 = df1.pivot(index=idx_col, columns='year', values=selected_indicator).reset_index()
    d3 = df2.groupby(idx_col).agg({selected_indicator:'mean'}).reset_index()
    d4 = d3.merge(d2, on=idx_col, how = 'inner')
    d4[d4.columns[2:]]= d4[d4.columns[2:]].round(2)
    d4['history'] = d4[d4.columns[2:]].values.tolist()
    d4 = d4.drop(columns = d4.columns[2:-1])
    return d4.round(2)

select = alt.selection_point(name="select", on="click")
highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

stroke_width = (
    alt.when(select).then(alt.value(2, empty=False))
    .when(highlight).then(alt.value(1))
    .otherwise(alt.value(0))
)

def alt_bar_chart(df, indicator, col_name, year):
    xlable, text = indicator_title(indicator, stat_dict)
    df = df.round(2)
    indicator_name = indicator.replace("_"," ")
    plot_title = f'{indicator_name.title()} - {str(year)}'
    # x_title = f'{xlable.title()} [{units[xlable]}]'
    # title = f'{indicator_name.title()} - {str(year)}'
    
    if len(col_name.split("_"))>1:
        area_id = col_name.split("_")[0]
    else:
        area_id = col_name
    
    
    row_count = len(df)
    pixel_size = 60 - 2* (row_count-5)
    height = row_count * pixel_size  # 30 pixels per row


    select = alt.selection_point(name="select", on="click")
    highlight = alt.selection_point(name="highlight", on="pointerover", empty=False)

    stroke_width = (
        alt.when(select).then(alt.value(2, empty=False))
        .when(highlight).then(alt.value(1))
        .otherwise(alt.value(0))
    )

    chart = alt.Chart(df).mark_bar().encode(
        y=alt.Y(f'{col_name}:N', sort=alt.EncodingSortField(field="indicator", op="count", order='descending'),title=area_id),  # Rename Y-axis
        x=alt.X(f'{indicator}:Q', title=indicator_name),  # Rename X-axis
        color=alt.Color(f'{indicator}:N', legend=None),  # Remove the legend
        fillOpacity=alt.when(select).then(alt.value(1)).otherwise(alt.value(0.3)),
        strokeWidth=stroke_width,

        tooltip=[
            alt.Tooltip(f'{col_name}:N', title=area_id),  
            alt.Tooltip(f'{indicator}:Q', title=indicator_name, format='.2f'),  # Format Value as decimal with 2 digits
        ],
        

    ).properties(
        height=height, #title=plot_title
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit"  # Ensures it resizes correctly
        ).add_params(select, highlight)

    return chart, plot_title


def move_rows_to_top(df, column, value):
    """Reorder DataFrame rows to bring specified column value to the top."""
    df_top = df[df[column] == value]  # Rows with the specified value
    df_rest = df[df[column] != value]  # All other rows
    return pd.concat([df_top, df_rest], ignore_index=True)  # Merge and reset index


def make_alt_linechart(df, indicator, col_name, year, selected_block):
    ylable, text = indicator_title(indicator, stat_dict)


    df=df.assign(year= pd.to_datetime(df['year'], format='%Y')).round(2)
    indicator_name = indicator.replace("_"," ")
    
    min_value = df[indicator].min()
    max_value = df[indicator].max()
    
    if len(col_name.split("_"))>1:
        area_id = col_name.split("_")[0]
    else:
        area_id = col_name

    plot_title = f'{text.title()} per {area_id} for the past seasons'
    y_title = f'{ylable.title()} [{units[ylable]}]' 

    # title = f'{indicator_name.title()} per {area_id}s for the past seasons'

    if selected_block is not None:
        # df['order'] = ['first' if x== selected_block else 'last' for x in df['block']]
        df_sorted = move_rows_to_top(df, 'block', selected_block).iloc[::-1]
               
        chart = alt.Chart(df_sorted).mark_line(size=3).encode(
            x=alt.X('year:T',title='Year'), 
            y=alt.Y(f'{indicator}:Q',title=y_title, scale=alt.Scale(domain=[min_value, max_value])),
            color=alt.Color(f'{col_name}:N', title = area_id,  legend=alt.Legend(orient="top")),
            opacity=alt.condition(
                alt.datum.block == selected_block,  # Highlight selected category
                alt.value(1),  # Full opacity for selected
                alt.value(0.2)  # Lower opacity for others
            ),

            tooltip=[
                alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('year:T', title='year',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
        #    title=plot_title,
           height=300,
           bounds="flush",  # Ensures title does not affect chart size
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit",  # Ensures it resizes correctly
        )
        
        return chart, plot_title
    else:
        chart = alt.Chart(df).mark_line().encode(
            x=alt.X('year:T',title='Year'),  
            y=alt.Y(f'{indicator}:Q',title=y_title, scale=alt.Scale(domain=[min_value, max_value])),
            color=alt.Color(f'{col_name}:N', title = area_id,  legend=alt.Legend(orient="top")),

            tooltip=[
                alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('year:T', title='year',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
            # title=plot_title,
            height=300, 
            bounds="flush",  # Ensures title does not affect chart size
        ).configure_view(
            continuousWidth=600,  # Default width to avoid shrinking
            continuousHeight=300
        ).configure(
            autosize="fit",  # Ensures it resizes correctly
        )

        return chart, plot_title


#=================================
# Load xarray dataset
@st.cache_data(ttl=300)
def create_base_map(center_lat, center_lon, zoom):
    """Create base map with satellite/aerial imagery as default and fullscreen control"""    
    # Add Satellite/Aerial imagery as default base layer
       # Add ESRI aerial imagery tile layer
    esri = folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Aerial Imagery",
        overlay=False,
        control=True
    )#.add_to(m))
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom,
        tiles=esri,
        control_scale=True
    )
    # Add OpenStreetMap as an alternative base layer
    folium.TileLayer("OpenStreetMap", name="OSM", control=True).add_to(m)
    
    # Add Fullscreen control
    Fullscreen(
        position='topleft',
        title='Fullscreen mode',
        title_cancel='Exit fullscreen mode',
        force_separate_button=True
    ).add_to(m)
    
    return m

def get_value_at_point(da, lat, lon, variable):
    ts = da.sel(lat=lat, lon=lon, method="nearest")
    ts = ts.to_dataframe().loc[:,variable] 
    return ts


def extraxt_ts(da, locations):
    # Extract time series for each location
    time_series = {}
    for idx, (lat, lon) in enumerate(locations):
        ts = da.sel(lat=lat, lon=lon, method="nearest")  # Nearest neighbor selection
        time_series[f'point_{idx+1}'] = ts.to_pandas()  # Convert to Pandas Series for easy manipulation

    # Convert to a DataFrame for better analysis
    df = pd.DataFrame(time_series).reset_index()
    return df

def alt_line_chart(df, indicator):
    # df2=df.assign(time= pd.to_datetime(df['time']).dt.year).dropna(axis=1, how='all').round(2)
    df2=df.assign(time= pd.to_datetime(df['time'])).dropna(axis=1, how='all').round(2)
    indicator_name = indicator.replace("_"," ")
    plot_title = f'{indicator_name.title()} for the pixels over the seasons'
    y_title = f'{indicator_name.title()} [{units[indicator_name]}]' 
    data = df2.melt('time')
    minv = data['value'].min()
    maxv = data['value'].max()
    chart = alt.Chart(data).mark_line().encode(
            x=alt.X('time:T',title='Year'),  
            y=alt.Y(f'value:Q', title=y_title, scale=alt.Scale(domain=[minv*0.9, maxv*1.1])),
            color=alt.Color(f'variable:N',  title='Point', legend=alt.Legend(orient="right")),

            tooltip=[
                # alt.Tooltip(f'{col_name}:N', title=area_id),
                alt.Tooltip('year:T', title='year',format='%Y'),    
                alt.Tooltip(f'{indicator}:Q', title=indicator, format='.2f'),  # Format Value as decimal with 2 digits
            ]
        ).properties(
            width=700, height=300, title=plot_title
        )
    
    chart = chart.configure_title(    # <----- this is the only difference
        fontSize=20,
        font='Courier',
        anchor='start',
        color='gray'
    )
    
    return chart


@st.cache_data
def read_dataset(ds_path):
    with xr.open_dataset(ds_path) as dataset:  
        # data = dataset.beneficial_fraction[0].values
        transform = dataset.rio.transform()
        crs = dataset.rio.crs
        nodata = -9999 #dataset.nodata
        bd = dataset.rio.bounds()
        bounds = rasterio.coords.BoundingBox(bd[0], bd[1], bd[2], bd[3])
    
    return dataset, transform, crs, nodata, bounds

@st.cache_data
def get_stats(_data):
      # Compute spatial statistics
    _data = _data.where(_data>0, np.nan)
    stats = {
        'Minimum': _data.min(dim=['lat', 'lon']),
        'Maximum': _data.max(dim=['lat', 'lon']),
        'Mean': _data.mean(dim=['lat', 'lon']),
        'Median': _data.median(dim=['lat', 'lon']),
        'St. deviation': _data.std(dim=['lat', 'lon']),
        "25% quantile": _data.quantile(0.25, dim=['lat', 'lon'], method='linear')
                        .drop_vars('quantile'),
        "75% quantile": _data.quantile(0.75, dim=['lat', 'lon'], method='linear')
                        .drop_vars('quantile'),
    }

    # pd.DataFrame.from_dict(d)
    df_stat = pd.DataFrame.from_dict({k: v.values.item() for k, v in stats.items()}, 
                                    orient='index', columns = ['Values']).round(2)
    df_stat.index.names = ['Stats']
    return df_stat


# Efficient function to get image data for overlay
def get_image_from_ds(data, minv, maxv, nodata, colors):

    try:
        data = np.nan_to_num(data, nan=nodata)
        data = np.flip(data, 0).astype(float)
        
        # Normalize and apply colormap
        norm = mcolors.Normalize(vmin=minv, vmax=maxv)
        cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=100)
        colored_data = cmap(norm(data))
        
        # Set alpha channel for no-data values
        colored_data[..., 3] = np.where(data == nodata, 0, 0.9)
        
        # Convert to PIL image and then to base64
        img = Image.fromarray((colored_data * 255).astype(np.uint8))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return base64.b64encode(img_bytes.getvalue()).decode()
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")

    
# Function to extract time series in a vectorized way
def extract_time_series(da, locations):
    lats, lons = zip(*locations)
    ts = da.sel(lat=list(lats), lon=list(lons), method="nearest")
    return ts.to_dataframe().reset_index()

# Efficient Folium Map Initialization
def create_folium_map( data, geo, bounds, crs, variable):
    # Calculate map center
    left, bottom, right, top = bounds
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    left, bottom = transformer.transform(left, bottom)
    right, top = transformer.transform(right, top)

    # Initialize map
    m = create_base_map((bottom + top) / 2, (left + right) / 2, 12)

    minv = data.min()
    maxv = data.max()

    # colors = ['red', 'orange', 'gold', 'yellow', 'greenyellow', 'lawngreen', 'green']
    colors = ['red', 'yellow',  'green']
     # Add the color map legend
    colormap = LinearColormap(colors=colors, vmin=minv, vmax=maxv)
    colormap.add_to(m)
    colormap.caption = f"{variable} Values"
    
    # Prepare image overlay
    img_base64 = get_image_from_ds(data, minv, maxv, -9999, colors)
    ImageOverlay(
        name=f"{variable.replace('_', ' ')}".title(),
        image=f"data:image/png;base64,{img_base64}",
        bounds=[[bottom, left], [top, right]],
        opacity=0.9,
    ).add_to(m)

        # Add the polygons
    geo_layer = folium.GeoJson(
        geo,
        name="irrigation divisions",
        style_function=lambda feature: {
            'fillColor': '#00000000', 
            'color': 'black',
            "weight": 0.5,
        },
    ).add_to(m)

    tooltip_choropleth = GeoJsonTooltip(
            fields=['section_name', 'block'],
            aliases=["Section: ", "Block: "],
            localize=True,
            sticky=False,
            labels=True,
            smooth_factor=0,
            style="""
                background-color: #F0EFEF;
                border: 1px solid black;
                border-radius: 3px;
                box-shadow: 3px;
                font-size: 12px;
                font-weight: normal;
            """,
            max_width=750,
        )
    geo_layer.add_child(tooltip_choropleth)

    m.fit_bounds(geo_layer.get_bounds())  # 2. fit the map to GeoJSON layer
    # Add Click event
    click_marker = folium.Marker(
        location=[0, 0],  # Default position (hidden initially)
        popup="Click on the map",
        icon=folium.Icon(color="red")
    )
    m.add_child(click_marker)

    # JavaScript for click event
    m.add_child(folium.LatLngPopup())  # Shows lat/lon on click

    # Add markers for all clicked locations
    for idx, (lat, lon) in enumerate(st.session_state.clicked_locations):
        folium.Marker([lat, lon], popup=f"Point {idx + 1}: lat: {lat:.4f}, lon: {lon:.4f}", 
                    icon=folium.Icon(color="red")).add_to(m)
        
        # Label marker (positioned slightly above the main marker)
        folium.Marker(
            location=[lat, lon],  # Slightly shift the label upwards
            icon=folium.DivIcon(
                icon_size=(50,50),
                icon_anchor=(3,17),
                html=f'<div style="font-size: 24ptpx;font-weight: bold; color: white;">{idx + 1}</div>'
            ),
            zIndexOffset=1000 
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    return m

@st.cache_data
def get_gdf_from_json(geo):
     return gpd.GeoDataFrame.from_features(geo['features'])
