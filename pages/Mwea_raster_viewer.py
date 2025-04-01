import streamlit as st
import folium
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from streamlit_folium import folium_static, st_folium
from folium.raster_layers import ImageOverlay

from folium.features import GeoJsonTooltip
from shapely.geometry import Point
import altair as alt

# import base64
from PIL import Image
from branca.colormap import LinearColormap
import matplotlib.colors as mcolors
# import io
from pyproj import Transformer
import geopandas as gpd
import json
from util import common2 as cm



cm.set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0
)

dfm, geo = cm.read_df_and_geo()
logo_small, logo_wide = cm.logos()

ipa_ds_path = r"data/Mwea_ipa_results.nc"
shp_path = r"data/Mwea_blocks2.json"


with st.sidebar:

    st.logo(logo_wide, size="large", link='https://www.un-ihe.org/', icon_image=logo_small)
    st.title('Mwea Irrigation Performance Indicators')

    year_list = list(dfm.year.unique())[::-1]
    ll = list(dfm.columns.unique())[3:][::-1]
    indicator_lst = [' '.join(l.split('_')[1:]) for l in ll]
    indicator_lst = list(set(indicator_lst))
    indicator_lst = [x for x in indicator_lst if x != ""]
    
    selected_year = st.selectbox('Select a year', year_list)
    indicator = st.selectbox('Select an indicator', indicator_lst, index = 0)
# Streamlit UI
# st.title("### Mwea IPA Raster Viewer")
try:
    # st.markdown(f"### Mwea IPA Raster Viewer")

    variable = indicator.replace(' ', '_')

    # Initialize session state
    if "clicked_locations" not in st.session_state:
        st.session_state.clicked_locations = []  # Store multiple clicks

    slected_time = f'{selected_year}-12-31'
    ds, transform, crs, nodata, bounds = cm.read_dataset(ipa_ds_path)

    # data =  ds.sel(time=slected_time)[variable]
    data_var =  ds[variable]
    data =  data_var.sel(time=slected_time)
    # st.write(bounds)
    # Transform bounds to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    left, bottom = transformer.transform(bounds.left, bounds.bottom)
    right, top = transformer.transform(bounds.right, bounds.top)

    # Generate map
    m = cm.create_base_map((bottom + top) / 2, (left + right) / 2, 12)
    minv = data.min()
    maxv = data.max()
    nodata = -9999
    # colors = ['red', 'orange', 'gold', 'yellow', 'greenyellow', 'lawngreen', 'green']
    colors = ['red', 'yellow',  'green']
    img_base64 = cm.get_image_from_ds(data, minv, maxv, nodata, colors)

    # Add the color map legend
    colormap = LinearColormap(colors=colors, vmin=minv, vmax=maxv)
    colormap.add_to(m)
    colormap.caption = f"{variable} Values"

    # Convert numpy array to image

    # Add Image Overlay
    ImageOverlay(
        name=f"{variable.replace('_', ' ')}".title(),
        # image="Xarray Layer",
        image=f"data:image/png;base64,{img_base64}",
        # bounds=bounds,
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

    df_stats = cm.get_stats(data)

    col = st.columns((5.5, 2.5), gap='small')
    with col[0]:
        st.markdown(f"### Mwea IPA Raster Viewer")
        # with st.spinner("Loading and processing data..."):
        #         
        # # Streamlit Folium map
        map_data = st_folium(m, height=500, use_container_width=True)

        gdf = gpd.GeoDataFrame.from_features(geo['features'])

        # Process Click Event
        if map_data and "last_clicked" in map_data and map_data["last_clicked"] != None :
            lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

            clicked_coords = Point(lon, lat,)
            
            if gdf.contains(clicked_coords).any():
                # Avoid duplicates
                if (lat, lon) not in st.session_state.clicked_locations:
                    st.session_state.clicked_locations.append((lat, lon))
                    st.rerun()  # Rerun only when a new point is added
            else:
                st.write("Please click within the raster layer.")

        # **Display all extracted values**
        data_all_points = cm.extraxt_ts(data_var, st.session_state.clicked_locations)

        if(len(data_all_points) > 0):
            chart = cm.alt_line_chart(data_all_points, variable)
            st.altair_chart(chart, use_container_width=True)

    with col[1]:
            st.write('')
            st.markdown(f"##### :blue[Stats of {indicator} [{cm.units[indicator]}]]",
                        unsafe_allow_html=True)
            st.dataframe(df_stats, use_container_width=True)
            
except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")


