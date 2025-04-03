#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import json
from shapely.geometry import shape, Point
import geopandas as gpd

# from shapely.ops import unary_union


import folium
from streamlit_folium import st_folium
from branca.colormap import LinearColormap
from PIL import Image
#######################
from util import common2 as cm
# import common
# from common import set_page_container_style



cm.set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0
)

dfm, geo = cm.read_df_and_geo()
logo_small, logo_wide = cm.logos()

col = st.columns((5.5, 2.5), gap='small')
# Add a Reset Button as a Folium Marker
def reset_map():
    """Resets the selected polygon state."""
    st.session_state.selected_polygon = None
    st.rerun()

# Sidebar
with st.sidebar:

    st.logo(logo_wide, size="large", link='https://www.un-ihe.org/', icon_image=logo_small)
    st.title('Mwea Irrigation Performance Indicators')

    # section_list =  list(set(dfm.section_name.to_list()))
    # section_list.append('All')
    # selected_section = st.selectbox('Select a section to zoom into:', section_list, index = 5)

    year_list = list(dfm.year.unique())[::-1]
    ll = list(dfm.columns.unique())[3:][::-1]
    indicator_lst = [' '.join(l.split('_')[1:]) for l in ll]
    indicator_lst = list(set(indicator_lst))
    indicator_lst = [x for x in indicator_lst if x != ""]
    
    selected_year = st.selectbox('Select a year', year_list, index = 0, 
                                 help="Choose the Year/Season to visualize")
    indicator = st.selectbox('Select an indicator', indicator_lst, index = 0,
                             help="Choose the IPA indicator type to visualize")
    selected_stat = st.selectbox('Select a statistics', cm.stat_dict.keys(), index = 3, 
                                 help="Choose the statistics to visualize")
    selected_stat_abbr = cm.stat_dict[selected_stat]
    selected_indicator = f'{selected_stat_abbr}_{indicator.replace(' ', '_')}'
       
    st.write(f'{cm.IPA_description[indicator]}')
    stat_description = cm.stat_dict[selected_stat]
   

    df_selected = dfm[dfm.year == selected_year][['section_name', selected_indicator]]
    df_selected_sorted = df_selected.sort_values(by=selected_indicator, ascending=False)

    #aggregate by section
    df_section=df_selected_sorted.groupby('section_name').agg({selected_indicator:'mean'})#.rename(columns=d)
    df_section = df_section.sort_values(by=selected_indicator, ascending=False).reset_index()

    st.markdown("---")
    with st.expander("ℹ️ About the Indicator Map"):
        st.markdown("""
        This Indicator Map provides view of the Irrigation Performance Indicators (IPA) for Mwea Irrigation Scheme.
        - IPAs are calculated using data from: [FAO WaPOR data](https://www.fao.org/in-action/remote-sensing-for-water-productivity/wapor-data/en).
        - :orange[**Indicator Map**]: Shows the irrigation schemes section or blocks values for the selected indicator and selected statistics.      
        - Year/Season, and indicator type and statistic type can be selected to view the indicator selected by year/season and by statistics type.
        - 📊 :orange[**Bar Chart**]: on the right side shows the indicator for the selected year for the section or the block depending on which view is on the indicator map sorted by the selected indicator.                             
        - 📈 :orange[**Line Chart**]: or the timeseries plot below the map shows the trend over the years (seasons) for the selected indicator.

        """)


#######################
# Dashboard Main Panel

df_map = pd.DataFrame()
df_chart = pd.DataFrame()
col_name = ''
# selected_block = None
units = cm.units

# geopandas dataframe of the geo (AIO)
gdf = gpd.GeoDataFrame.from_features(geo['features'])

# Initialize session state
if "selected_polygon" not in st.session_state:
    st.session_state.selected_polygon = None
if "selected_block" not in st.session_state:
    st.session_state.selected_block = None

# Filter data based on selection
if st.session_state.selected_polygon is not None:
    # Add a reset button to Streamlit UI (works better than Folium custom HTML)
    selected_poly = st.session_state.selected_polygon
    filtered = [sgeo for sgeo in geo["features"] if sgeo['properties']['section_name'] in selected_poly]
    
    filtered_geojson = {
        "type": "FeatureCollection",
        'name': 'test',#geo['name'],
        'crs': geo['crs'],
        "features": filtered
    }
    
    df_block = dfm[dfm.year == selected_year][['section_name', "block", selected_indicator]]
    df_block = df_block.sort_values(by=selected_indicator, ascending=False)
    df_block_section = df_block.loc[df_block["section_name"]==selected_poly]
    col_name = df_block_section.columns[1]
    geo2plot = filtered_geojson
    df_map = df_block_section
    dfm_var = dfm[['year','section_name', 'block',selected_indicator]]
    df_chart = dfm_var.loc[dfm_var["section_name"]==selected_poly]
else:
    col_name = df_section.columns[0]
    geo2plot = cm.merge_blocks_to_sections(geo, df_section)
    df_map = df_section

    dfm_var = dfm[['year','section_name', selected_indicator]].groupby(['year','section_name'])
    df_chart = dfm_var.agg({selected_indicator:'mean'}).reset_index()

    
# if(selected_section == 'All'):
# st.write(st.session_state.selected_block)
choropleth = cm.make_folium_choropleth(geo2plot,selected_indicator, df_map, col_name)
line_chart, title = cm.make_alt_linechart(df_chart, selected_indicator, col_name, 
                                   selected_year, st.session_state.selected_block)
title = f'<p style="font:Courier; color:gray; font-size: 20px;">{title}</p>'

with col[0]:
    # st.markdown('#####        Indicator Map')
    # st.markdown("<h4 style='text-align: center; color: white;'>Indicator Map</h4>", unsafe_allow_html=True)

    left, right = st.columns([0.9, 0.1])
    # left, right = st.columns((6, 2), gap='medium')
    left.subheader("Indicator Map")

    with right:
        st.markdown("<br><br>", unsafe_allow_html=True) 
        if st.session_state.selected_polygon is not None:
            if st.button("🔄 Reset Map"):
                st.session_state.selected_block = None
                reset_map()
    with left:
        map_data = st_folium(choropleth,  height=450, use_container_width=True)
    
        # st.write(map_data)
        st.write("")  
        st.markdown(title, unsafe_allow_html=True)  
        st.altair_chart(line_chart, use_container_width=True)
        
        if map_data and "last_clicked" in map_data and map_data["last_clicked"] != None :
            # Find the clicked polygon
            clicked_point = Point(map_data["last_clicked"]["lng"], map_data["last_clicked"]["lat"])
            matching_polygon = gdf[gdf.contains(clicked_point)]
            if not matching_polygon.empty:
                clicked_section = matching_polygon.iloc[0]["section_name"]
                clicked_block = matching_polygon.iloc[0]["block"]
                if (st.session_state.selected_polygon != clicked_section) or (st.session_state.selected_block != clicked_block):
                        st.session_state.selected_polygon = clicked_section
                        st.session_state.selected_block = clicked_block
                        st.rerun()

        # clicked_coords = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
        # for idx, row in gdf.iterrows():
        #     if row.geometry.contains(gpd.points_from_xy([clicked_coords[1]], [clicked_coords[0]])[0]):
        #         if (st.session_state.selected_polygon != row["section_name"]) or (st.session_state.selected_block != row["block"]):
        #             st.session_state.selected_polygon = row["section_name"]
        #             st.session_state.selected_block = row["block"]                        
        #             st.rerun()

            
    

with col[1]:
    # st.markdown('###### Bar chart of the selected indicator')
    st.markdown("<br><br>", unsafe_allow_html=True)
    chart, title  = cm.alt_bar_chart(df_map, selected_indicator, col_name, selected_year)
    title = f'<p style="font:Courier; color:gray; font-size: 20px;">{title}</p>'
    st.markdown(title, unsafe_allow_html=True)

    st.altair_chart(chart, use_container_width=True)

    # with st.expander('About', expanded=False):
    #         st.write('''
    #             - Irrigation Performance Indicators are calculated from data: [FAO WaPOR data](https://www.fao.org/in-action/remote-sensing-for-water-productivity/wapor-data/en).
    #             - :orange[**Indicator Map**]: Shows the irrigation schemes section or blocks values for the selected indicator and selected statistics.
    #             - :orange[**Bar Chart**]: shows the selected  indicator for the selected year for the section or the block depending on which view is on the indicator map.
    #             - :orange[**Line Chart**]: shows the trend over the years (seasons) for the selected  indicator. The clicked block is drawn in red and the rest in grey color.
    #             ''')
