import streamlit as st
import matplotlib.cm as cm
from streamlit_folium import st_folium
from shapely.geometry import Point

from util import common2 as cm


cm.set_page_container_style(
        max_width = 1100, max_width_100_percent = True,
        padding_top = 0, padding_right = 0, padding_left = 0, padding_bottom = 0
)

dfm, geo = cm.read_df_and_geo()
logo_small, logo_wide = cm.logos()
gdf = cm.get_gdf_from_json(geo)
ipa_ds_path = r"data/Mwea_ipa_results.nc"

# Initialize session state
session_state_defaults = {
    'last_clicked': None,
    'clicked_locations': [],
    'time_series_generated': False,
    'button_clicked' : False
}
for key, default_value in session_state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value



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

    st.markdown("---")
    with st.expander("ℹ️ About the raster viewer"):
        st.markdown("""
        This viewer provides raster view of the Irrigation Performance Indicators.
        
        - Year/Season and indicators can be selected to view the raster for year/season and indicator selected.
        - 📊 The dataframe on the right side provides statistic of the selected raster                             
        - 📈 You can click points (as many points as needed) on the raster and generate a time series plot of the points.

        """)

try:
    # st.markdown(f"### Mwea IPA Raster Viewer")

    variable = indicator.replace(' ', '_')
    slected_time = f'{selected_year}-12-31'
    ds, transform, crs, nodata, bounds = cm.read_dataset(ipa_ds_path)
    data_var =  ds[variable]
    data =  data_var.sel(time=slected_time)
    df_stats = cm.get_stats(data)


    col = st.columns((5.5, 2.5), gap='small')
    with col[0]:
        st.markdown(f"### Mwea IPA Raster Viewer - {selected_year}")
        # with st.spinner("Loading and processing data..."):
               
        if map_data := st_folium(cm.create_folium_map(data, geo, bounds, crs, variable), 
                                 height=500, width=None,
                                 returned_objects=["last_clicked"]):

            # Process Click Event
            if map_data["last_clicked"] and map_data["last_clicked"] != st.session_state.last_clicked:
                lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

                clicked_coords = Point(lon, lat,)
                
                if gdf.contains(clicked_coords).any():
                    # Avoid duplicates
                    if (lat, lon) not in st.session_state.clicked_locations:
                        st.session_state.last_clicked = map_data["last_clicked"]
                        st.session_state.clicked_locations.append((lat, lon))
                        st.rerun()  # Rerun only when a new point is added
                        
                else:
                    st.write("Please click within the raster layer.")

            if st.session_state.clicked_locations and not st.session_state.button_clicked:

                st.markdown("---")
                if st.button("📈 Generate Time Series"):
                    st.session_state.time_series_generated = True
                    st.session_state.button_clicked = True
                    st.rerun()

    with col[1]:
            st.write('')
            title = f'<p style="font:Courier; color:gray; font-size: 20px;">Stats of {indicator} [{cm.units[indicator]}] - {selected_year}</p>'
            st.markdown(title, unsafe_allow_html=True)
            st.dataframe(df_stats, use_container_width=True)

    # **Display all extracted values**
    locations = st.session_state.clicked_locations
    if st.session_state.time_series_generated:
        data_all_points = cm.extraxt_ts(data_var, locations)
        

        if(len(data_all_points) > 0):
            chart = cm.alt_line_chart(data_all_points, variable)
            st.altair_chart(chart, use_container_width=True)

except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")
