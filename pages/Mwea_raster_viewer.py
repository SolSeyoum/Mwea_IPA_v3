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

try:
    # st.markdown(f"### Mwea IPA Raster Viewer")

    variable = indicator.replace(' ', '_')
    slected_time = f'{selected_year}-12-31'
    ds, transform, crs, nodata, bounds = cm.read_dataset(ipa_ds_path)

    # data =  ds.sel(time=slected_time)[variable]
    data_var =  ds[variable]
    data =  data_var.sel(time=slected_time)
    df_stats = cm.get_stats(data)


    # Initialize session state
    if "clicked_locations" not in st.session_state:
        st.session_state.clicked_locations = []  # Store multiple clicks

    col = st.columns((5.5, 2.5), gap='small')
    with col[0]:
        st.markdown(f"### Mwea IPA Raster Viewer")
        # with st.spinner("Loading and processing data..."):
        #         
        # Process clicked locations and display map
        if map_data := st_folium(cm.create_folium_map(data, geo, bounds, crs, variable), height=500, width=700):

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
            locations = st.session_state.clicked_locations
            if locations:
                data_all_points = cm.extraxt_ts(data_var, locations)

                if(len(data_all_points) > 0):
                    chart = cm.alt_line_chart(data_all_points, variable)
                    st.altair_chart(chart, use_container_width=True)
    with col[1]:
            st.write('')
            st.markdown(f"##### :blue[Stats of {indicator} [{cm.units[indicator]}]]")
            st.dataframe(df_stats, use_container_width=True)
            
except Exception as e:
        st.error(f"An error occurred while processing the data: {str(e)}")


