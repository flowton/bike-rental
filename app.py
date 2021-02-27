import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pydeck as pdk
from sklearn.cluster import KMeans
#from matplotlib_venn import venn2, venn2_circles
import matplotlib.pyplot as plt

st.markdown("""
# What is Data Science?
## *Explained with bikes in Gothenburg*
""")

st.markdown("""
### 1. :question: **Ask the right questions**
> What stations exist for bike rental in Gothenburg? Are they all used the same?
""")


st.markdown("""
### 2. :keycap_ten: **Prepare the data**
> What stations exist for bike rental in Gothenburg? Are they all used the same?
* Raw data is one row for **each bike ride**
* => Goal is to have data per **station** instead
""")

#-------Import df-----------
@st.cache(suppress_st_warning=True)
def read_data():
    gif_runner = st.image('loading.gif')
    df = pd.read_excel('bike-rental-gbg-travel-2020.xlsx')
    #df = pd.read_csv('bike-rental-gbg-travel-2020.csv', encoding = 'latin-1')
    df = df.rename(columns = {"Start time": "start_time", "End time": "end_time", "Duration" : "duration",
                              "Start station number" : "start_station_num", "Rental place" : "start_station_name",
                              "End station number" : "end_station_num", "Return place" : "end_station_name",
                              "Bike number" : "bike_number"})
    gif_runner.empty()
    return df

df = read_data()
st.dataframe(df.head())

##------------Create DF plot------------

df_plot = df.copy()
df_plot['day'] = df['start_time'].dt.day_name().astype(pd.api.types.CategoricalDtype(categories = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday']))
df_plot['month'] = df['start_time'].dt.month_name().astype(pd.api.types.CategoricalDtype(categories = ['June', 'July', 'August', 'September', 'October', 'November', 'December']))
df_plot['start_hour'] = df['start_time'].dt.hour
df_plot['end_hour'] = df['end_time'].dt.hour
# Simply get the count
month_day = pd.crosstab(df_plot['month'], df_plot['day'])
day_hour = pd.crosstab(df_plot['start_hour'], df_plot['day'])
month_hour = pd.crosstab(df_plot['start_hour'], df_plot['month'])
#tab = pd.crosstab(df_plot['month'], df_plot['day']).reindex(days_order, axis = 1, months_order, )
# Get the duration. Be careful that some might have very high value!
heatmap_data = pd.pivot_table(df_plot[['month', 'day', 'duration']], values = 'duration', index = ['month'], columns = 'day')


##------------------Filter Data-------------------

run_data_process = st.checkbox(
    'Run data processing'
)

if run_data_process:

    st.markdown("""
    * Data displayed on **station level**, grouped by trips for different **times of the day**
    """)

    def filter_data_2(h_start = 0, h_end = 23, sort_order = 'diff_percentage', min_trips = 0, num_results = 3,input_df = df_plot):
        data = input_df[(input_df['start_hour'] >= h_start) & (input_df['end_hour'] <= h_end)]
        start_station = data.groupby('start_station_name')['start_station_name'].count()
        start_station = start_station.where(start_station > min_trips).dropna()
        end_station = data.groupby('end_station_name')['end_station_name'].count()
        end_station = end_station.where(end_station > min_trips).dropna()
        start_end = pd.concat([start_station, end_station], axis = 1)
        start_end = start_end.rename(columns = {'start_station_name': 'total_starts', 'end_station_name' : 'total_ends'})
        start_end['total_diff'] = start_end['total_starts'] - start_end['total_ends']
        start_end['diff_percentage'] = start_end['total_diff']/(start_end['total_starts'] + start_end['total_ends'])

        # TODO Should NA be dropped? Or replace it with zero? Need domain expertise resarch
        popular_start =  start_end.sort_values(by=sort_order).dropna().head(num_results)
        popular_end = start_end.sort_values(by=sort_order).dropna().tail(num_results)
        popular_stations = list(popular_start.index) + list(popular_end.index)

        return start_end, popular_stations

    data, stations = filter_data_2(h_start = 6, h_end = 9, min_trips = 100, num_results= 10, sort_order= 'total_diff')


    ##----------Grouping the data-----------------

    # TODO - Make this nicer so not having strings, but its hard when there can be any amount
    # of possible hours. Look at size of cutoff_points, build another function thar groups by this condition
    def grouping_starthour_adv(df, cuttof_points, type = 'proportion', keep_total_count = False):
        cutof_dict = {}
        for i, cutof in enumerate(cuttof_points):
            if cutof == min(cuttof_points):
                id = str(0)+'-'+str(cutof)
                cutof_dict[i] = tuple([id ,0, cutof])
            else:
                id = str(cutof_dict[i-1][-1]+1) + '-' + str(cutof)
                cutof_dict[i] = tuple([id ,cutof_dict[i-1][-1]+1,cutof])
        end_key = len(cutof_dict)
        id = str(str((max(cuttof_points)+1)) + '-' + '23')
        cutof_dict[end_key] = tuple([id,max(cuttof_points)+1, 23])

        string = '['
        for key, value in cutof_dict.items():
            string = string + '(' + '\'' + str(value[0]) + '\'' + ',' + 'lambda x: sum((x >=' + str(value[1]) + ')' + ' & ' + '(x <= ' + str(value[2])  + '))),' +'\n'
        string =  string[:-2] # remove the last comma
        string = string + ',' + '\'count\'' + ']'

        temp = df.groupby(['start_station_name'])['start_hour'].agg(
            eval(string)
         )

        if type == 'proportion':
            total_count = temp['count']
            res = temp.iloc[:,:-1].div(total_count, axis = 0) #divide all col in row with total row count
            if keep_total_count:
                res['total_count'] = total_count
        return res.reset_index()

    filtered_data = df_plot[df_plot['start_station_name'].isin(stations)]
    cutoff_points = [6,12,16,19]
    grouped_data = grouping_starthour_adv(filtered_data,cutoff_points)

    st.dataframe(grouped_data.head())

st.markdown("""
### 3. :computer: **Use machine learning**
> What patterns of the stations can the computer find when using machine learning?
""")

##---------Clustering the data------------


def find_station_label(data, model, k):
    labeldict = {}
    for i in range(k):
        labeldict[i] = []
    for index, label in enumerate(kmeans_model.labels_):
        station = grouped_data.iloc[index,0]
        labeldict[label].append(station)
    return labeldict

run_ml = st.checkbox(
    'Run Machine Learning'
)
if run_ml:
    k = 4
    kmeans_input_data = grouped_data.drop(columns=['start_station_name', '0-6', '20-23'])
    kmeans_model = KMeans(n_clusters=k).fit(kmeans_input_data)  # Drop station as it's not a feature
    st.write('Cluster centers: ')
    st.write(kmeans_model.cluster_centers_)
    st.write('Inertia: ')
    st.write(kmeans_model.inertia_)
    t = find_station_label(grouped_data, kmeans_model, k)



##------------Visualizing the data--------------

st.markdown("""
### 4. :bar_chart: **Visualize results**
> How can we clearly communicate our findings?
""")

run_map_viz = st.checkbox(
    'Run map visualization'
)
if run_map_viz:
    aug_stations = pd.read_excel('bike-rental-gbg-setpoints-stations - augmented with coordinates.xlsx')

    coordinates_per_cluster = []
    for key in t:
        lat = []
        lon = []
        for i in range(len(t[key])):
            temporary = aug_stations[aug_stations['Place name'] == t[key][i]].iloc[:, -2:]

            lat.append(temporary.iloc[0, 0])
            lon.append(temporary.iloc[0, 1])

        station_coordinates = {'name': t[key], 'lat': lat, 'lon': lon}
        coordinates_per_cluster.append(station_coordinates)

    def visualize_single_map(data, color='[240,30,70, 180]'):
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=57.7068,
                longitude=11.965,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    get_position='[lon, lat]',
                    get_color=color,
                    get_radius=100,
                ),
            ],
        ))


    def visualize_multiple_map(all_data, num_maps, color='[240,30,70, 180]'):
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=57.7068,
                longitude=11.965,
                zoom=11,
                pitch=50,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=data,
                    get_position='[lon, lat]',
                    get_color=color,
                    get_radius=100,
                ),
            ],
        ))

    colors = ['[240,30,70, 180]', '[0,250,0, 180]', '[0,0,250, 180]', '[100,100,100, 180]', '[60,120,180, 180]']
    for i, dict in enumerate(coordinates_per_cluster):
        df_current_cluster = pd.DataFrame(dict)
        st.write('Cluster: ' + str(i) + '\n')
        st.write(df_current_cluster['name'])
        visualize_single_map(df_current_cluster, colors[
            i % len(colors)])  # Visualize map, enforce that color indice is within size of available colors.

run_bar_viz = st.checkbox(
    'Run bar Visualization'
)
if run_bar_viz:
    def stacked_bar_clustering_viz(labeldict, data,model_data, points_of_cutoff, cluster_centers, max_results = 1000):
        def find_stacked_bottom(n, row):
            bottom = 0
            for i in range(0,n):
                bottom = bottom + row.iloc[:, (i+1)]
            return bottom

        all_cutoff_pts = points_of_cutoff.copy()
        all_cutoff_pts = [0] + all_cutoff_pts + [23]
        colors = ['whitesmoke', 'lightskyblue', 'cornflowerblue', 'dodgerblue', 'whitesmoke']
        print(all_cutoff_pts)
        fig, ax = plt.subplots()
        for key, value in enumerate(labeldict.items()):
            for i, station in enumerate(value[1]):
                row = data[data['start_station_name'] == station]
                for n in range(0,len(all_cutoff_pts)-1):
                    bt = find_stacked_bottom(n,row)
                    plt.barh(station, row.iloc[:,(n+1)], left = bt, color = colors[n], label = station)
            center = [ "{:.0%}".format(x) for x in cluster_centers[key]]
            combo= ['Hour ' + model_data.columns[x] + ': ' +  center[x] for x in range(len(cluster_centers[0]))]
            pretty_print = '\n'.join([str(elem) for elem in combo])
            #print(pretty_print)
            plt.title('Cluster Centers:  \n' + str(pretty_print))
            plt.show()
            st.pyplot(plt.gcf())


    stacked_bar_clustering_viz(t, grouped_data, kmeans_input_data, cutoff_points, kmeans_model.cluster_centers_)
