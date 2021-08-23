from get_data import *
from graph_utils import *
import networkx as nx
import pandas as pd
import plotly.express as px

city_id = 'cGl0dHNidXJnaHBlbm5zeWx2YW5pYXVuaXRlZCBzdGF0ZXM' # Pittsburgh
hotspot_list = list_hotspots_in_city(city_id)

city_graph, hotspot_list = create_graph_from_hotspot_list(hotspot_list)

fig = generate_figure(city_graph)
fig.show()

# pagerank
pg = nx.pagerank(city_graph)
pg_sorted = dict(sorted(pg.items(), key=lambda item: item[1], reverse=True))

# betweenness centrality
bc = nx.betweenness_centrality(city_graph, normalized=True, endpoints=True)
bc_sorted = dict(sorted(bc.items(), key=lambda item: item[1], reverse=True))

rewards_list = []
bc_list = []
pg_list = []
num_witnesses_list = []
rssi_list = []
snr_list = []
redundancy_list = []
names_list = []
i = 0
for hotspot in hotspot_list:
    rewards_list.append(hotspot['rewards'])
    pg_list.append(pg[hotspot['address']])
    bc_list.append(bc[hotspot['address']])
    num_witnesses_list.append(hotspot['num_witnesses'])
    rssi_list.append(hotspot['rssi'])
    snr_list.append(hotspot['snr'])
    redundancy_list.append(hotspot['redundancy'])
    names_list.append(hotspot['name'])
    i += 1
    if i % 10 == 0:
        print(f"{str(i)} out of {str(len(hotspot_list))} hotspots complete...")

rewards_df = pd.DataFrame(data=[names_list, rewards_list, pg_list, bc_list, rssi_list,
                                snr_list, redundancy_list, num_witnesses_list]).transpose()
rewards_df.columns = ['name', 'rewards', 'pagerank', 'betweenness_centrality', 'rssi', 'snr', 'redundancy', 'num_witnesses']
rewards_df['num_witnesses'] = rewards_df['num_witnesses'].astype('int')

fig = px.scatter(rewards_df, x='pagerank', y='rewards', color='num_witnesses', hover_data=['name'])
fig.show()

fig = px.scatter(rewards_df, x='betweenness_centrality', y='rewards', color='num_witnesses', hover_data=['name'])
fig.show()

fig = px.scatter(rewards_df, x='redundancy', y='rssi', color='num_witnesses', hover_data=['name'])
fig.show()