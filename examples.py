from helium_network_graphs import *
import networkx as nx
import matplotlib.pyplot as plt


## CityGraph: generate graph of city

city_id = 'c3RhdGUgY29sbGVnZXBlbm5zeWx2YW5pYXVuaXRlZCBzdGF0ZXM' # State College, PA
city_graph_obj = CityGraph(city_id=city_id)

# generating the graph will take some time, depending on the number of hotspots in city
city_graph = city_graph_obj.generate_graph(calculate_rewards=False) # can also get this graph from CityGraph.g
assert type(city_graph) == nx.Graph

# calculate metrics using class methods or networkx
pg = city_graph_obj.pagerank()
bc = city_graph_obj.betweenness_centrality() # same as nx = nx.betweenness_centrality(city_graph)

# simple plotting with networkx and matplotlib, or for advanced, interactive plots, use Plotly
pg_colors = []
scaling_factor = 1 / max(pg.values())
for node in city_graph.nodes():
    pg_colors.append((pg[node] * scaling_factor, 0, 0))

fig, ax = plt.subplots()
nx.draw(city_graph, pos=city_graph_obj.positions, node_color=pg_colors, ax=ax)
ax.set_title('Hotspot Adjacency in State College, PA - Colored by Pagerank Value')
plt.show()


## MapperGraph: generate graph based on Helium Coverage Mapping data (https://mappers.helium.com/)

city_id = 'YmFsdGltb3JlbWFyeWxhbmR1bml0ZWQgc3RhdGVz' # Baltimore, MD
mapper_graph_obj = MapperGraph(city_id=city_id)

# this class needs some work, as right now it only considers hexes that contain hotspots
mapper_graph = mapper_graph_obj.generate_graph()
snr_values = []
for node in mapper_graph.nodes():
    snr_hex = mapper_graph.nodes[node]['snr']
    if not snr_hex:
        snr_hex = 0
    snr_values.append(snr_hex)

min_snr, max_snr = min(snr_values), max(snr_values)
snr_colors = []
for snr_value in snr_values:
    snr_value = (snr_value - min_snr) / (max_snr - min_snr)
    snr_colors.append((snr_value, 0, 0))

fig, ax = plt.subplots()
nx.draw(mapper_graph, pos=mapper_graph_obj.positions, node_color=snr_colors, ax=ax)
ax.set_title('Coverage Mapping in Baltimore, MD - Colored by SNR')
plt.show()