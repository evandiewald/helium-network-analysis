from get_data import *
from graph_utils import *
import networkx as nx

city_id = 'cGl0dHNidXJnaHBlbm5zeWx2YW5pYXVuaXRlZCBzdGF0ZXM' # Pittsburgh
hotspot_list = list_hotspots_in_city(city_id)

# g = create_hotspot_hex_graph(hotspot_list)
g = nx.read_gpickle('pittsburgh_graph.pkl')
g = clean_graph_mapping_data(g)

fig = generate_figure(g)
fig.show()

