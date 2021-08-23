from get_data import *
from graph_utils import *
from networkx import read_gpickle


g = read_gpickle('pittsburgh_graph_cleaned.pkl')

for node in g.nodes():
    node_data = g.nodes()[node]
