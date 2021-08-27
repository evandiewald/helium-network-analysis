import numpy as np
import addressing_utils as au
import matplotlib.pyplot as plt
import h3
import datetime
import math
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
from get_data import *
import pickle
import json
import torch
import os
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import from_networkx
import random

visualize_graph = False
city_id = ['cGl0dHNidXJnaHBlbm5zeWx2YW5pYXVuaXRlZCBzdGF0ZXM',        # Pittsburgh
           'ZGFsbGFzdGV4YXN1bml0ZWQgc3RhdGVz',                       # Dallas
           'Y29sb3JhZG8gc3ByaW5nc2NvbG9yYWRvdW5pdGVkIHN0YXRlcw',     # Colorado Springs
           'c2FuIGZyYW5jaXNjb2NhbGlmb3JuaWF1bml0ZWQgc3RhdGVz',       # San Francisco
           'cGhpbGFkZWxwaGlhcGVubnN5bHZhbmlhdW5pdGVkIHN0YXRlcw',     # Philadelphia
           'Y2hpY2Fnb2lsbGlub2lzdW5pdGVkIHN0YXRlcw',                 # Chicago
           'YXRsYW50YWdlb3JnaWF1bml0ZWQgc3RhdGVz',                   # Atlanta
           'bWlhbWlmbG9yaWRhdW5pdGVkIHN0YXRlcw',                     # Miami
           'bmV3IG9ybGVhbnNsb3Vpc2lhbmF1bml0ZWQgc3RhdGVz',           # New Orleans
           'c3RhdGUgY29sbGVnZXBlbm5zeWx2YW5pYXVuaXRlZCBzdGF0ZXM',    # State College
           'cmljaG1vbmR2aXJnaW5pYXVuaXRlZCBzdGF0ZXM']                # Richmond

hotspot_list = []
for city in city_id:
    hotspot_list += list_hotspots_in_city(city)

random.shuffle(hotspot_list)

with open('hex_list.pkl', 'rb') as f:
    hex_list = pickle.load(f)
    print('Loaded existing hex list from disk.')


try:
    with open('hotspot_data.json', 'r') as f:
        hotspot_data = json.load(f)
    print('Hotspot data loaded from disk.')
except FileNotFoundError:
    print('Gathering hotspot witness data.')
    hotspot_data = {}
    hotspot_counter = 0
    for hotspot in hotspot_list:

        if hotspot['status']['online'] == 'offline':
            continue

        witnesses = list_witnesses_for_hotspot(hotspot['address'])
        witness_list = []
        for witness in witnesses:
            if witness['status']['online'] == 'offline':
                continue
            witness_list.append(witness['address'])
        try:
            elev = hex_list[hotspot['location_hex']]['elev']
        except KeyError:
            elev = get_elevation_of_coords((hotspot['lat'], hotspot['lng']))
        hotspot_data[hotspot['address']] = {
            'location_hex': hotspot['location_hex'],
            'elev': elev,
            'lat': hotspot['lat'],
            'lng': hotspot['lng'],
            # 'rewards': get_hotspot_rewards(hotspot['address'], 5),
            'witnesses': witness_list
        }
        hotspot_counter += 1
        if np.mod(hotspot_counter, 100) == 0:
            print(f"{str(hotspot_counter)} out of {str(len(hotspot_list))} hotspots processed...")
            with open('hotspot_data.json', 'w') as f:
                json.dump(hotspot_data, f)


data_list = []
print('Preparing subgraphs. This may take a while...')
counter = 0
for h in list(hotspot_data.keys()): ##########################################################################
# h = list(hotspot_data.keys())[30]
    g = nx.Graph()
    ring_hexes = list(h3.k_ring(hotspot_data[h]['location_hex'], 3))
    ring_hexes.append(hotspot_data[h]['location_hex'])
    hotspots_in_ring = []
    witnesses_in_ring = []
    for ring_hex in ring_hexes:
        try:
            hotspots_in_hex, n_hotspots, elev = hex_list[ring_hex]['hotspots'], hex_list[ring_hex]['n_hotspots'], \
                                                hex_list[ring_hex]['elev']
        except KeyError:
            hotspots_in_hex = list_hotspots_in_hex(ring_hex)
            hex_list[ring_hex] = {
                'hotspots': hotspots_in_hex,
                'n_hotspots': len(hotspots_in_hex),
                'elev': get_elevation_of_coords(h3.h3_to_geo(ring_hex))
            }

        hotspots_in_ring += hex_list[ring_hex]['hotspots']


    pos_dict = {}
    g.add_node(h, pos=(hotspot_data[h]['lng'], hotspot_data[h]['lat']),
               elev=hotspot_data[h]['elev'], node_class=1)
    pos_dict[h] = g.nodes[h]['pos']

    if len(hotspot_data[h]['witnesses']) == 0:
        continue
    else:
        for witness_address in hotspot_data[h]['witnesses']:
            if witness_address in hotspots_in_ring:
                node_class = 1
            else:
                node_class = 0
            try:
                existing_node = g.nodes[witness_address]
            except KeyError:


                try:
                    g.add_node(witness_address,
                               pos=(hotspot_data[witness_address]['lng'], hotspot_data[witness_address]['lat']),
                               elev=hotspot_data[witness_address]['elev'], node_class=node_class) #hotspot_data[
                    # witness_address][
                    # 'rewards'])
                except KeyError:
                    details = get_hotspot_details(witness_address)
                    g.add_node(witness_address,
                               pos=(details['lng'], details['lat']),
                               elev=get_elevation_of_coords(h3.h3_to_geo(details['location_hex'])),
                               # rewards=get_hotspot_rewards(details['address'], 5))
                               node_class=node_class)
            pos_dict[witness_address] = g.nodes[witness_address]['pos']
            dist = math.sqrt((g.nodes[witness_address]['pos'][0] - g.nodes[h]['pos'][0])**2 + (g.nodes[witness_address]['pos'][0] - g.nodes[h]['pos'][0])**2)
            g.add_edge(h, witness_address, dist=dist)

        try:
            gt = from_networkx(g)
        except TypeError:
            print('Skipping hotspot ', h)
            continue

        data = Data(x=gt.elev.reshape((gt.num_nodes, 1)), edge_index=gt.edge_index, pos=gt.pos.reshape((gt.num_nodes,2)),
                    y=gt.node_class, edge_attr=gt.dist, num_classes=2)  #
        # predict rewards of ALL nodes
        data_list.append(data)
        counter += 1
        if np.mod(counter, 100) == 0:
            print(f"{str(counter)} out of {str(len(hotspot_data.keys()))} subgraphs generated...")

        if visualize_graph:
            nx.draw(g, pos=pos_dict)
            plt.show()


print('Subgraph generation complete. Saving data.')
with open(f"subgraph_datasets/subgraph_data_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.pkl", 'wb') as f:
    pickle.dump(data_list, f, protocol=pickle.HIGHEST_PROTOCOL)

# data_list can be read and then used in a DataLoader object in torch_geometric.data