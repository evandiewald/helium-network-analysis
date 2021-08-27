from get_data import *
import math
import networkx as nx
import h3
import json
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import pickle

class CityGraph:
    def __init__(self, city_id):
        self.city_id = city_id
        self.g = None
        self.hotspot_list = list_hotspots_in_city(city_id)

    def generate_graph(self, calculate_rewards=False):
        """Uses list of hotspots and witness data to create nodes and adjacency with networkx"""
        g = nx.Graph()

        # create nodes for all hotspots in list
        for hotspot in self.hotspot_list:
            if calculate_rewards:
                hotspot['rewards'] = get_hotspot_rewards(hotspot['address'], 5)
            else:
                hotspot['rewards'] = 0
            g.add_node(hotspot['address'], pos=(hotspot['lng'], hotspot['lat']), lat=hotspot['lat'], lng=hotspot['lng'],
                       location_hex=hotspot['location_hex'], name=hotspot['name'], rewards=hotspot['rewards'],
                       gain=hotspot['gain'], elevation=hotspot['elevation'])

        # create edges representing hotspots that witness each other
        i = 0
        for hotspot in self.hotspot_list:
            witness_list = list_witnesses_for_hotspot(hotspot['address'])
            hotspot['num_witnesses'] = len(witness_list)
            g.nodes[hotspot['address']]['num_witnesses'] = hotspot['num_witnesses']
            for witness in witness_list:

                # make sure that all witnesses are also in this subset
                if not any(witness['address'] == h['address'] for h in self.hotspot_list):
                    continue

                # calculate distance between witness and challengee
                dist = math.sqrt((hotspot['lat'] - witness['lat']) ** 2 + (hotspot['lng'] - witness['lng']) ** 2)
                g.add_edge(hotspot['address'], witness['address'], weight=dist)

            i += 1
            if i % 10 == 0:
                print(f"{str(i)} out of {str(len(self.hotspot_list))} hotspots complete...")

        self.g = g

    def pagerank(self):
        if not self.g:
            raise NameError('You must generate a graph first with CityGraph.generate_graph()')
        return nx.pagerank(self.g)

    def betweenness_centrality(self):
        if not self.g:
            raise NameError('You must generate a graph first with CityGraph.generate_graph()')
        return nx.betweenness_centrality(self.g)


class MapperGraph:
    def __init__(self, city_id: str):
        self.city_id = city_id
        self.hotspot_list = list_hotspots_in_city(city_id)
        self.city_hexes = get_unique_hex_list(self.hotspot_list)
        self.g = None

    def generate_graph(self):
        g = nx.Graph()

        # most straightforward to create all the nodes first, then edges
        for h in self.city_hexes.keys():
            num_hotspots_in_hex = len(self.city_hexes[h]['hotspots'])
            # get mapper data, which is collected at h9 res
            mapper_stats = get_mapper_uplinks_for_location_hex(h3.h3_to_center_child(h, 9))
            g.add_node(h, num_hotspots=num_hotspots_in_hex, pos=h3.h3_to_geo(h), rssi=mapper_stats['bestRssi'],
                       snr=mapper_stats['bestSnr'], redundancy=mapper_stats['redundancy'])

        for h in self.city_hexes.keys():
            hex_neighbors = h3.h3_k_ring(h, 1)
            for neighbor in hex_neighbors:
                if neighbor in g.nodes():
                    g.add_edge(h, neighbor)
                else:
                    continue


class HotspotData:
    def __init__(self, hotspot_list):
        self.hotspot_list = hotspot_list

    def generate_hotspot_data(self, online_only=True, save_checkpoints=True, save_path='hotspot_data.json'):
        hotspot_data = {}
        hotspot_counter = 0
        for hotspot in self.hotspot_list:

            if online_only and hotspot['status']['online'] == 'offline':
                continue

            witnesses = list_witnesses_for_hotspot(hotspot['address'])
            witness_list = []
            for witness in witnesses:
                if witness['status']['online'] == 'offline':
                    continue
                witness_list.append(witness['address'])

            elev = get_elevation_of_coords((hotspot['lat'], hotspot['lng']))

            hotspot_data[hotspot['address']] = {
                'location_hex': hotspot['location_hex'],
                'elev': elev,
                'lat': hotspot['lat'],
                'lng': hotspot['lng'],
                'witnesses': witness_list
            }
            hotspot_counter += 1
            if np.mod(hotspot_counter, 100) == 0:
                print(f"{str(hotspot_counter)} out of {str(len(self.hotspot_list))} hotspots processed...")
                if save_checkpoints:
                    with open(save_path, 'w') as f:
                        json.dump(hotspot_data, f)

        return hotspot_data

    def generate_hex_list(self, save_checkpoint=True, save_path='hex_list.pkl'):
        changes_made = False
        hex_list = get_unique_hex_list(self.hotspot_list)
        hex_counter = 0
        for h in hex_list.keys():
            hex_counter += 1
            try:
                elev, n_hotspots, bnd = hex_list[h]['elev'], hex_list[h]['n_hotspots'], hex_list[h]['bnd']
            except KeyError:
                hex_list[h]['elev'] = get_elevation_of_coords(h3.h3_to_geo(h))
                hex_list[h]['n_hotspots'] = len(hex_list[h]['hotspots'])
                hex_list[h]['bnd'] = np.array(h3.h3_to_geo_boundary(h))
                changes_made = True
            if np.mod(hex_counter, 100) == 0:
                print(f"{str(hex_counter)} out of {str(len(hex_list.keys()))} hexes complete...")
                if changes_made and save_checkpoint:
                    print('Changes made, saving hex data...')
                    with open(save_path, 'wb') as f:
                        pickle.dump(hex_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                        changes_made = False


class HotspotGraph:
    def __init__(self, hotspot_address: str):
        self.address = hotspot_address
        self.witnesses = list_witnesses_for_hotspot(hotspot_address)

    def generate_nearby_hotspot_graph(self, k_rings: int, hotspot_data, hex_list, format='torch'):
        """Generates edges connecting this hotspot to others within the k rings of its h8 hex.
        Each node will have a label of 0, meaning it is not a witness, or 1, meaning it is a witness of this hotspot.
        Useful for creating node classification-based graph neural networks."""

        g = nx.Graph()
        ring_hexes = list(h3.k_ring(hotspot_data[self.address]['location_hex'], k_rings))
        ring_hexes.append(hotspot_data[self.address]['location_hex'])

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
        g.add_node(self.address, pos=(hotspot_data[self.address]['lng'], hotspot_data[self.address]['lat']),
                   elev=hotspot_data[self.address]['elev'], node_class=1)
        pos_dict[self.address] = g.nodes[self.address]['pos']

        for witness_address in hotspot_data[self.address]['witnesses']:
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
                               elev=hotspot_data[witness_address]['elev'], node_class=node_class)
                except KeyError:
                    details = get_hotspot_details(witness_address)
                    g.add_node(witness_address,
                               pos=(details['lng'], details['lat']),
                               elev=get_elevation_of_coords(h3.h3_to_geo(details['location_hex'])),
                               # rewards=get_hotspot_rewards(details['address'], 5))
                               node_class=node_class)
            pos_dict[witness_address] = g.nodes[witness_address]['pos']
            dist = math.sqrt((g.nodes[witness_address]['pos'][0] - g.nodes[self.address]['pos'][0]) ** 2 + (
                        g.nodes[witness_address]['pos'][0] - g.nodes[self.address]['pos'][0]) ** 2)
            g.add_edge(self.address, witness_address, dist=dist)

        if format == 'torch':
            gt = from_networkx(g)

            data = Data(x=gt.elev.reshape((gt.num_nodes, 1)), edge_index=gt.edge_index,
                        pos=gt.pos.reshape((gt.num_nodes, 2)),
                        y=gt.node_class, edge_attr=gt.dist, num_classes=2)
            return data
        else:
            return g


    def generate_witness_graph(self, hotspot_data, format='networkx'):
        """Generates edges connecting this hotspot to all of its recent witnesses."""

        g = nx.Graph()
        g.add_node(self.address)
        g.add_node(self.address, pos=(hotspot_data[self.address]['lng'], hotspot_data[self.address]['lat']),
                   elev=hotspot_data[self.address]['elev'])
        for witness in self.witnesses:
            try:
                g.add_node(witness['address'], pos=(hotspot_data[witness['address']]['lng'], hotspot_data[witness['address']]['lat']),
                       elev=hotspot_data[witness['address']]['elev'])
            except KeyError:
                witness_details = get_hotspot_details(witness['address'])
                elevation = get_elevation_of_coords((witness_details['lat'], witness_details['lng']))
                g.add_node(witness['address'],
                           pos=(witness_details['lng'], witness_details['lat']),
                           elev=elevation)
            g.add_edge(self.address, witness['address'])

        if format == 'torch':
            gt = from_networkx(g)

            data = Data(x=gt.elev.reshape((gt.num_nodes, 1)), edge_index=gt.edge_index,
                        pos=gt.pos.reshape((gt.num_nodes, 2)))
            return data

        else:
            return g







