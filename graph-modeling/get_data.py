import requests
import datetime
import h3
import networkx as nx
import urllib
import numpy as np
import pandas as pd
import time
import random


def list_hotspots_in_city(city_id: str):
    """retrieves all the hotspots in a city"""
    url = 'https://api.helium.io/v1/cities/' + city_id + '/hotspots'
    r = requests.get(url)
    hotspots = r.json()['data']
    return hotspots


def list_witnesses_for_hotspot(hotspot_address: str):
    """lists a hotspot's witnesses over the last 5 days"""
    url = 'https://api.helium.io/v1/hotspots/' + hotspot_address + '/witnesses'
    r = requests.get(url)
    witnesses = r.json()['data']
    return witnesses


def get_hotspot_rewards(hotspot_address: str, n_days: int):
    """Get a hotspot's cumulative rewards for the past n days"""
    start_time = datetime.datetime.isoformat(datetime.datetime.now() - datetime.timedelta(days=n_days))
    url = f"https://api.helium.io/v1/hotspots/{hotspot_address}/rewards/sum?min_time={start_time}"
    r = requests.get(url)
    rewards = r.json()['data']['total']
    return rewards


def get_hotspot_beacon_count(hotspot_address: str, n_days: int):
    start_time = datetime.datetime.timestamp(datetime.datetime.now() - datetime.timedelta(days=n_days))
    url = f"https://api.helium.io/v1/hotspots/{hotspot_address}/activity?filter_types=poc_receipts_v1"
    beacon_count = 0
    while 1:
        r = requests.get(url)
        res = r.json()
        try:
            cursor = res['cursor']
        except KeyError:
            break
        for challenge in res['data']:
            if challenge['time'] < start_time:
                break
            else:

                beacon_count += 1
        url = f"https://api.helium.io/v1/hotspots/{hotspot_address}/activity?filter_types=beacons&cursor={cursor}"
    return beacon_count


def get_mapper_uplinks_for_location_hex(hex_id: str):
    assert h3.h3_get_resolution(hex_id) == 9
    url = f"https://mappers.helium.com/api/v1/uplinks/hex/{hex_id}"
    r = requests.get(url)
    uplinks = r.json()['uplinks']
    if len(uplinks) == 0:
        bestRssi = None
        bestSnr = None
        redundancy = None
    else:
        for i in range(len(uplinks)):
            if i == 0:
                bestRssi = uplinks[i]['rssi']
                bestSnr = uplinks[i]['snr']
                redundancy = len(uplinks)
            else:
                if uplinks[i]['rssi'] > bestRssi:
                    bestRssi = uplinks[i]['rssi']
                if uplinks[i]['snr'] > bestSnr:
                    bestSnr = uplinks[i]['snr']
    stats = {
        'bestRssi': bestRssi,
        'bestSnr': bestSnr,
        'redundancy': redundancy
    }
    return stats


def get_unique_hex_list(hotspot_list: list):
    """Gets list of unique h8 hexes that contain hotspots from list"""
    unique_hexes = {}
    for hotspot in hotspot_list:
        if hotspot['location_hex'] not in unique_hexes.keys():
            unique_hexes[hotspot['location_hex']] = {}
            unique_hexes[hotspot['location_hex']]['hotspots'] = [hotspot['address']]
        else:
            unique_hexes[hotspot['location_hex']]['hotspots'].append(hotspot['address'])
    return unique_hexes


def get_elevation_of_coords(coords):
    # USGS Elevation Point Query Service
    url = r'https://nationalmap.gov/epqs/pqs.php?'
    (lat, lon) = coords
    # define rest query params
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }
    # format query string and return query value
    while True:
        backoff = 1
        try:
            result = requests.get((url + urllib.parse.urlencode(params)))
            break
        except ConnectionError:
            print('Backing off USGS service...')
            time.sleep(backoff)
            backoff *= 2 + random.rand()
            if backoff > 512:
                print(f"Backoff time: {str(backoff)}")
                break
    elevation = result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']
    return elevation


def list_hotspots_in_hex(h):
    url = 'https://api.helium.io/v1/hotspots/hex/' + h
    r = requests.get(url)
    hotspots = r.json()['data']
    hotspots_in_hex = []
    for hotspot in hotspots:
        hotspots_in_hex.append(hotspot['address'])
    return hotspots_in_hex


def create_hotspot_hex_graph(hotspot_list: list):
    g = nx.Graph()
    hexes = get_unique_hex_list(hotspot_list)
    count = 0
    for hex_h8 in hexes.keys():
        # PART ONE: GET ADJACENT HEXES (i.e. hexes that contain a witness to one of the hotspots in this hex)
        (lat, lon) = h3.h3_to_geo(hex_h8)
        g.add_node(hex_h8, num_hotspots=len(hexes[hex_h8]), pos=(lon, lat), elev=get_elevation_of_coords((lat, lon)))
        # todo: add more geographic features to node, like terrain/elevation
        connected_hexes = []
        for hotspot in hexes[hex_h8]:
            witnesses = list_witnesses_for_hotspot(hotspot)
            for witness in witnesses:
                # make sure that all witnesses are also in this subset
                if not any(witness['address'] == h['address'] for h in hotspot_list):
                    continue
                # todo: here, could make the weight of the edge equal to the RSSI reported in receipt
                if witness['location_hex'] not in connected_hexes:
                    connected_hexes.append(witness['location_hex'])
        for edge_hex in connected_hexes:
            g.add_edge(hex_h8, edge_hex)

        # PART TWO: GET AVERAGE RSSI/SNR IN HEX FROM MAPPER DATA
        child_hexes = h3.h3_to_children(hex_h8, 9)
        rssi_list = []
        snr_list = []
        for child in child_hexes:
            child_stats = get_mapper_uplinks_for_location_hex(child)
            if child_stats['bestRssi']:
                rssi_list.append(child_stats['bestRssi'])
                snr_list.append(child_stats['bestSnr'])

        if len(rssi_list) > 0:
            g.nodes[hex_h8]['rssi'], g.nodes[hex_h8]['snr'] = np.mean(rssi_list), np.mean(snr_list)
        else:
            # g.nodes[hex_h8]['rssi'], g.nodes[hex_h8]['snr'] = None, None
            continue

        count += 1
        if np.mod(count, 10) == 0:
            print(f"{str(count)} of {len(hexes.keys())} hexes analyzed...")

    return g


def clean_graph_mapping_data(g: nx.Graph):
    rssi_list = []
    snr_list = []
    for node in g.nodes():
        try:
            rssi_list.append(g.nodes[node]['rssi'])
            snr_list.append(g.nodes[node]['snr'])
        except KeyError:
            continue

    rssi_mean = np.mean(rssi_list)
    snr_mean = np.mean(snr_list)
    for node in g.nodes():
        try:
            a = g.nodes[node]['rssi']
        except KeyError:
            g.nodes[node]['rssi'] = rssi_mean
            g.nodes[node]['snr'] = snr_mean
    return g


def graph_to_node_data_dict(g: nx.Graph):
    pg = nx.pagerank(g)
    bc = nx.betweenness_centrality(g)
    node_data = []
    for node in g.nodes():
        data = g.nodes[node]
        data['pg'], data['bc'] = pg[node], bc[node]
        # data['hex'] =
        try:
            data.pop('pos')
        except KeyError:
            pass
        node_data.append(data)
    df = pd.DataFrame(node_data)
    return df




