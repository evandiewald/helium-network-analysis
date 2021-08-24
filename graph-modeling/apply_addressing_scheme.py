import numpy as np
import addressing_utils as au
import matplotlib.pyplot as plt
import h3
import math
import matplotlib.patches as mpatch
from matplotlib.collections import PatchCollection
from get_data import *
import pickle
import json
import os

update_hex_list = True
overwrite_images = False
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

try:
    with open('hex_list.pkl', 'rb') as f:
        hex_list = pickle.load(f)
        print('Loaded existing hex list from disk.')
except FileNotFoundError:
    update_hex_list = True
    print('Hex list not found. Creating new list...')

if update_hex_list:
    changes_made = False
    print('Updating hexes...')
    hex_list = get_unique_hex_list(hotspot_list)
    print(f"{str(len(hotspot_list))} hotspots, {str(len(city_id))} cities, and {str(len(hex_list.keys()))} unique hexes found!")
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
            if changes_made:
                print('Changes made, saving hex data...')
                with open('hex_list.pkl', 'wb') as f:
                    pickle.dump(hex_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                    changes_made = False


labels = {}
counter = 0
hex_key_list = list(hex_list.keys())
for h in hex_key_list:
    save_path = f"hex_images/{h}.png"
    if os.path.exists(save_path) and not overwrite_images:
        continue

    # h = list(hex_list.keys())[40]
    labels[h] = {}
    unique_witnesses = []
    for hotspot in hex_list[h]['hotspots']:
        witnesses = list_witnesses_for_hotspot(hotspot)
        for witness in witnesses:
            if witness['address'] not in unique_witnesses:
                unique_witnesses.append(witness['address'])
    num_witnesses = len(unique_witnesses)

    ring_hexes = h3.k_ring(h, 3)
    total_hotspots_in_ring = 0
    update_file = False
    for h3_hex in ring_hexes:
        if h3_hex not in hex_list.keys():
            update_file = True
            hex_list[h3_hex] = {}
            hex_list[h3_hex]['hotspots'] = list_hotspots_in_hex(h3_hex)
            hex_list[h3_hex]['n_hotspots'] = len(hex_list[h3_hex]['hotspots'])
            hex_list[h3_hex]['elev'] = get_elevation_of_coords(h3.h3_to_geo(h3_hex))
            hex_list[h3_hex]['bnd'] = np.array(h3.h3_to_geo_boundary(h3_hex))
        total_hotspots_in_ring += hex_list[h3_hex]['n_hotspots']
        try:
            if hex_list[h3_hex]['elev'] < min_elev:
                min_elev = hex_list[h3_hex]['elev']
            if hex_list[h3_hex]['elev'] > max_elev:
                max_elev = hex_list[h3_hex]['elev']
        except NameError:
            min_elev, max_elev = hex_list[h3_hex]['elev'], hex_list[h3_hex]['elev']
    if update_file:
        print('New hexes found! Updating list...')
        with open('hex_list.pkl', 'wb') as f:
            pickle.dump(hex_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print('Still no changes to make. Continuing...')

    elev_range = max_elev - min_elev

    fig = plt.figure(figsize=(1, 1), dpi=100)
    for ring_hex in ring_hexes:
        normalized_num_hotspots = hex_list[ring_hex]['n_hotspots'] / total_hotspots_in_ring
        normalized_elev = (hex_list[ring_hex]['elev'] - min_elev) / elev_range
        plt.fill(hex_list[ring_hex]['bnd'][:,0], hex_list[ring_hex]['bnd'][:,1],
                 facecolor=(normalized_elev, normalized_num_hotspots, 0))

    labels[h] = {'num_witnesses': num_witnesses,
                 'total_hotspots_in_ring': total_hotspots_in_ring,
                 'witness_ratio': num_witnesses / total_hotspots_in_ring}

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    counter += 1
    if np.mod(counter, 10) == 0:
        print(f"{str(counter)} images complete out of {str(len(hex_key_list))}...")

with open('hex_images_labels/labels.json', 'w') as f:
    json.dump(labels, f)

