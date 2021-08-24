import networkx as nx
import h3
from get_data import *
import math
import plotly.graph_objects as go


def create_graph_from_hotspot_list(hotspot_list: list):
    """Uses list of hotspots and witness data to create nodes and adjacency with networkx"""
    g = nx.Graph()

    for hotspot in hotspot_list:
        print(hotspot['address'])
        hotspot['rewards'] = get_hotspot_rewards(hotspot['address'], 5)
        hotspot['num_beacons'] = get_hotspot_beacon_count(hotspot['address'], 5)
        # h9_hex = h3.h3_to_parent(hotspot['location'], 9) # mapping data is collected at h9, but locations are h12
        # mapping_stats = get_mapper_uplinks_for_location_hex(h9_hex)
        # hotspot['rssi'] = mapping_stats['bestRssi']
        # hotspot['snr'] = mapping_stats['bestSnr']
        # hotspot['redundancy'] = mapping_stats['redundancy']
        g.add_node(hotspot['address'], pos=(hotspot['lng'], hotspot['lat']), lat=hotspot['lat'], lon=hotspot['lng'],
                   location_hex=hotspot['location_hex'], name=hotspot['name'], rewards=hotspot['rewards'],
                   gain=hotspot['gain'], elevation=hotspot['elevation'], num_beacons=hotspot['num_beacons']) #, rssi=mapping_stats['bestRssi'],
                   # snr=mapping_stats['bestSnr'], redundancy=mapping_stats['redundancy'])

    i = 0
    for hotspot in hotspot_list:
        witness_list = list_witnesses_for_hotspot(hotspot['address'])
        hotspot['num_witnesses'] = len(witness_list)
        g.nodes[hotspot['address']]['num_witnesses'] = hotspot['num_witnesses']
        for witness in witness_list:
            # make sure that all witnesses are also in this subset
            if not any(witness['address'] == h['address'] for h in hotspot_list):
                continue
            dist = math.sqrt((hotspot['lat'] - witness['lat'])**2 + (hotspot['lng'] - witness['lng'])**2)
            g.add_edge(hotspot['address'], witness['address'], weight=dist)
        i += 1
        if i % 10 == 0:
            print(f"{str(i)} out of {str(len(hotspot_list))} hotspots complete...")
    return g, hotspot_list


def generate_figure(G: nx.Graph):
    edge_x = []
    edge_y = []
    xtext = []
    ytext = []
    # edge_dist = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        xtext.append((x0 + x1) / 2)
        ytext.append((y0 + y1) / 2)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        # dist = str(round(G.edges[edge[0], edge[1]]['weight'],3))
        # edge_dist.append(dist)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines')

    # eweights_trace = go.Scatter(x=xtext, y=ytext, mode='markers',
    #                             marker=dict(size=0),
    #                             textposition='top center',
    #                             hovertemplate='Distance: %{text}<extra></extra>')
    # eweights_trace.text = edge_dist

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            # colorscale options
            #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
            #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
            #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='RSSI',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_rssi = []
    node_text = []
    node_pg = list(nx.pagerank(G).values())
    node_bc = list(nx.betweenness_centrality(G).values())
    node_elev = []
    node_hex_x = []
    node_hex_y = []
    # for node, adjacencies in enumerate(G.adjacency()):
    #     node_adjacencies.append(len(adjacencies[1]))
    #     # node_text.append('# of connections: ' + str(len(adjacencies[1])))
    #     node_text.append(G.nodes[adjacencies[0]]['name'])
    for node in G.nodes():
        node_elev.append(G.nodes[node]['elev'])
        node_rssi.append(G.nodes[node]['rssi'])
        node_text.append(f"RSSI: {str(G.nodes[node]['rssi'])}<br>"
                         f"SNR: {str(G.nodes[node]['snr'])}<br>"
                         f"Num. Hotspots: {str(G.nodes[node]['num_hotspots'])}")
        bnd = h3.h3_to_geo_boundary(node)
        for b in bnd:
            node_hex_y.append(b[0])
            node_hex_x.append(b[1])
        node_hex_y.append(bnd[0][0])
        node_hex_x.append(bnd[0][1])
        node_hex_y.append(None)
        node_hex_x.append(None)


    # node_trace.marker.size = np.multiply(node_bc, 1e3)
    node_trace.marker.size = np.multiply(node_elev, 1e-1) #np.multiply(node_pg, 1e3)
    node_trace.marker.color = node_rssi
    node_trace.text = node_text

    token = 'pk.eyJ1IjoiZXZhbmRpZXdhbGQiLCJhIjoiY2tzbGxxMjVtMDIwcDJvbWdweTNiemJwZyJ9.wcviBe-rK44wTEoNJGT0Gg'
    map_trace = go.Scattermapbox(lat=node_y, lon=node_x)
    # fig = ff.create_hexbin_mapbox(lat=node_y, lon=node_x)
    # fig = go.Figure()
    # fig.update_layout(mapbox_style="dark", mapbox_accesstoken=token, mapbox_zoom=12, mapbox_center_lat=40.44,
    #                   mapbox_center_lon=-80, )
    hex_trace = go.Scatter(x=node_hex_x, y=node_hex_y, fill='toself')

    fig = go.Figure(data=[hex_trace, edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Helium Network in Pittsburgh, PA - Markers are Sized by Elevation',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        annotations=[dict(
                            # text="Python code: <a href='https://plotly.com/ipython-notebooks/network-graphs/'> https://plotly.com/ipython-notebooks/network-graphs/</a>",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002)],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    return fig