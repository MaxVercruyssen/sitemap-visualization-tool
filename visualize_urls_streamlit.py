# '''
#     Visualize a list of URLs by site path.
#     This script reads in the sitemap_layers.csv file created by the
#     categorize_urls.py script and builds a graph visualization using Graphviz.
#     Graph depth can be specified by executing a call like this in the
#     terminal:
#         python visualize_urls.py --depth 4 --limit 10 --title "My Sitemap" --style "dark" --size "40"
#     The same result can be achieved by setting the variables manually at the head
#     of this file and running the script with:
#         python visualize_urls.py
# '''
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_tags import st_tags, st_tags_sidebar
import colorsys
import math
from ast import literal_eval
import re
st.set_page_config(layout="wide")


# Set global variables


limit = st.sidebar.number_input('Maximum number of nodes for a branch', min_value=0, value = 50)       # Maximum number of nodes for a branch
graph_depth = 5 # Number of layers deep to plot categorization

#title = st.sidebar.text_input('Graph title', value="")    # Graph title
title = ''
style = 'light'  # Graph style, can be "light" or "dark"
size = '8,5'     # Size of rendered graph
output_format = 'pdf'   # Format of rendered image - pdf,png,tiff
skip = ''        # List of branches to restrict from expanding
inclEndpoint = st.sidebar.checkbox('Include endpoints', value=False)

# Import external library dependencies

import pandas as pd
# import graphviz
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--depth', type=int, default=graph_depth,
#                     help='Number of layers deep to plot categorization')
# parser.add_argument('--limit', type=int, default=limit,
#                     help='Maximum number of nodes for a branch')
# parser.add_argument('--title', type=str, default=title,
#                     help='Graph title')
# parser.add_argument('--style', type=str, default=style,
#                     help='Graph style, can be "light" or "dark"')
# parser.add_argument('--size', type=str, default=size,
#                     help='Size of rendered graph')
# parser.add_argument('--output-format', type=str, default=output_format,
#                     help='Format of the graph you want to save. Allowed formats are jpg, png, pdf or tif')
# parser.add_argument('--skip', type=str, default=skip,
#         help="List of branches that you do not want to expand. Comma separated: e.g. --skip 'news,events,datasets'")
# args = parser.parse_args()


# Update variables with arguments if included

# graph_depth = args.depth
# limit = args.limit
# title = args.title
# style = args.style
# size = args.size
# output_format = args.output_format
# skip = args.skip.split(',')

# Main script functions

def listinCSV(dfWithLists):
  for col in dfWithLists:
    if isinstance(dfWithLists[col][0], str):
      if dfWithLists[col][0][0] == "[":
      #if col != 'Unnamed: 0'and col != 'link'and col != 'HtmlBody'and col != 'content_merge':
        dfWithLists[col]=dfWithLists[col].apply(literal_eval)
  return dfWithLists

def make_sitemap_graph(df, layers=graph_depth, limit=limit, size=size, output_format=output_format, skip=skip, inclEndpoint=inclEndpoint, methodeInterpretation = '', keywordList = []):
    ''' Make a sitemap graph up to a specified layer depth.

    sitemap_layers : DataFrame
        The dataframe created by the peel_layers function
        containing sitemap information.

    layers : int
        Maximum depth to plot.

    limit : int
        The maximum number node edge connections. Good to set this
        low for visualizing deep into site maps.
    
    output_format : string
        The type of file you want to save in PDF, PNG, TIFF, JPG

    skip : list
        List of branches that you do not want to expand.
    '''
    nodes = []
    edges = []

    # Check to make sure we are not trying to plot too many layers
    if layers > len(df) - 1:
        layers = len(df)-1
        print('There are only %d layers available to plot, setting layers=%d'
              % (layers, layers))


    # Initialize graph
    # f = graphviz.Digraph('sitemap', filename='sitemap_graph_%d_layer' % layers, format='%s' % output_format)
    # f.body.extend(['rankdir=LR', 'size="%s"' % size])


    def add_branch(existingNodes, names, vals,valsKey, limit,inclEndpoint, maxCounts, connect_to=''):
        ''' Adds a set of nodes and edges to nodes on the previous layer. '''

        # Get the currently existing node names
        node_names = [existingNodes[i].id for i in range(len(existingNodes))]

        # Only add a new branch it it will connect to a previously created node
        if connect_to:
            if connect_to in node_names:
                for name, val, valKey in list(zip(names, vals, valsKey))[:limit]:
                    if val > int( not inclEndpoint):
                        # f.node(name='%s-%s' % (connect_to, name), label=name)
                        
                        if methodeInterpretation == 'Just Count':
                            x=val/maxCounts
                            countslabel= str(val)
                        else:
                            x=valKey/maxCounts
                            countslabel=str(valKey) + '/' + str(val)

                        s = 2*math.log(1+x,2)/(1+math.log(1+x,2))
                        (r, g, b) = colorsys.hsv_to_rgb(248/360,s, 82/100)
                        hexColor  = '%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

                        nodes.append( Node(id='%s-%s' % (connect_to, name), 
                                label=name, 
                                size=400,
                                color="#{}".format(hexColor),
                                symbolType="circle",
                                        ) 
                        ) 

                        # f.edge(connect_to, '%s-%s' % (connect_to, name), label=)
                        edges.append( Edge(source=connect_to, 
                                    label='{}'.format(countslabel), 
                                    target='%s-%s' % (connect_to, name),
                                    labelPosition="center",
                                    type="CURVE_SMOOTH") 
                        )
                    # else:
                    #     print(name)


    ## f.attr('node', shape='rectangle') # Plot nodes as rectangles
    if methodeInterpretation == 'Just Count':
        maxCounts = df.shape[0]
    else:
        maxCounts = max(df['countKey'].sum(),1)
    # Add the first layer of nodes
    for name, counts, countKey in df.groupby(['0']).agg({'counts': 'sum',
                                                           'countKey': 'sum'}).reset_index()\
                                                            .sort_values(['counts'], ascending=False).values:
        ## f.node(name=name, label='{} ({:,})'.format(name, counts))
        if methodeInterpretation == 'Just Count':
            x=counts/maxCounts
            countslabel= str(counts)
        else:
            x=countKey/maxCounts
            countslabel=str(countKey) + '/' + str(counts)

        s = 2*math.log(1+x,2)/(1+math.log(1+x,2))
        (r, g, b) = colorsys.hsv_to_rgb(248/360,s, 82/100)
        hexColor  = '%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

        nodes.append( 
            Node(id=name, 
                label='{} ({})'.format(name, countslabel), 
                size=400,
                color="#{}".format(hexColor),
                symbolType="square",
            ) 
        ) 

    if layers == 0:
        return nodes, edges

    # f.attr('node', shape='oval') # Plot nodes as ovals
    # f.graph_attr.update()

    # Loop over each layer adding nodes and edges to prior nodes
    for i in range(1, layers+1):
        cols = [str(i_) for i_ in range(i)]
        nodes_ = df[cols].drop_duplicates().values
    
        for j, k in enumerate(nodes_):

            # Compute the mask to select correct data
            mask = True
            for j_, ki in enumerate(k):
                mask &= df[str(j_)] == ki

            # Select the data then count branch size, sort, and truncate
            data = df[mask].groupby([str(i)]).agg({'counts': 'sum',
                                                    'countKey': 'sum'})\
                                                    .reset_index().sort_values(['counts'], ascending=False)
            
            # Add to the graph unless specified that we do not want to expand k-1
            if (not skip) or (k[-1] not in skip):
                

                add_branch(nodes,
                       names=data[str(i)].values,
                       vals=data['counts'].values,
                       valsKey=data['countKey'].values,
                       limit=limit,
                       inclEndpoint = inclEndpoint,
                       maxCounts = maxCounts,
                       connect_to='-'.join(['%s']*i) % tuple(k))

            #print(('Built graph up to node %d / %d in layer %d' % (j, len(nodes_), i))\
                    # .ljust(50), end='\r')

    return nodes, edges


# def apply_style(f, style, title=''):
#     ''' Apply the style and add a title if desired. More styling options are
#     documented here: http://www.graphviz.org/doc/info/attrs.html#d:style

#     f : graphviz.dot.Digraph
#         The graph object as created by graphviz.

#     style : str
#         Available styles: 'light', 'dark'

#     title : str
#         Optional title placed at the bottom of the graph.
#     '''

#     dark_style = {
#         'graph': {
#             'label': title,
#             'bgcolor': '#3a3a3a',
#             'fontname': 'Helvetica',
#             'fontsize': '18',
#             'fontcolor': 'white',
#         },
#         'nodes': {
#             'style': 'filled',
#             'color': 'white',
#             'fillcolor': 'black',
#             'fontname': 'Helvetica',
#             'fontsize': '14',
#             'fontcolor': 'white',
#         },
#         'edges': {
#             'color': 'white',
#             'arrowhead': 'open',
#             'fontname': 'Helvetica',
#             'fontsize': '12',
#             'fontcolor': 'white',
#         }
#     }

#     light_style = {
#         'graph': {
#             'label': title,
#             'fontname': 'Helvetica',
#             'fontsize': '18',
#             'fontcolor': 'black',
#         },
#         'nodes': {
#             'style': 'filled',
#             'color': 'black',
#             'fillcolor': '#dbdddd',
#             'fontname': 'Helvetica',
#             'fontsize': '14',
#             'fontcolor': 'black',
#         },
#         'edges': {
#             'color': 'black',
#             'arrowhead': 'open',
#             'fontname': 'Helvetica',
#             'fontsize': '12',
#             'fontcolor': 'black',
#         }
#     }

#     if style == 'light':
#         apply_style = light_style

#     elif style == 'dark':
#         apply_style = dark_style

#     f.graph_attr = apply_style['graph']
#     f.node_attr = apply_style['nodes']
#     f.edge_attr = apply_style['edges']

#     return f

# def dot_to_json(file_in):
#     import networkx
#     from networkx.readwrite import json_graph
#     import pydot
#     graph_netx = networkx.drawing.nx_pydot.read_dot(file_in)
#     graph_json = json_graph.node_link_data( graph_netx )
#     return json_graph.node_link_data(graph_netx)

def main():

    # Read in categorized data
    sitemap_layers_raw = pd.read_csv('sitemap_layers.csv', dtype=str)
    sitemap_layers_raw.columns = sitemap_layers_raw.columns.str.replace('/', '_')
    

    maxi=0
    for i in list(sitemap_layers_raw.columns):
        try:
            if int(i)> maxi:
                maxi=int(i)
        except:
            pass
    graph_depth = st.slider('Graph depth', value=3, min_value=0, max_value=maxi, key="slider")  # Number of layers deep to plot categorization

    tags = pd.read_csv('dfresult.csv', dtype=str)
    tags.columns = tags.columns.str.replace('/', '_')
    tags[pd.isna(tags)]="['x']"
    tags = listinCSV(tags)


    filterTags = tags.loc[:, 'KeyBert':]
    sitemap_layers = pd.concat([sitemap_layers_raw, filterTags], axis=1)
    sitemap_layers = sitemap_layers.applymap(lambda s: s.lower() if type(s) == str else s)
    MethodeOption = list(filterTags.columns)
    MethodeOption.insert(0,'Just Count')
    methode = st.sidebar.selectbox('What model would you like to use?',MethodeOption)
    keywordList = []
    sitemap_layers['countKey'] =0
    if methode != 'Just Count':
        keywordList = st_tags_sidebar(
                    label='# Enter Keywords:',
                    text='Press enter to add more',
                    suggestions=sum(filterTags[methode],[])
                    )
        dropDown = st.sidebar.selectbox(
                    'suggestions tags:',
                    sum(filterTags[methode],[]))
        for keyFilter in keywordList:
            r = re.compile(keyFilter.lower())
            #sitemap_layers['countKey']=sitemap_layers.apply(lambda x : x.countKey+1 if keyFilter in x[methode] else x.countKey, axis=1 )
            sitemap_layers['countKey']=sitemap_layers.apply(lambda x : x.countKey+1 if len(list(filter(r.match, x[methode])))>0 else x.countKey, axis=1 )

    sitemap_urls = open('sitemap_urls.dat', 'r').read().splitlines()
    # Convert numerical column to integer
    sitemap_layers.counts = sitemap_layers.counts.apply(int)
    print('Loaded {:,} rows of categorized data from sitemap_layers.csv'\
            .format(len(sitemap_layers)))

    print('Building %d layer deep sitemap graph' % graph_depth)

    

    nodes, edges = make_sitemap_graph(sitemap_layers, layers=graph_depth,
                            limit=limit, size=size, output_format=output_format, skip=skip, inclEndpoint=inclEndpoint, methodeInterpretation = methode, keywordList = keywordList)
    # f = apply_style(f, style=style, title=title)

    # f.render(cleanup=True)
    config = Config(width=1700,
                height=500, 
                directed=True,
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6", # or "blue"
                collapsible=True,
                graphviz_layout='dot', #'layout',['dot', 'neato', 'circo', 'fdp','sfdp']
                graphviz_config={"rankdir": 'LR', "ranksep": 0, "nodesep": 0}, #"rankdir", ['BT', 'TB', 'LR', 'RL']
                node={'labelProperty':'label', 'renderLabel': True},
                link={'labelProperty': 'label', 'renderLabel': True}
                # **kwargs e.g. node_size=1000 or node_color="blue"
                ) 
    return_value = agraph(nodes=nodes, 
                      edges=edges, 
                      config=config)
    # print('Exported graph to sitemap_graph_%d_layer.%s' % (graph_depth, output_format))

    # f_json = dot_to_json(f.body)
    # with open('json_data.json', 'w') as outfile:
    #     outfile.write(f_json)

    statistics = pd.read_csv('dftopics.csv', dtype=str)
    statistics[pd.isna(statistics)]='[x]'
    statistics = listinCSV(statistics)

    pd.options.display.float_format = '{:.2%}'.format
    dfstatistics = pd.DataFrame(statistics)
    st.table(dfstatistics.loc[:, :'unique topics'])


if __name__ == '__main__':
    main()
