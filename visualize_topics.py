# '''
#    
#     
# '''
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_tags import st_tags_sidebar
import colorsys
import math
from ast import literal_eval
import re
st.set_page_config(layout="wide")
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os.path
from os import path
import faiss
from collections import defaultdict
from collections import Counter
from sklearn import preprocessing
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


def listinCSV(dfWithLists):
  for col in dfWithLists:
    if isinstance(dfWithLists[col][0], str):
      if dfWithLists[col][0][0] == "[":
      #if col != 'Unnamed: 0'and col != 'link'and col != 'HtmlBody'and col != 'content_merge':
        dfWithLists[col]=dfWithLists[col].apply(literal_eval)
  return dfWithLists

siteinfo = pd.read_csv(r'C:\Users\Max\Documents\Visual Studio Code\Uman\sitemap-visualization-tool\dfresult.csv', dtype=str)
siteinfo.columns = siteinfo.columns.str.replace('/', '_')
siteinfo = listinCSV(siteinfo)
ListURLS = siteinfo['link']
TopicList_full = siteinfo['ml6team_keyphrase-extraction-distilbert-openkp']
TopicList_fullConcat = sum(TopicList_full,[])
TopicList_setCnt =  Counter(TopicList_fullConcat).most_common()
TopicList = [a_tuple[0] for a_tuple in TopicList_setCnt]



# # Set global variables
# Query = st_tags_sidebar(
#                     label='Enter Query:',
#                     text= 'Press enter to add more',
#                     suggestions=Counter(TopicList_fullConcat).most_common(1),
#                     maxtags= 1
#                     )


@st.cache(allow_output_mutation=True) # , show_spinner=False)
def load_my_model():
    logger.info("Start model load")
    modelName_1 = "all-mpnet-base-v2" #@param ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2']{allow-input: true, type:"string"}
    model_1 = SentenceTransformer(modelName_1)
    modelName_2 = "paraphrase-MiniLM-L6-v2" #@param ['paraphrase-MiniLM-L6-v2', 'all-mpnet-base-v2']{allow-input: true, type:"string"}
    model_2 = SentenceTransformer(modelName_2)
    logger.info("End model load")
    return model_1, model_2, modelName_1, modelName_2

@st.cache(allow_output_mutation=True) # , show_spinner=False)
def load_emeddings(btnEmbed, model_1, model_2, modelName_1, modelName_2, TopicList):
    logger.info("Start emeddings load")
    pathEmbedding = '.\\embeddings\\'
    savefileEmbeddings = pathEmbedding + 'topicsDict.npy'
    if btnEmbed:
        topics_FullRange_1 = model_1.encode(TopicList)
        topics_FullRange_2 = model_2.encode(TopicList)
        topics_embeddings_1  = preprocessing.minmax_scale(topics_FullRange_1.T).T
        topics_embeddings_2 = preprocessing.minmax_scale(topics_FullRange_2.T).T

        dict_embeddings = {modelName_1: topics_embeddings_1,  modelName_2: topics_embeddings_2, 'topics': TopicList}

        os.makedirs(pathEmbedding, exist_ok=True) 
        np.save(savefileEmbeddings,  dict_embeddings)

        sidebarWrite = "Embeddings made and saved"
        logger.info("Embeddings made and saved")
    elif path.exists(savefileEmbeddings):
        dict_embeddings = np.load(savefileEmbeddings, allow_pickle=True)
        topics_embeddings_1 = dict_embeddings[()][modelName_1]
        topics_embeddings_2 = dict_embeddings[()][modelName_2]
        # TopicList = dict_embeddings[()]['topics']D_12
        sidebarWrite = "Embeddings loaded"
        logger.info("Embeddings loaded from local drive")
    else:
        topics_FullRange_1 = model_1.encode(TopicList)
        topics_FullRange_2 = model_2.encode(TopicList)
        topics_embeddings_1  = preprocessing.minmax_scale(topics_FullRange_1.T).T
        topics_embeddings_2 = preprocessing.minmax_scale(topics_FullRange_2.T).T
        sidebarWrite = "Embeddings calculated"
        logger.info("Embeddings made without saving")
    logger.info("Stop emeddings load")
    return sidebarWrite, topics_embeddings_1, topics_embeddings_2, dict_embeddings

def make_sitemap_graph(Query, Topics, Distances):
    nodes = []
    edges = []
    hueRange = range(0,360,math.floor(360/(len(Query)+1)))

    # Initialize graph
    # f = graphviz.Digraph('sitemap', filename='sitemap_graph_%d_layer' % layers, format='%s' % output_format)
    # f.body.extend(['rankdir=LR', 'size="%s"' % size])


    def add_branch(existingNodes, names, vals,color,connect_to=''):
        ''' Adds a set of nodes and edges to nodes on the previous layer. '''

        # Get the currently existing node names
        node_names = [existingNodes[i].id for i in range(len(existingNodes))]

        vals_scaled = preprocessing.minmax_scale(vals)
        # Only add a new branch it it will connect to a previously created node
        if connect_to:
            if connect_to in node_names:
                for name, val, satVal in list(zip(names, vals, vals_scaled)):

                    fraqVal=1-satVal
                    s = 2*math.log(1+fraqVal,2)/(1+math.log(1+fraqVal,2))
                    (r, g, b) = colorsys.hsv_to_rgb(color/360,s, 82/100)
                    hexColor  = '%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

                    nodes.append( Node(id='%s-%s' % (connect_to, name), 
                            label=name, 
                            size=200,
                            color="#{}".format(hexColor),
                            symbolType="circle",
                                    ) 
                    ) 

                    # f.edge(connect_to, '%s-%s' % (connect_to, name), label=)
                    edges.append( Edge(source=connect_to, 
                                label= u'Δ: {:.2f}'.format(val), 
                                target='%s-%s' % (connect_to, name),
                                labelPosition="center",
                                type="CURVE_SMOOTH") 
                    )
                    # else:
                    #     print(name)


    # Add the first layer of nodes
    for iq, xq in enumerate(Query):

        # Add the first layer of nodes
        (r, g, b) = colorsys.hsv_to_rgb(hueRange[iq]/360, 1 , 82/100)
        hexColor  = '%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

        nodes.append( 
            Node(id=xq, 
                label='{}'.format(xq), 
                size=200,
                color="#{}".format(hexColor),
                symbolType="square",
            ) 
        ) 

        add_branch(nodes,
                names=Topics[iq],
                vals=Distances[iq],
                color=hueRange[iq],
                connect_to='-'.join(['%s']) % xq
                )


    return nodes, edges

def main():
    
    model_1, model_2, modelName_1, modelName_2 = load_my_model() # 1='all-mpnet-base-v2', 2= 'paraphrase-MiniLM-L6-v2' 

    btnEmbed = st.sidebar.button('Make Embeddings')
    sidebarWrite , topics_embeddings_1, topics_embeddings_2, dict_embeddings = load_emeddings(btnEmbed, model_1, model_2, modelName_1, modelName_2, TopicList)
    st.sidebar.write(sidebarWrite + '\n')

    d_1=topics_embeddings_1.shape[1]
    index_1 = faiss.IndexFlatL2(d_1)
    index_1.add(topics_embeddings_1)
    
    d_2=topics_embeddings_2.shape[1]
    index_2 = faiss.IndexFlatL2(d_2)
    index_2.add(topics_embeddings_2)

    #@markdown # Search # nearest
    st.sidebar.write('# Maximum # Nearest Neighbors')
    k_NN = st.sidebar.number_input(label='', min_value=1, max_value=min(d_1,d_2), value=200, step=10)
    st.sidebar.write('# Maximum distance')
    max_d = st.sidebar.slider(label='', value=14, min_value=0, max_value=30)

    #@markdown # Search query
    #query =  "Food" #@param ["SAP", "AI", "Compliance Manager", "Food", "Soil"]{allow-input: true, type:"string"}
    #queryList = ['SAP', 'FAST', 'ERP', 'Azure', 'Data Migration', 'Service Management']
    queryList = st_tags_sidebar(
                    label='# Enter Keywords:',
                    text='Press enter to add more',
                    value=[word for word, _ in Counter(TopicList_fullConcat).most_common(3)],
                    suggestions=TopicList)


    IList_1 = []
    DList_1 = []
    URLList_1 = []
    d_1 = defaultdict(list)
    topicSelection_1 = []
    DSelList_1 = []

    IList_2 = []
    DList_2 = []
    URLList_2 = []
    d_2 = defaultdict(list)
    topicSelection_2 = []
    DSelList_2 = []

    for query in queryList:
        xq_1_w = model_1.encode([query])
        xq_1 = preprocessing.minmax_scale(xq_1_w.T).T
        D_1, I_1 = index_1.search(xq_1, k_NN)
        IList_1.append(I_1[0])
        DList_1.append(D_1[0])
        
        tupleDist_1 = [ (TopicList[i], d)  for i,d in zip(I_1[0],D_1[0]) if d<=max_d]
        [topicSel_1, distSel_1] = list(map(list, zip(*tupleDist_1)))
        DSelList_1.append(distSel_1)
        ListTopicUrls_1 = [(topic,ListURLS[iURL]) for topic in topicSel_1 for iURL, URLTopicList in enumerate(TopicList_full) if topic in URLTopicList ]

        d_1 = defaultdict(list)
        for k, v in ListTopicUrls_1:
            d_1[k].append(v)
        URLList_1.append(list(d_1.values()))
        topicSelection_1.append(list(d_1.keys()))



        xq_2_w = model_2.encode([query])
        xq_2 = preprocessing.minmax_scale(xq_2_w.T).T
        D_2, I_2 = index_2.search(xq_2, k_NN)
        IList_2.append(I_2[0])
        DList_2.append(D_2[0])
        tupleDist_2 = [ (TopicList[i], d) for i,d in zip(I_2[0],D_2[0]) if d<=max_d]
        [topicSel_2, distSel_2] = list(map(list, zip(*tupleDist_2)))
        DSelList_2.append(distSel_2)
        ListTopicUrls_2 = [(topic,ListURLS[iURL]) for topic in topicSel_2 for iURL, URLTopicList in enumerate(TopicList_full) if topic in URLTopicList ]

        d_2 = defaultdict(list)
        for k, v in ListTopicUrls_2:
            d_2[k].append(v)
        URLList_2.append(list(d_2.values()))
        topicSelection_2.append(list(d_2.keys()))

    dfM1 = pd.DataFrame(list(zip(queryList,topicSelection_1,DSelList_1, URLList_1)),
               columns =['Query', 'Topics', 'Distances', 'Reference URL'])
    dfM2 = pd.DataFrame(list(zip(queryList,topicSelection_2, DSelList_2,URLList_2)),
               columns =['Query', 'Topics','Distances', 'Reference URL'])

    nodes_1, edges_1 = make_sitemap_graph(dfM1['Query'], dfM1['Topics'], dfM1['Distances'])
    nodes_2, edges_2 = make_sitemap_graph(dfM2['Query'], dfM2['Topics'], dfM2['Distances'])
    config = Config(width=1700, 
                height=1000,
                graphviz_layout='fdp', #'layout',['dot', 'neato', 'circo', 'fdp','sfdp']
                graphviz_config={"rankdir": 'LR', "ranksep": 0, "nodesep": 0}, #"rankdir", ['BT', 'TB', 'LR', 'RL']
                directed=True,
                nodeHighlightBehavior=True, 
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty':'label'},
                link={'labelProperty': 'label', 'renderLabel': True},
                maxZoom=2,
                minZoom=0.1,
                staticGraphWithDragAndDrop=False,
                staticGraph=True,
                initialZoom=1
                ) 
    return_value = agraph(nodes=nodes_1, 
                      edges=edges_1, 
                      config=config)

    return_value = agraph(nodes=nodes_2, 
                      edges=edges_2, 
                      config=config)

    st.write('***')
 

    st.write("Model comparison table")
    dfCompareM1M2 = pd.DataFrame(list(zip(queryList,topicSelection_1,topicSelection_2)),
               columns =['Query', 'Topics '+modelName_1 , 'Topics '+modelName_2 ])
    if not dfCompareM1M2.empty: st.dataframe(dfCompareM1M2) 


    st.write('\n Topics '+modelName_1)
    if not dfM1.empty: st.dataframe(dfM1)

    st.write('\n Topics '+modelName_2)
    if not dfM2.empty: st.dataframe(dfM2)

    st.write('---')
    st.write('\n Topics '+modelName_1)
    st.json(dfM1.to_json(orient="records"),expanded=False)
    st.write('***')
    st.write('\n Topics '+modelName_2)
    st.json(dfM2.to_json(orient="records"),expanded=False)
    st.write('***')
    st.write('All topics found on site, sorted by frequency')
    st.table(data=TopicList_setCnt)




    #------------------------------------------------------------------------------------------------------------------

    # # Read in categorized data
    # sitemap_layers_raw = pd.read_csv('sitemap_layers.csv', dtype=str)
    # sitemap_layers_raw.columns = sitemap_layers_raw.columns.str.replace('/', '_')
    

    # maxi=0
    # for i in list(sitemap_layers_raw.columns):
    #     try:
    #         if int(i)> maxi:
    #             maxi=int(i)
    #     except:
    #         pass
    # graph_depth = st.slider('Graph depth', value=3, min_value=0, max_value=maxi, key="slider")  # Number of layers deep to plot categorization

    # tags = pd.read_csv('dfresult.csv', dtype=str)
    # tags.columns = tags.columns.str.replace('/', '_')
    # tags[pd.isna(tags)]="['x']"
    # tags = listinCSV(tags)


    # filterTags = tags.loc[:, 'KeyBert':]
    # sitemap_layers = pd.concat([sitemap_layers_raw, filterTags], axis=1)
    # sitemap_layers = sitemap_layers.applymap(lambda s: s.lower() if type(s) == str else s)
    # MethodeOption = list(filterTags.columns)
    # MethodeOption.insert(0,'Just Count')
    # methode = st.selectbox('What model would you like to use?',MethodeOption)
    # keywordList = []
    # sitemap_layers['countKey'] =0
    # if methode != 'Just Count':
    #     keywordList = st_tags_sidebar(
    #                 label='# Enter Keywords:',
    #                 text='Press enter to add more',
    #                 maxtags: 1,
    #                 suggestions=sum(filterTags[methode],[])
    #                 )
    #     dropDown = st.sidebar.selectbox(
    #                 'suggestions tags:',
    #                 sum(filterTags[methode],[]))
    #     for keyFilter in keywordList:
    #         r = re.compile(keyFilter.lower())
    #         #sitemap_layers['countKey']=sitemap_layers.apply(lambda x : x.countKey+1 if keyFilter in x[methode] else x.countKey, axis=1 )
    #         sitemap_layers['countKey']=sitemap_layers.apply(lambda x : x.countKey+1 if len(list(filter(r.match, x[methode])))>0 else x.countKey, axis=1 )

    # sitemap_urls = open('sitemap_urls.dat', 'r').read().splitlines()
    # # Convert numerical column to integer
    # sitemap_layers.counts = sitemap_layers.counts.apply(int)
    # print('Loaded {:,} rows of categorized data from sitemap_layers.csv'\
    #         .format(len(sitemap_layers)))

    # print('Building %d layer deep sitemap graph' % graph_depth)

    

    # nodes, edges = make_sitemap_graph(sitemap_layers, layers=graph_depth,
    #                         limit=limit, size=size, output_format=output_format, skip=skip, inclEndpoint=inclEndpoint, methodeInterpretation = methode, keywordList = keywordList)
    # # f = apply_style(f, style=style, title=title)

    # # f.render(cleanup=True)
    # config = Config(width=1700,
    #             height=500, 
    #             directed=True,
    #             nodeHighlightBehavior=True, 
    #             highlightColor="#F7A7A6", # or "blue"
    #             collapsible=True,
    #             graphviz_layout='dot', #'layout',['dot', 'neato', 'circo', 'fdp','sfdp']
    #             graphviz_config={"rankdir": 'LR', "ranksep": 0, "nodesep": 0}, #"rankdir", ['BT', 'TB', 'LR', 'RL']
    #             node={'labelProperty':'label', 'renderLabel': True},
    #             link={'labelProperty': 'label', 'renderLabel': True}
    #             # **kwargs e.g. node_size=1000 or node_color="blue"
    #             ) 
    # return_value = agraph(nodes=nodes, 
    #                   edges=edges, 
    #                   config=config)
    # # print('Exported graph to sitemap_graph_%d_layer.%s' % (graph_depth, output_format))

    # # f_json = dot_to_json(f.body)
    # # with open('json_data.json', 'w') as outfile:
    # #     outfile.write(f_json)

    # statistics = pd.read_csv('dftopics.csv', dtype=str)
    # statistics[pd.isna(statistics)]='[x]'
    # statistics = listinCSV(statistics)

    # pd.options.display.float_format = '{:.2%}'.format
    # dfstatistics = pd.DataFrame(statistics)
    # st.table(dfstatistics.loc[:, :'unique topics'])


if __name__ == '__main__':
    main()
