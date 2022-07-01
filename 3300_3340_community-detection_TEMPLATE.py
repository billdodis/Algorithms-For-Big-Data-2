import numpy as np
import pandas as pd
import random
import networkx as nx
import itertools
from networkx.algorithms.community import girvan_newman
from tqdm import tqdm
import time
import math
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm
from networkx.algorithms import community as comm
from networkx.algorithms import centrality as cntr
import os
##################################### SOS #####################################
# FILES HAVE TO BE IN THE SAME                                                #
# DIRECTORY AS THIS EXECUTABLE                                                #
# PYTHON FILE IN ORDER TO                                                     #
# RUN PROPERLY                                                                #
# CHECK LIBRARIES TOO!                                                        #
##################################### SOS #####################################
############################### FG COLOR DEFINITIONS ###############################
class bcolors:
    HEADER      = '\033[95m'
    OKBLUE      = '\033[94m'
    OKCYAN      = '\033[96m'
    OKGREEN     = '\033[92m'
    WARNING     = '\033[93m'
    FAIL        = '\033[91m'
    ENDC        = '\033[0m'    # RECOVERS DEFAULT TEXT COLOR
    BOLD        = '\033[1m'
    UNDERLINE   = '\033[4m'

    def disable(self):
        self.HEADER     = ''
        self.OKBLUE     = ''
        self.OKGREEN    = ''
        self.WARNING    = ''
        self.FAIL       = ''
        self.ENDC       = ''

########################################################################################
############################## MY ROUTINES LIBRARY STARTS ##############################
########################################################################################

# SIMPLE ROUTINE TO CLEAR SCREEN BEFORE SHOWING A MENU...
def my_clear_screen():

    os.system('cls' if os.name == 'nt' else 'clear')

# CREATE A LIST OF RANDOMLY CHOSEN COLORS...
def my_random_color_list_generator(REQUIRED_NUM_COLORS):

    my_color_list = [   'red',
                        'green',
                        'cyan',
                        'brown',
                        'olive',
                        'orange',
                        'darkblue',
                        'purple',
                        'yellow',
                        'hotpink',
                        'teal',
                        'gold']

    my_used_colors_dict = { c:0 for c in my_color_list }     # DICTIONARY OF FLAGS FOR COLOR USAGE. Initially no color is used...
    constructed_color_list = []

    if REQUIRED_NUM_COLORS <= len(my_color_list):
        for i in range(REQUIRED_NUM_COLORS):
            constructed_color_list.append(my_color_list[i])
        
    else: # REQUIRED_NUM_COLORS > len(my_color_list)   
        constructed_color_list = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(REQUIRED_NUM_COLORS)]
 
    return(constructed_color_list)


# VISUALISE A GRAPH WITH COLORED NODES AND LINKS
def my_graph_plot_routine(G,fb_nodes_colors,fb_links_colors,fb_links_styles,graph_layout,node_positions,flagg):
    plt.figure(figsize=(10,10))
    
    if len(node_positions) == 0:
        if graph_layout == 'circular':
            node_positions = nx.circular_layout(G)
        elif graph_layout == 'random':
            node_positions = nx.random_layout(G, seed=50)
        elif graph_layout == 'planar':
            node_positions = nx.planar_layout(G)
        elif graph_layout == 'shell':
            node_positions = nx.shell_layout(G)
        else:   #DEFAULT VALUE == spring
            node_positions = nx.spring_layout(G)

    nx.draw(G, 
        with_labels=True,           # indicator variable for showing the nodes' ID-labels
        style=fb_links_styles,      # edge-list of link styles, or a single default style for all edges
        edge_color=fb_links_colors, # edge-list of link colors, or a single default color for all edges
        pos = node_positions,       # node-indexed dictionary, with position-values of the nodes in the plane
        node_color=fb_nodes_colors, # either a node-list of colors, or a single default color for all nodes
        node_size = 100,            # node-circle radius
        alpha = 0.9,                # fill-transparency 
        width = 0.5                 # edge-width
        )
    if flagg == True:
        plt.show()

    return(node_positions)


########################################################################################
# MENU 1 STARTS: creation of input graph ### 
########################################################################################
def my_menu_graph_construction(G,node_names_list,node_positions):

    my_clear_screen()

    breakWhileLoop  = False

    while not breakWhileLoop:
        print(bcolors.OKGREEN 
        + '''
========================================
(1.1) Create graph from fb-food data set (fb-pages-food.nodes and fb-pages-food.nodes)\t[format: L,<NUM_LINKS>]
(1.2) Create RANDOM Erdos-Renyi graph G(n,p).\t\t\t\t\t\t[format: R,<number of nodes>,<edge probability>]
(1.3) Print graph\t\t\t\t\t\t\t\t\t[format: P,<GRAPH LAYOUT in {spring,random,circular,shell }>]    
(1.4) Continue with detection of communities.\t\t\t\t\t\t[format: N]
(1.5) EXIT\t\t\t\t\t\t\t\t\t\t[format: E]
----------------------------------------
        ''' + bcolors.ENDC)
        
        my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')
        
        if my_option_list[0] == 'L':
            MAX_NUM_LINKS = 2102    # this is the maximum number of links in the fb-food-graph data set...

            if len(my_option_list) > 2:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Too many parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            

            else:
                if len(my_option_list) == 1:
                    NUM_LINKS = MAX_NUM_LINKS
                else: #...len(my_option_list) == 2...
                    NUM_LINKS = int(my_option_list[1])

                if NUM_LINKS > MAX_NUM_LINKS or NUM_LINKS < 1:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR Invalid number of links to read from data set. It should be in {1,2,...,2102}. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)            
                else:
                    # LOAD GRAPH FROM DATA SET...
                    G, node_names_list = _3300_3340_read_graph_from_csv(NUM_LINKS)
                    print(  "\tConstructing the FB-FOOD graph with n =",G.number_of_nodes(),
                            "vertices and m =",G.number_of_edges(),"edges (after removal of loops).")

        elif my_option_list[0] == 'R':

            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            
            else: # ...len(my_option_list) <= 3...
                if len(my_option_list) == 1:
                    NUM_NODES = 100                     # DEFAULT NUMBER OF NODES FOR THE RANDOM GRAPH...
                    ER_EDGE_PROBABILITY = 2 / NUM_NODES # DEFAULT VALUE FOR ER_EDGE_PROBABILITY...

                elif len(my_option_list) == 2:
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = 2 / max(1,NUM_NODES) # AVOID DIVISION WITH ZERO...

                else: # ...NUM_NODES == 3...
                    NUM_NODES = int(my_option_list[1])
                    ER_EDGE_PROBABILITY = float(my_option_list[2])

                if ER_EDGE_PROBABILITY < 0 or ER_EDGE_PROBABILITY > 1 or NUM_NODES < 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Invalid probability mass or number of nodes of G(n,p). Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    G = nx.erdos_renyi_graph(NUM_NODES, ER_EDGE_PROBABILITY)
                    print(bcolors.ENDC +    "\tConstructing random Erdos-Renyi graph with n =",G.number_of_nodes(),
                                            "vertices and edge probability p =",ER_EDGE_PROBABILITY,
                                            "which resulted in m =",G.number_of_edges(),"edges.")

                    node_names_list = [ x for x in range(NUM_NODES) ]

        elif my_option_list[0] == 'P':                  # PLOT G...
            print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
            if len(my_option_list) > 3:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
            else:
                if len(my_option_list) <= 1:
                    graph_layout = 'spring'     # ...DEFAULT graph_layout value...
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                elif len(my_option_list) == 2: 
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = 'Y'  # ...DEFAULT decision: erase node_positions...

                else: # ...len(my_option_list) == 3...
                    graph_layout = str(my_option_list[1])
                    reset_node_positions = str(my_option_list[2])

                if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                elif reset_node_positions not in ['Y','y','N','n']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible decision for resetting node positions. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if reset_node_positions in ['y','Y']:
                        node_positions = []         # ...ERASE previous node positions...

                    node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions,True)

        elif my_option_list[0] == 'N':
            NUM_NODES = G.number_of_nodes()
            if NUM_NODES == 0:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: You have not yet constructed a graph to work with. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            else:
                my_clear_screen()
                breakWhileLoop = True
            
        elif my_option_list[0] == 'E':
            quit()

        else:
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

    return(G,node_names_list,node_positions)

########################################################################################
# MENU 2: detect communities in the constructed graph 
########################################################################################
def my_menu_community_detection(G,node_names_list,node_positions,hierarchy_of_community_tuples):

    breakWhileLoop = False

    while not breakWhileLoop:
            print(bcolors.OKGREEN 
                + '''
========================================
(2.1) Add random edges from each node\t\t\t[format: RE,<NUM_RANDOM_EDGES_PER_NODE>,<EDGE_ADDITION_PROBABILITY in [0,1]>]
(2.2) Add hamilton cycle (if graph is not connected)\t[format: H]
(2.3) Print graph\t\t\t\t\t[format: P,<GRAPH LAYOUT in { spring, random, circular, shell }>,<ERASE NODE POSITIONS in {Y,N}>]
(2.4) Compute communities with GIRVAN-NEWMAN\t\t[format: C,<ALG CHOICE in { O(wn),N(etworkx) }>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.5) Compute a binary hierarchy of communities\t\t[format: D,<NUM_DIVISIONS>,<GRAPH LAYOUT in {spring,random,circular,shell }>]
(2.6) Compute modularity-values for all community partitions\t[format: M]
(2.7) Visualize the communities of the graph\t\t[format: V,<GRAPH LAYOUT in {spring,random,circular,shell}>]
(2.8) EXIT\t\t\t\t\t\t[format: E]
----------------------------------------
            ''' + bcolors.ENDC)

            my_option_list = str(input('\tProvide your (case-sensitive) option: ')).split(',')

            if my_option_list[0] == 'RE':                    # 2.1: ADD RANDOM EDGES TO NODES...

                if len(my_option_list) > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. [format: D,<NUM_RANDOM_EDGES>,<EDGE_ADDITION_PROBABILITY>]. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if len(my_option_list) == 1:
                        NUM_RANDOM_EDGES = 1                # DEFAULT NUMBER OF RANDOM EDGES TO ADD (per node)
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    elif len(my_option_list) == 2:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = 0.25    # DEFAULT PROBABILITY FOR ADDING EACH RANDOM EDGE (independently of other edges) FROM EAC NODE (independently from other nodes)...

                    else:
                        NUM_RANDOM_EDGES = int(my_option_list[1])
                        EDGE_ADDITION_PROBABILITY = float(my_option_list[2])
            
                    # CHECK APPROPIATENESS OF INPUT AND RUN THE ROUTINE...
                    if NUM_RANDOM_EDGES-1 not in range(5):
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Too many random edges requested. Should be from {1,2,...,5}. Try again..." + bcolors.ENDC) 
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif EDGE_ADDITION_PROBABILITY < 0 or EDGE_ADDITION_PROBABILITY > 1:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Not appropriate value was given for EDGE_ADDITION PROBABILITY. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else: 
                        G = _3300_3340_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)

            elif my_option_list[0] == 'H':                  #2.2: ADD HAMILTON CYCLE...

                    G = _3300_3340_add_hamilton_cycle_to_graph(G,node_names_list)

            elif my_option_list[0] == 'P':                  # 2.3: PLOT G...
                print("Printing graph G with",G.number_of_nodes(),"vertices,",G.number_of_edges(),"edges and",nx.number_connected_components(G),"connected components." )
                
                if len(my_option_list) > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:
                    if len(my_option_list) <= 1:
                        graph_layout = 'spring'     # ...DEFAULT graph_layout value...

                    else: # ...len(my_option_list) == 2... 
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                            print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice for graph layout. Try again..." + bcolors.ENDC)
                            print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        if len(my_option_list) == 2:
                            node_positions = []         # ...ERASE previous node positions...

                        node_positions = my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions,True)

            elif my_option_list[0] == 'C':      # 2.4: COMPUTE ONE-SHOT GN-COMMUNITIES
                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        alg_choice  = 'N'            # DEFAULT COMM-DETECTION ALGORITHM == NX_GN
                        graph_layout = 'spring'     # DEFAULT graph layout == spring

                    elif NUM_OPTIONS == 2:
                        alg_choice  = str(my_option_list[1])
                        graph_layout = 'spring'     # DEFAULT graph layout == spring
                
                    else: # ...NUM_OPTIONS == 3...
                        alg_choice      = str(my_option_list[1])
                        graph_layout    = str(my_option_list[2])

                    # CHECKING CORRECTNESS OF GIVWEN PARAMETERS...
                    if alg_choice == 'N' and graph_layout in ['spring','circular','random','shell']:
                        G = _3300_3340_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)

                    elif alg_choice == 'O'and graph_layout in ['spring','circular','random','shell']:
                        G = _3300_3340_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)

                    else:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible parameters for executing the GN-algorithm. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

            elif my_option_list[0] == 'D':          # 2.5: COMUTE A BINARY HIERARCHY OF COMMUNITY PARRTITIONS
                NUM_OPTIONS = len(my_option_list)
                NUM_NODES = G.number_of_nodes()
                NUM_COMPONENTS = nx.number_connected_components(G)
                MAX_NUM_DIVISIONS = min( 8*NUM_COMPONENTS , np.floor(NUM_NODES/4) )

                if NUM_OPTIONS > 3:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                else:
                    if NUM_OPTIONS == 1:
                        number_of_divisions = 2*NUM_COMPONENTS      # DEFAULT number of communities to look for 
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                        
                    elif NUM_OPTIONS == 2:
                        number_of_divisions = int(my_option_list[1])
                        num_divisions = number_of_divisions
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: #...NUM_OPTIONS == 3...
                        number_of_divisions = int(my_option_list[1])
                        num_divisions = number_of_divisions
                        graph_layout = str(my_option_list[2])

                    # CHECKING SYNTAX OF GIVEN PARAMETERS...
                    if number_of_divisions < NUM_COMPONENTS or number_of_divisions > MAX_NUM_DIVISIONS:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: The graph has already",NUM_COMPONENTS,"connected components." + bcolors.ENDC)
                        print(bcolors.WARNING + "\tProvide a number of divisions in { ",NUM_COMPONENTS,",",MAX_NUM_DIVISIONS,"}. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    elif graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        percentage = str(input('\tInsert the percentage of the nodes to be involved in the computation of the BC values. If you want them all involved, insert 100(for 100%).\n'))
                        percentage = int(percentage)
                        G = _3300_3340_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions,percentage)

            elif my_option_list[0] == 'M':      # 2.6: DETERMINE PARTITION OF MIN-MODULARITY, FOR A GIVEN BINARY HIERARCHY OF COMMUNITY PARTITIONS
                _3300_3340_determine_opt_community_structure(G,hierarchy_of_community_tuples)


            elif my_option_list[0] == 'V':      # 2.7: VISUALIZE COMMUNITIES WITHIN GRAPH

                NUM_OPTIONS = len(my_option_list)

                if NUM_OPTIONS > 2:
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                    print(bcolors.WARNING + "\tERROR MESSAGE: Wrong number of parameters. Try again..." + bcolors.ENDC)
                    print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                
                else:

                    if NUM_OPTIONS == 1:
                        graph_layout = 'spring'                     # DEFAULT graph layout == spring
                    
                    else: # ...NUM_OPTIONS == 2...
                        graph_layout = str(my_option_list[1])

                    if graph_layout not in ['spring','random','circular','shell']:
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                        print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible choice of a graph layout. Try again..." + bcolors.ENDC)
                        print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)

                    else:
                        _3300_3340_visualize_communities(G,community_tuples,graph_layout,node_positions)

            elif my_option_list[0] == 'E':
                #EXIT the program execution...
                quit()

            else:
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
                print(bcolors.WARNING + "\tERROR MESSAGE: Incomprehensible input was provided. Try again..." + bcolors.ENDC)
                print(bcolors.WARNING + "\t++++++++++++++++++++++++++++++++++++++++" + bcolors.ENDC)
    ### MENU 2 ENDS: detect communities in the constructed graph ### 

########################################################################################
############################### MY ROUTINES LIBRARY ENDS ############################### 
########################################################################################

########################################################################################
########################## STUDENT_AM ROUTINES LIBRARY STARTS ##########################
# FILL IN THE REQUIRED ROUTINES FROM THAT POINT ON...
########################################################################################

########################################################################################
def _3300_3340_read_graph_from_csv(MAX_NUM_LINKS):

    print(bcolors.ENDC + "\t" + '''
        ########################################################################################
        # CREATE GRAPH FROM EDGE_CSV DATA 
        # ...(if needed) load all details for the nodes to the fb_nodes DATAFRAME
        # ...create DATAFRAME edges_df with the first MAX_NUM_LINKS, from the edges dataset
        # ...edges_df has one row per edge, and two columns, named 'node_1' and 'node_2
        # ...CLEANUP: remove each link (from the loaded ones) which is a LOOP
        # ...create node_names_list of NODE IDs that appear as terminal points of the loaded edges
        # ...create graph from the edges_df dataframe
        # ...return the constructed graph
        ########################################################################################
''')

    print(bcolors.ENDC + "\tUSEFUL FUNCTIONS:")

    print(  bcolors.ENDC + "\t\t The routine " 
            + bcolors.OKCYAN + "nx.from_pandas_edgelist(...) "
            + bcolors.ENDC 
            + '''creates the graph from a dataframe representing its edges,
                 one per edge, in two columns representing the tail-node 
                 (node_1) and the head-node (node_2) of the edge.\n''')
    # taking data for nodes in a dataframe with read_csv command of pandas.
    # saving it in fb_links variable.

    fb_links = pd.read_csv("fb-pages-food.edges")
    fb_links_rows = fb_links.shape[0]
    node1list = []
    node2list = []
    uniquevalues = []  # needed for coloring the nodes.
    for i in range(0, MAX_NUM_LINKS):
        values = fb_links.values[i]
        node1list.append(values[0])
        node2list.append(values[1])
        if values[0] not in uniquevalues:
            uniquevalues.append(values[0])
            node_names_list.append(values[0])
            neighborhoodarray.append([])
        if values[1] not in uniquevalues:
            uniquevalues.append(values[1])
            node_names_list.append(values[1])
            neighborhoodarray.append([])
        indx = node_names_list.index(values[0])
        indx2 = node_names_list.index(values[1])
        # every element's neighborhood is in the same position as the element is in nodenameslist array
        # so, nodenameslist[i] has neighbours the elements of neighborhoodarray[i]
        # neighborhoodarray is an array of arrays.
        neighborhoodarray[indx].append(values[1])
        neighborhoodarray[indx2].append(values[0])
    # those 2 if conditions help us carry the length of the unique nodes. We need this length for coloring the graph.

    # creating the new dataframe with the wanted rows only.
    fb_links_df = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
    G = nx.from_pandas_edgelist(fb_links_df, "node_1", "node_2", create_using=nx.Graph())
    # 'circular':
    # 'random':
    # 'planar':
    # 'shell':
    # DEFAULT VALUE == spring
    print('a. Circular,\nb. Random,\nc. Planar,\nd. Spring.')
    layout = input('Please select one layout for the graph. Enter a,b,c or d :\n')
    while layout not in 'abcde':
        layout = input('Wrong input! Enter a,b,c or d :\n')
    if layout == 'a':
        layout = 'circular'
    elif layout == 'b':
        layout = 'random'
    elif layout == 'c':
        layout = 'shell'
    else:
        layout = 'spring'
    print('nodes', node_names_list)
    print('neighbors', neighborhoodarray)
    my_graph_plot_routine(G, my_random_color_list_generator(len(uniquevalues)),
                          my_random_color_list_generator(MAX_NUM_LINKS), '-', layout, [],True)
    return G, node_names_list

######################################################################################################################
# ...(a) STUDENT_AM IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def _3300_3340_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions):

    print(  bcolors.ENDC 
            + "\tCalling routine " 
            + bcolors.HEADER + "_3300_3340_one_shot_girvan_newman_for_communities(G,graph_layout,node_positions)\n"
            + bcolors.ENDC)

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # PROVIDE YOUR OWN ROUTINE WHICH CREATES K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, AS FOLLOWS:
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        # ...THE K CONNECTED COMPONENTS OF G ARE COMPUTED. 
        # ...THE LIST community_tuples IS INITIALIZED, WITH ONE NODES-TUPLE FOR EACH CONNECTED COMPONENT DEFINING A DIFFERENT COMMUNITY.
        # ...GCC = THE SUBGRAPH OF G INDUCED BY THE NODES OF THE LARGEST COMMUNITY, LC.
        # ...SPLIT LC IN TWO SUBCOMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES FROM GCC, UNTIL ITS DISCONNECTION.
        # ...THE NODE-LISTS OF THE TWO COMPONENTS OF GCC (AFTER THE EDGE REMOVALS) ARE THE NEW SUBCOMMUNITIES THAT SUBSTITUTE LC 
        # IN THE LIST community_tuples, AS SUGGESTED BY THE GIRVAN-NEWMAN ALGORITHM...
        ######################################################################################################################
''')

    start_time = time.time()

    print(bcolors.ENDC  + "\tUSEFUL FUNCTIONS:")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "sorted(nx.connected_components(G), key=len, reverse=True) " + bcolors.ENDC 
                        + "initiates the community_tuples with the node-sets of the connected components of the graph G, sorted by their size")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "G.subgraph(X).copy() " 
                                    + bcolors.ENDC      + "creates the subgraph of G induced by a subset (even in list format) X of nodes.")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "edge_betweenness_centrality(...) " 
                                    + bcolors.ENDC      + "of networkx.algorithms.centrality computes edge-betweenness values.")
 
    print(bcolors.ENDC  + "\t\t"    + bcolors.OKCYAN    + "is_connected(G) " 
                                    + bcolors.ENDC      + "of networkx checks connectedness of G.\n")
    compute_coherent_components()
    print('comm:', communities)
    print('Number of our communities', len(communities))
    startinglength = len(communities)
    # keeping communities list, because we clear it in the end of the while loop
    fakecommunities = []
    for i in communities:
        fakecommunities.append(i)
    communitiesnum = len(communities)
    maxlength = 0
    for i in range(len(communities)):
        if len(communities[i]) > maxlength:
            maxlength = len(communities[i])
            gcc = communities[i]
    node1list = []
    node2list = []
    for i in range(len(gcc)):
        index = node_names_list.index(gcc[i])
        for y in range(len(neighborhoodarray[index])):
            node1list.append(gcc[i])
            node2list.append(neighborhoodarray[index][y])
    counter = 0
    while communitiesnum == len(communities):
        counter += 1
        gccdf = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
        G = nx.from_pandas_edgelist(gccdf, "node_1", "node_2", create_using=nx.Graph())
        if counter == 1:
            LCgraph = G
            LCgraphlist.append(G)
            LCs.append(gcc)
        # creating the graph because, to find the BC values, we have to put it as an argument!
        x = nx.edge_betweenness_centrality(G, None, False, None, None)
        max_value = max(x.values())
        max_key = max(x, key=x.get)
        index1 = node_names_list.index(max_key[0])
        index2 = node_names_list.index(max_key[1])
        deletingindex = 0
        flag = False
        for i in range(len(neighborhoodarray[index1])):
            if neighborhoodarray[index1][i] == max_key[1]:
                deletingindex = i
                flag = True
        if flag:
            del neighborhoodarray[index1][deletingindex]
            flag = False
        for i in range(len(neighborhoodarray[index2])):
            if neighborhoodarray[index2][i] == max_key[0]:
                deletingindex = i
                flag = True
        if flag:
            del neighborhoodarray[index2][deletingindex]
        deletedlist = []
        for i in range(len(node1list)):
            if (node1list[i] == max_key[0] and node2list[i] == max_key[1]) or (
                    node1list[i] == max_key[1] and node2list[i] == max_key[0]):
                deletedlist.append(i)
        for i in deletedlist:
            del node1list[i]
            del node2list[i]
        communities.clear()
        compute_coherent_components()
    # WE ARE OUT OF THE WHILE LOOP, SO THE PARTITION HAS BEEN MADE
    # WE HAVE KEPT THE GRAPH OF THE LC ON VARIABLE LCgraph
    # WE WILL FIND MODULARITY FROM THIS GRAPH AND THE LC1 and LC2 (PARTITIONS OF LC)
    LCss = []
    exists = 0
    # with this loop we hold the LC1 and LC2 on the LCss array so we will put them as an argument for finding modularity
    print('communities', communities)
    print('fakecommunities', fakecommunities)
    for fk in range(len(communities)):
        for fk2 in range(len(fakecommunities)):
            if communities[fk] == fakecommunities[fk2]:
                exists = 1
                # if exists 1 it means we had this community before, so this is not a partition.
                # if a community is not the same with a community before the partition
                # this means it is LC1 or LC2.
        if exists == 0:
            LCss.append(communities[fk])
        else:
            exists = 0
    LC1s.append(LCss[0])
    LC2s.append(LCss[1])
    LC1 = set(LCss[0])
    LC2 = set(LCss[1])
    modularity = nx_comm.modularity(LCgraph, [LC1, LC2])
    MODULARITIES.append(modularity)
   
    end_time = time.time()
    # we created a graph for gcc so we could take the BC values we needed.
    # now we will recreate the graph with all the nodes involved.
    G = reconstruct_graph(graph_layout, node_positions, 'GV')
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tYOUR OWN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return G
######################################################################################################################
# ...(b) USE NETWORKX IMPLEMENTATION OF ONE-SHOT GN-ALGORITHM...
######################################################################################################################
def _3300_3340_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_3300_3340_use_nx_girvan_newman_for_communities(G,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # USE THE BUILT-IN ROUTINE OF NETWORKX FOR CREATING K+1 COMMUNITIES FOR A GRAPH WITH K CONNECTED COMPONENTS, WHERE 
        # THE GIANT COMPONENT IS SPLIT IN TO COMMUNITIES, BY REPEATEDLY REMOVING MAX_BETWEENNESS EDGES, AS SUGGESTED BY THE 
        # GIRVAN-NEWMAN ALGORITHM
        # 
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #    community_tuples = a list of K+1 tuples of node_IDs, each tuple determining a different community
        # PSEUDOCODE:    
        #   ...simple...
        ######################################################################################################################
''')

    start_time = time.time()

    print(bcolors.ENDC  + "USEFUL FUNCTIONS:")
    print(  bcolors.ENDC    + "\t\t" + bcolors.OKCYAN + "girvan_newman(...) " 
            + bcolors.ENDC  + "from networkx.algorithms.community, provides the requiured list of K+1 community-tuples.\n")
    comp = girvan_newman(G)
    x = nx.number_connected_components(G)
    print('Connected components: ', x)
    print('Printing the list which holds the communities. Every list is a community!\n')
    community_tuples = list(sorted(c) for c in next(comp))
    print('commtuples:', community_tuples)
    print('Number of communities: ', len(community_tuples))

    end_time = time.time()

    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tBUILT-IN computation of ONE-SHOT Girvan-Newman clustering for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")
    return G


######################################################################################################################
def _3300_3340_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions,percentage):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_3300_3340_divisive_community_detection(G,number_of_divisions,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # CREATE HIERARCHY OF num_divisions COMMUNITIES FOR A GRAPH WITH num_components CONNECTED COMPONENTS, WHERE 
        # MIN(num_nodes / 4, 10*num_components) >= num_divisions >= K. 
        #
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   graph_layout = a string determining how to position the graph nodes in the plane.
        #   node_positions = a list of previously computed node positions in the plane (for next call of my_graph_plot_routine)
        # OUTPUT: 
        #   A list, hierarchy_of_community_tuples, whose first item is the list community_tuples returned by the 
        #   ONE-SHOT GN algorithm, and each subsequent item is a triple of node-tuples: 
        #    * the tuple of the community to be removed next, and 
        #    * the two tuples of the subcommunities to be added for its substitution
        #
        # PSEUDOCODE:   
        #   HIERARCHY = LIST WITH ONE ITEM, THE community_tuple (LIST OF TUPLES) WITH K+1 COMMUNITIES DETERMINED BY THE 
        # ONE-SHOT GIRVAN-NEWMAN ALGORITHM
        #   REPEAT:
        #       ADD TO THE HIERARCHY A TRIPLE: THE COMMUNITY TO BE REMOVED AND THE TWO SUB-COMMUNITIES THAT MUST SUBSTITUTE IT 
        #       * THE COMMUNITY TO BE REMOVED IS THE LARGEST ONE IN THE PREVIOUS STEP.
        #       * THE TWO SUBCOMMUNITIES TO BE ADDED ARE DETERMINED BY REMOVING MAX_BC EDGES FROM THE SUBGRAPH INDUCED BY THE 
        #       COMMUNITY TO BE REMOVED, UNTIL DISCONNECTION. 
        #   UNTIL num_communities REACHES THE REQUIRED NUMBER num_divisions OF COMMUNITIES
        ######################################################################################################################
''')

    start_time = time.time()

    print(bcolors.ENDC  + "\tUSEFUL FUNCTIONS:")

    print(bcolors.ENDC  + "\t\t" 
                        + bcolors.OKCYAN    + "girvan_newman(...) " 
                        + bcolors.ENDC      + "from networkx.algorithms.community, provides TWO communities for a given CONNECTED graph.")

    print(bcolors.ENDC  + "\t\t"
                        + bcolors.OKCYAN    + "G.subgraph(X) " 
                        + bcolors.ENDC      + "extracts the induced subgraph of G, for a given subset X of nodes.") 

    print(bcolors.ENDC  + "\t\t" 
                        + bcolors.OKCYAN    + "community_tuples.pop(GiantCommunityIndex) " 
                        + bcolors.ENDC      + "removes the giant community's tuple (to be split) from the community_tuples data structure.")
    print(bcolors.ENDC  + "\t\t"    
                        + bcolors.OKCYAN    + "community_tuples.append(...) " 
                        + bcolors.ENDC      + "can add the computed subcommunities' tuples, in substitution of the giant component.")
    _3300_3340_one_shot_girvan_newman_for_communities2(G, number_of_divisions, percentage)
    end_time = time.time()
    G = reconstruct_graph(graph_layout, node_positions, 'else')
    print(bcolors.ENDC  + "\t===================================================")
    print(bcolors.ENDC  + "\tComputation of HIERARCHICAL BIPARTITION of G in communities, "
                        + "using the BUILT-IN girvan-newman algorithm, for a graph with",G.number_of_nodes(),"nodes and",G.number_of_edges(),"links. "
                        + "Computation time =", end_time - start_time,"\n")

    return G


######################################################################################################################
def _3300_3340_determine_opt_community_structure(G,hierarchy_of_community_tuples):
    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_3300_3340_determine_opt_community_structure(G,hierarchy_of_community_tuples)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN A HIERARCHY OF COMMUNITY PARTITIONS FOR A GRAPH, COMPUTE THE MODULARITY OF EAC COMMUNITY PARTITION. 
        # RETURN THE COMMUNITY PARTITION OF MINIMUM MODULARITY VALUE
        # INPUT: 
        #   G = the constructed graph to work with. 
        #   hierarchy_of_community_tuples = the output of the DIVISIVE_COMMUNITY_DETECTION routine
        # OUTPUT: 
        #   The partition which achieves the minimum modularity value, within the hierarchy 
        #        #
        # PSEUDOCODE:
        #   Iterate over the HIERARCHY, to construct (sequentially) all the partitions of the graph into communities
        #       For the current_partition, compute its own current_modularity value
        #   IF      current_modularity_vale < min_modularity_value (so far)
        #   THEN    min_partition = current_partition
        #           min_modularity_value = current_modularity_value
        #   RETURN  (min_partition, min_modularity_value)   
        ######################################################################################################################
        ''')
    print('Modularities array: ', MODULARITIES)
    maxmodularity = max(MODULARITIES)
    indexofmaxmodularity = MODULARITIES.index(maxmodularity)
    print('Maximum Modularity:', maxmodularity)
    print('LC with the maximum modularity:', LCs[indexofmaxmodularity])
    print('LC1:', LC1s[indexofmaxmodularity])
    print('LC2:', LC2s[indexofmaxmodularity])
    print('LCs graph:')
    my_graph_plot_routine(LCgraphlist[indexofmaxmodularity], 'grey', 'darkblue', '-', 'spring', [], True)
    yesorno = input('Do you want to see a specifics partition modularity?(YES or NO)\n')
    if yesorno == 'YES':
        number = input('Number:\n')
        number = int(number)
        number = num_divisions - number
        print('Modularity of this partition is:', MODULARITIES[number])
    partitionlist = []
    length = len(communities) - len(MODULARITIES)
    for x in range(len(MODULARITIES)):
        partitionlist.append(length)
        length += 1
    dataf = pd.DataFrame({'Number of partition': partitionlist, 'Modularity': MODULARITIES})
    ax = dataf.plot.bar(x='Number of partition', y='Modularity', rot=0)
    plt.show()

######################################################################################################################
def _3300_3340_add_hamilton_cycle_to_graph(G,node_names_list):

    print(bcolors.ENDC + "\tCalling routine " + bcolors.HEADER + "_3300_3340_add_hamilton_cycle_to_graph(G,node_names_list)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, and a list of node-IDs.
        # OUTPUT: The augmented graph of G, with a (MAMILTON) cycle containing all the nodes from the node_names_list (in that order).
        #
        # COMMENT: The role of this routine is to add a HAMILTON cycle, i.e., a cycle that contains all the nodes in the graph.
        # Such an operation will guarantee that the graph is connected and, moreover, there is no bridge in the new graph.
        ######################################################################################################################
    ''')

    print(  bcolors.ENDC        + "\tUSEFUL FUNCTIONS:") 

    print("\t\t"
            + bcolors.OKCYAN    + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) " 
            + bcolors.ENDC      + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")
    node1list = []
    node2list = []
    for i in range(0, len(node_names_list) - 1):
        # checking if i - i+1 are connected.
        # if they arent, i'm connecting them.
        if node_names_list[i + 1] not in neighborhoodarray[i]:
            node1list.append(node_names_list[i])
            node2list.append(node_names_list[i + 1])
            # updating array with neighbours
            neighborhoodarray[i].append(node_names_list[i + 1])
            neighborhoodarray[i + 1].append(node_names_list[i])
    # checking also the last element with the first.
    if node_names_list[0] not in neighborhoodarray[len(node_names_list) - 1]:
        node1list.append(node_names_list[len(node_names_list) - 1])
        node2list.append(node_names_list[0])
        # updating array with neighbours
        neighborhoodarray[len(node_names_list) - 1].append(node_names_list[0])
        neighborhoodarray[0].append(node_names_list[len(node_names_list) - 1])
    for i in range(len(node1list)):
        G.add_edge(node1list[i], node2list[i])
    # lastdf = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
    # hamdf = pd.concat([df, lastdf], ignore_index=True)
    my_graph_plot_routine(G, my_random_color_list_generator(len(node_names_list)),
                          my_random_color_list_generator(G.number_of_edges()), '-', layout, [],True)
    return G
    
######################################################################################################################
# ADD RANDOM EDGES TO A GRAPH...
######################################################################################################################
def _3300_3340_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY):

    print(  bcolors.ENDC     + "\tCalling routine " 
            + bcolors.HEADER + "_3300_3340_add_random_edges_to_graph(G,node_names_list,NUM_RANDOM_EDGES,EDGE_ADDITION_PROBABILITY)"
            + bcolors.ENDC   + "\n")

    print(bcolors.ENDC + "\t" + '''
        ######################################################################################################################
        # GIVEN AN INPUT GRAPH WHICH IS DISCONNECTED, ADD A (HAMILTON) CYCLE CONTAINING ALL THE NODES IN THE GRAPH
        # INPUT: A graph G, an integer indicating the max number of random edges to be added per node, and an edge addition 
        # probability, for each random edge considered. 
        # 
        # OUTPUT: The augmented graph of G, containing also  all the newly added random edges.
        #
        # COMMENT: The role of this routine is to add, per node, a number of random edges to other nodes.
        # Create a for-loop, per node X, and then make NUM_RANDOM_EDGES attempts to add a new random edge from X to a 
        # randomly chosen destination (outside its neighborhood). Each attempt will be successful with probability 
        # EDGE_ADDITION_PROBABILITY (i.e., only when a random coin-flip in [0,1] returns a value < EDGE_ADDITION_PROBABILITY).")
        ######################################################################################################################
    ''')

    print(bcolors.ENDC          + "\tUSEFUL FUNCTIONS:")
    print("\t\t" 
            + bcolors.OKCYAN    + "my_graph_plot_routine(G,'grey','blue','solid',graph_layout,node_positions) " 
            + bcolors.ENDC      + "plots the graph with the grey-color for nodes, and (blue color,solid style) for the edges.\n")
    # randedgesnum = input('Please enter the number of new random edges:\n')
    # NUM_RANDOM_EDGES = int(randedgesnum)
    # edgeaddprob = input('Please enter the propability of every edge:(0-100 e.g 50 for 50%.)\n')
    # EDGE_ADDITION_PROBABILITY = int(edgeaddprob)
    # we will do the coin flip with a random number generator from 0-100.
    # if the random num is lower than the probabiblity of the edge, the edge is put to graph.
    node1list = []
    node2list = []
    for i in range(0, len(node_names_list)):
        for y in range(NUM_RANDOM_EDGES):
            rndm = random.randint(0, len(node_names_list) - 1)
            while ((rndm == i) or (node_names_list[rndm] in neighborhoodarray[i])) and len(neighborhoodarray[i]) < len(
                    node_names_list) - 1:
                rndm = random.randint(0, len(node_names_list) - 1)
            # get random number that is not the element that we want to connect
            # and is not a neighbour of the element that we want to connect
            # now we get another random number to do the coin flip
            # if the random number is lower than the possibility number given as input by the user
            # we connect the elements
            rndm2 = random.random()
            if rndm2 <= EDGE_ADDITION_PROBABILITY and len(neighborhoodarray[i]) < len(node_names_list) - 1:
                neighborhoodarray[i].append(node_names_list[rndm])
                neighborhoodarray[rndm].append(node_names_list[i])
                node1list.append(node_names_list[i])
                node2list.append(node_names_list[rndm])
    for i in range(len(node1list)):
        G.add_edge(node1list[i], node2list[i])
    my_graph_plot_routine(G, my_random_color_list_generator(len(node_names_list)),
                          my_random_color_list_generator(G.number_of_edges()), '-', layout, [],True)
    return G
######################################################################################################################
# VISUALISE COMMUNITIES WITHIN A GRAPH
######################################################################################################################
def _3300_3340_visualize_communities(G,community_tuples,graph_layout,node_positions):

    print(bcolors.ENDC      + "\tCalling routine " 
                            + bcolors.HEADER + "_3300_3340_visualize_communities(G,community_tuples,graph_layout,node_positions)" + bcolors.ENDC +"\n")

    print(bcolors.ENDC      + "\tINPUT: A graph G, and a list of lists/tuples each of which contains the nodes of a different community.")
    print(bcolors.ENDC      + "\t\t The graph_layout parameter determines how the " + bcolors.OKCYAN + "my_graph_plot_routine will position the nodes in the plane.")
    print(bcolors.ENDC      + "\t\t The node_positions list contains an existent placement of the nodes in the plane. If empty, a new positioning will be created by the " + bcolors.OKCYAN + "my_graph_plot_routine" + bcolors.ENDC +".\n")
 
    print(bcolors.ENDC      + "\tOUTPUT: Plot the graph using a different color per community.\n")

    print(bcolors.ENDC      + "\tUSEFUL FUNCTIONS:")
    print(bcolors.OKCYAN    + "\t\tmy_random_color_list_generator(number_of_communities)" + bcolors.ENDC + " initiates a list of random colors for the communities.")
    print(bcolors.OKCYAN    + "\t\tmy_graph_plot_routine(G,node_colors,'blue','solid',graph_layout,node_positions)" + bcolors.ENDC + " plots the graph with the chosen node colors, and (blue color,solid style) for the edges.")
    node1list = []
    node2list = []
    noduplicates = []
    for ii in range(len(node_names_list)):
        for yy in range(len(neighborhoodarray[ii])):
            dupl = []
            dupl.append(node_names_list[ii])
            dupl.append(neighborhoodarray[ii][yy])
            rev = []
            rev.append(dupl[1])
            rev.append(dupl[0])
            if rev not in noduplicates and (rev[0] != rev[1]):
                node1list.append(node_names_list[ii])
                node2list.append(neighborhoodarray[ii][yy])
                noduplicates.append(dupl)
    for ii in range(len(communities)):
        # in case we have an alone node, we make it connect itself so we have as much communities as we wanted.
        if len(communities[ii]) == 1:
            node1list.append(communities[ii][0])
            node2list.append(communities[ii][0])
            indexofalonenode = node_names_list.index(communities[ii])
            neighborhoodarray[indexofalonenode].append(communities[ii][0])
    gcc4df = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
    G = nx.from_pandas_edgelist(gcc4df, "node_1", "node_2", create_using=nx.Graph())
    if graph_layout == 'circular':
        pos = nx.circular_layout(G)
    elif graph_layout == 'random':
        pos = nx.random_layout(G, seed=50)
    elif graph_layout == 'planar':
        pos = nx.planar_layout(G)
    elif graph_layout == 'shell':
        pos = nx.shell_layout(G)
    else:  # DEFAULT VALUE == spring
        pos = nx.spring_layout(G)
    my_graph_plot_routine(G, 'brown', 'darkblue', '-', graph_layout, pos,False)
    if len(communities) == 0:
        compute_coherent_components()
    my_color_list = my_random_color_list_generator(len(communities))
    for i in range(len(communities)):
        nx.draw_networkx_nodes(G, pos, nodelist=communities[i], node_color=my_color_list[i])
    plt.show()


def _3300_3340_one_shot_girvan_newman_for_communities2(G, divs, psb):
    if len(communities) == 0:
        compute_coherent_components()
    divs = divs + 1 - len(communities)
    for i in range(divs):
        compute_coherent_components()
        print('comm:', communities)
        print('Number of our communities', len(communities))
        fakecommunities = []
        for xxx in communities:
            fakecommunities.append(xxx)
        communitiesnum = len(communities)
        maxlength = 0
        for i in range(len(communities)):
            if len(communities[i]) > maxlength:
                maxlength = len(communities[i])
                gcc = communities[i]
        node1list = []
        node2list = []

        for i in range(len(gcc)):
            index = node_names_list.index(gcc[i])
            for y in range(len(neighborhoodarray[index])):
                node1list.append(gcc[i])
                node2list.append(neighborhoodarray[index][y])
        counter = 0
        while communitiesnum == len(communities):
            counter += 1
            gccdf = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
            G = nx.from_pandas_edgelist(gccdf, "node_1", "node_2", create_using=nx.Graph())
            if counter == 1:
                LCgraph = G
                LCgraphlist.append(G)
                LCs.append(gcc)
            # creating the graph because to find the BC values we have to put it as an argument!
            sources = []    # sources will be the percentage given by the user * all the nodes
            # sources nodes will be selected randomly.
            targets = gcc    # targets will be all the nodes of the community
            sourceslength = len(gcc) * (psb / 100)
            sourceslength = int(math.ceil(sourceslength))
            for zzz in range(sourceslength):
                rndmint = random.randint(0, len(gcc) - 1) # selecting the INDEX!
                while gcc[rndmint] in sources:
                    rndmint = random.randint(0, len(gcc) - 1)
                sources.append(gcc[rndmint])
            x = nx.edge_betweenness_centrality_subset(G, sources, targets, None, None)
            max_value = max(x.values())
            max_key = max(x, key=x.get)
            index1 = node_names_list.index(max_key[0])
            index2 = node_names_list.index(max_key[1])
            deletingindex = 0
            flag = False
            for i in range(len(neighborhoodarray[index1])):
                if neighborhoodarray[index1][i] == max_key[1]:
                    deletingindex = i
                    flag = True
            if flag :
                del neighborhoodarray[index1][deletingindex]
                flag = False
            for i in range(len(neighborhoodarray[index2])):
                if neighborhoodarray[index2][i] == max_key[0]:
                    deletingindex = i
                    flag = True
            if flag:
                del neighborhoodarray[index2][deletingindex]
            deletedlist = []
            for i in range(len(node1list)):
                if (node1list[i] == max_key[0] and node2list[i] == max_key[1]) or (node1list[i] == max_key[1] and node2list[i] == max_key[0]):
                    deletedlist.append(i)
            for i in deletedlist:
                del node1list[i]
                del node2list[i]
            communities.clear()
            compute_coherent_components()
        # WE ARE OUT OF THE WHILE LOOP, SO THE PARTITION HAS BEEN MADE
        # WE HAVE KEPT THE GRAPH OF THE LC ON VARIABLE LCgraph
        # WE WILL FIND MODULARITY FROM THIS GRAPH AND THE LC1 and LC2 (PARTITIONS OF LC)
        LCss = []
        exists = 0
        # with this loop we hold the LC1 and LC2 on the LCss array so we will put them as an argument for finding modularity
        for fk in range(len(communities)):
            for fk2 in range(len(fakecommunities)):
                if communities[fk] == fakecommunities[fk2]:
                    exists = 1
                    # if exists 1 it means we had this community before, so this is not a partition.
                    # if a community is not the same with a community before the partition
                    # this means it is LC1 or LC2.
            if exists == 0:
                LCss.append(communities[fk])
            else:
                exists = 0
        LC1s.append(LCss[0])
        LC2s.append(LCss[1])
        LC1 = set(LCss[0])
        LC2 = set(LCss[1])
        modularity = nx_comm.modularity(LCgraph, [LC1, LC2])
        MODULARITIES.append(modularity)
    return


def compute_coherent_components():
    for i in range(0, len(neighborhoodarray)):
        supportlist = []
        x = node_names_list[i]
        supportlist.append(x)
        boolvar = 0
        counter = []
        for y in range(0, len(neighborhoodarray[i])):
            for z in range(0, len(communities)):
                if neighborhoodarray[i][y] in communities[z]:
                    boolvar = 1
                    if z not in counter:
                        counter.append(z)
            supportlist.append(neighborhoodarray[i][y])
        if boolvar == 0:
            communities.append(supportlist)
        else:
            if len(counter) == 1:
                for xx in supportlist:
                    if xx not in communities[counter[0]]:
                        communities[counter[0]].append(xx)
            else:
                counter = sorted(counter)
                for xxx in range(1, len(counter)):
                    for xxx2 in communities[counter[xxx]]:
                        if xxx2 not in communities[counter[0]]:
                            communities[counter[0]].append(xxx2)
                offset = 0
                for xxx in range(1, len(counter)):
                    del communities[counter[xxx] - offset]
                    offset += 1
    for i in communities:
        i = sorted(i)
    return


def reconstruct_graph(layoutt, positions,str):
    node1list = []
    node2list = []
    noduplicates = []
    for ii in range(len(node_names_list)):
        for yy in range(len(neighborhoodarray[ii])):
            dupl = []
            dupl.append(node_names_list[ii])
            dupl.append(neighborhoodarray[ii][yy])
            rev = []
            rev.append(dupl[1])
            rev.append(dupl[0])
            if rev not in noduplicates and (rev[0] != rev[1]):
                node1list.append(node_names_list[ii])
                node2list.append(neighborhoodarray[ii][yy])
                noduplicates.append(dupl)
    for ii in range(len(communities)):
        # in case we have an alone node, we make it connect itself so we have as much communities as we wanted.
        if len(communities[ii]) == 1:
            node1list.append(communities[ii][0])
            node2list.append(communities[ii][0])
            indexofalonenode = node_names_list.index(communities[ii])
            neighborhoodarray[indexofalonenode].append(communities[ii][0])
    gcc3df = pd.DataFrame({'node_1': node1list, 'node_2': node2list})
    G = nx.from_pandas_edgelist(gcc3df, "node_1", "node_2", create_using=nx.Graph())
    if str =='GV':
        print('Printing graph after the Girvan Newman!')
    else:
        print('Printing graph after the divisive community detection!')
    my_graph_plot_routine(G, my_random_color_list_generator(len(node_names_list)),
                          my_random_color_list_generator(gcc3df.shape[0]), '-', layoutt, positions,True)
    return G


########################################################################################
########################### STUDENT_AM ROUTINES LIBRARY ENDS ###########################
########################################################################################


########################################################################################
############################# MAIN MENUS OF USER CHOICES ############################### 
########################################################################################

############################### GLOBAL INITIALIZATIONS #################################
G = nx.Graph()                      # INITIALIZATION OF THE GRAPH TO BE CONSTRUCTED
flagg = True
node_names_list = []
node_positions = []                 # INITIAL POSITIONS OF NODES ON THE PLANE ARE UNDEFINED...
community_tuples = []               # INITIALIZATION OF LIST OF COMMUNITY TUPLES...
hierarchy_of_community_tuples = []  # INITIALIZATION OF HIERARCHY OF COMMUNITY TUPLES
neighborhoodarray = []              # INITIALIZATION OF ARRAY FOR NEIGHBOURS
communities = []                    # INITIALIZATION OF COMMUNITIES ARRAY
arrayforratingedges = []            # INITIALIZATION OF ARRAY FOR RATING THE EDGES FOR BETWEENESS
LCs = []                            # INITIALIZATION OF ARRAY WHICH HOLDS THE COMMUNITIES THAT HAVE BEEN PARTIOTIONED
LC1s = []                           # INITIALIZATION OF ARRAY WHICH HOLDS THE FIRST PARTITION COMMUNITY OF THE LCs
LC2s = []                           # INITIALIZATION OF ARRAY WHICH HOLDS THE SECOND PARTITION COMMUNITY OF THE LCs
MODULARITIES = []                   # INITIALIZATION OF ARRAY WHICH HOLDS ALL THE MODULARITIES
LCgraphlist = []
startinglength = 0
# MODULARITIES[i] contains info for the LCs[i] and the partitions of LCs[i] -> LC1s[i] and LC2s[i]
communitiesnum = 0
MAX_NUM_LINKS = 0
num_divisions = 0
layout = ''

G, node_names_list, node_positions = my_menu_graph_construction(G, node_names_list, node_positions)

my_menu_community_detection(G, node_names_list, node_positions, hierarchy_of_community_tuples)
