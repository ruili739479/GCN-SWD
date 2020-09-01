from scipy.io import loadmat
import torch
from torch_geometric.data import Data
import numpy as np
import os
from torch_geometric.io.tu import split

def create_torch_data(connectivity_measure):

    # initialize for both graphs: Baseline and SWD
    num_graph = 0
    node_this = []
    edge_this = []
    edge_attr_this = []
    index_graph = []
    y = []

    # load in Baseline data
    # load in graph topology
    file_name = connectivity_measure+"_BL.mat"
    load_name = os.path.join('E:', 'Project_SWD', 'data', file_name)
    load1 = loadmat(load_name)
    file_BL = load1[file_name[0:-4]]

    # load in node feature
    load_name = os.path.join('E:', 'Project_SWD', 'data', 'PSD_Baseline.mat')
    load1 = loadmat(load_name)
    Node_feature_BL = load1['PSD_Baseline']

    for igraph in range(len(file_BL)):
        # get the connectivity network computed in matlab
        graph_this = file_BL[igraph, 0]

        # get the edge list and edge attributes for the network
        for irow in range(graph_this.shape[0]):
            for icol in range(graph_this.shape[1]):
                if irow != icol and graph_this[irow, icol]>0.6:
                    edge_this.append([num_graph*graph_this.shape[0]+irow, num_graph*graph_this.shape[0]+icol])
                    edge_attr_this.append(graph_this[irow, icol])
            index_graph.append(num_graph)
            node_this.append(Node_feature_BL[igraph, irow])
        y.append(0)
        num_graph = num_graph + 1

    # load in SWD data
    file_name = connectivity_measure + "_SWD.mat"
    load_name = os.path.join('E:', 'Project_SWD', 'data', file_name)
    load1 = loadmat(load_name)
    file_SWD = load1[file_name[0:-4]]

    # load in node feature
    load_name = os.path.join('E:', 'Project_SWD', 'data', 'PSD_SWD.mat')
    load1 = loadmat(load_name)
    Node_feature_SWD = load1['PSD_SWD']

    for igraph in range(len(file_SWD)):
        # get the connectivity network computed in matlab
        graph_this = file_SWD[igraph, 0]

        # get the edge list and edge attributes for the network
        for irow in range(graph_this.shape[0]):
            for icol in range(graph_this.shape[1]):
                if irow != icol and graph_this[irow, icol]>0.6:
                    edge_this.append(
                        [num_graph * graph_this.shape[0] + irow, num_graph * graph_this.shape[0] + icol])
                    edge_attr_this.append(graph_this[irow, icol])
            index_graph.append(num_graph)
            node_this.append(Node_feature_SWD[igraph, irow])
        y.append(1)
        num_graph = num_graph + 1

    aaa = 1

    # transform the data to torch data format
    node_this = np.reshape(node_this, (len(node_this), 1))
    Node_feature_torch = torch.tensor(np.array(node_this), dtype=torch.float)
    edge_this_torch = torch.tensor(np.transpose(np.array(edge_this)), dtype=torch.long)
    edge_attr_this_torch = torch.tensor(np.array(edge_attr_this), dtype=torch.float)
    index_graph_torch = torch.tensor(np.array(index_graph), dtype=torch.long)
    y_torch = torch.tensor(np.array(y), dtype=torch.long)

    # add label to this graph as 0
    data = Data(x=Node_feature_torch, edge_index=edge_this_torch, edge_attr=edge_attr_this_torch, y=y_torch)
    data, slices = split(data, index_graph_torch)

    return data, slices












