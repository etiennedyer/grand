import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from math import ceil
from torch_geometric.nn import GATConv

torch.manual_seed(1)

# karate club dataset
edges = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0),
        (7, 1), (7, 2), (7, 3), (8, 0), (8, 2), (9, 2),
        (10, 0), (10, 4), (10, 5), (11, 0), (12, 0),
        (12, 3), (13, 0), (13, 1), (13, 2), (13, 3),
        (16, 5), (16, 6), (17, 0), (17, 1), (19, 0),
        (19, 1), (21, 0), (21, 1), (25, 23), (25, 24),
        (27, 2), (27, 23), (27, 24), (28, 2), (29, 23),
        (29, 26), (30, 1), (30, 8), (31, 0), (31, 24),
        (31, 25), (31, 28), (32, 2), (32, 8), (32, 14),
        (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9),
        (33, 13), (33, 14), (33, 15), (33, 18), (33, 19),
        (33, 20), (33, 22), (33, 23), (33, 26), (33, 27),
        (33, 28), (33, 29), (33, 30), (33, 31), (33, 32)]

num_nodes = 34
features = torch.eye(34)  # one-hot encoding for each node

# convert edges to pytorch geom format
edge_index_temp = torch.tensor(edges, dtype=torch.long).t().contiguous()

# create an undirected graph
edge_index = torch.cat([edge_index_temp, edge_index_temp.flip(0)], dim=1)

def pairwise_dist(representations):
    # not sure what this does but error message said to use this 
    representations = representations.detach().numpy()
    
    # get pairwise distances
    distances = pdist(representations, metric='euclidean')
    distance_matrix = squareform(distances)
    
    return distance_matrix

def analyze(model, representations_list, depths):
    distance_matrices = []
    mean_distances = []
    std_distances = []

    # calculate pairwise distances at each depth
    for i, reps in enumerate(representations_list):
        dist_matrix = pairwise_dist(reps)
        distance_matrices.append(dist_matrix)
        
        # get statistics
        mean_dist = np.mean(dist_matrix)
        std_dist = np.std(dist_matrix)
        mean_distances.append(mean_dist)
        std_distances.append(std_dist)
        
        print("Depth", depths[i], ": Mean distance =", f"{mean_dist:.4f}", ", Std =", f"{std_dist:.4f}")
    
    # get rate of distance decay
    if len(mean_distances) > 1:
        initial_dist = mean_distances[0]
        final_dist = mean_distances[-1]
        decay_rate = (initial_dist - final_dist) / initial_dist
        print(f"Distance decay rate: {decay_rate:.4f} ({decay_rate*100:.2f}%)")
    
    return mean_distances, std_distances, distance_matrices

# GCN 
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        adj_norm = adj / adj.sum(dim=1, keepdim=True)
        return F.relu(self.linear(torch.mm(adj_norm, x)))

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.layer1 = GCNLayer(input_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, hidden_dim)
        self.layer3 = GCNLayer(hidden_dim, hidden_dim)
        self.layer4 = GCNLayer(hidden_dim, hidden_dim)
        self.layer5 = GCNLayer(hidden_dim, num_classes)
        
    def forward(self, x, adj):
        # adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes))
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1  # undirected graph
        
        # list of intermediate representations for charting
        intermediate_reps = []
        
        # initialize the five layers
        h1 = self.layer1(x, adj)
        intermediate_reps.append(h1)
        
        h2 = self.layer2(h1, adj)
        intermediate_reps.append(h2)
        
        h3 = self.layer3(h2, adj)
        intermediate_reps.append(h3)
        
        h4 = self.layer4(h3, adj)
        intermediate_reps.append(h4)
        
        h5 = self.layer5(h4, adj)
        intermediate_reps.append(h5)
        
        return h5, intermediate_reps


# GAT model
# did the GCN from scratch but couldn't figure out how to do it for GAT so I used GATConv from pytorch geom 

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=5, heads=8):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        
        # input layer
        self.layers.append(GATConv(input_dim, hidden_dim, heads=heads))
        
        # hidden layers
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
        self.layers.append(GATConv(hidden_dim * heads, hidden_dim, heads=heads))
            
        # output layer
        self.layers.append(GATConv(hidden_dim * heads, num_classes, heads=1, concat=False))

    def forward(self, x, edge_index):
        intermediate_reps = []
        
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != 5: # apply activation to all but the last layer
                x = F.elu(x)
            
            intermediate_reps.append(x)
        return x, intermediate_reps


# GRAND implementation
class GraphAttention(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.W_K = nn.Linear(feature_dim, feature_dim)
        self.W_Q = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, edges):
        num_nodes = x.shape[0]
        A = torch.zeros((num_nodes, num_nodes))

        W_Kx = self.W_K(x)
        W_Qx = self.W_Q(x)

        # calculate similarity scores
        for u, v in edges:
            score = torch.dot(W_Kx[u], W_Qx[v]) / x.shape[1] 
            A[u, v] = torch.exp(score)
        
        row_sums = A.sum(dim=1, keepdim=True)
        A = A / torch.where(row_sums == 0, torch.tensor(1.0), row_sums)
        return A

def fwd_diffusion(edges, x_initial, attention, tau, T, return_intermediate=False):
    num_nodes = x_initial.shape[0]
    x_k = x_initial.clone()
    A = attention(x_initial, edges)
    I = torch.eye(num_nodes)

    update_matrix = I + tau * (A - I)
    steps = ceil(T / tau)
    
    intermediate_reps = [x_k.clone()]
    depths = [0]

    for step in range(steps):
        x_k = torch.matmul(update_matrix, x_k)
        if step % (steps // 4) == 0:  # sample 4 intermediate states
            intermediate_reps.append(x_k.clone())
            depths.append((step + 1) * tau)

    return x_k, intermediate_reps, depths

class GRAND(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.attention = GraphAttention(feature_dim=hidden_dim)
        self.decoder = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_in, edges, tau, T):
        x_initial = self.encoder(x_in)
        
        final_features, intermediate_reps, depths = fwd_diffusion(
            edges=edges, x_initial=x_initial, attention=self.attention,
            tau=tau, T=T)
        
        return self.decoder(final_features), intermediate_reps, depths


# run analysis
gcn_model = GCN(input_dim=34, hidden_dim=16, num_classes=2)

'''using 2 hiddens dims for GAT because it has 8 heads, 
so equivalent to having hidden dim = 16 
in the other architectures'''
gat_model = GAT(input_dim=34, hidden_dim=2, num_classes=2)

grand_model = GRAND(input_dim=34, hidden_dim=16, num_classes=2)

# GCN analysis
_, gcn_reps = gcn_model(features, None)
gcn_depths = [1, 2, 3, 4, 5]

# GAT analysis  
_, gat_reps = gat_model(features, edge_index)
gat_depths = [1, 2, 3, 4, 5]

# GRAND analysis
_, grand_reps, grand_depths = grand_model(features, edges, tau=0.1, T=5.0)

# analyze oversmoothing for each model
gcn_mean, gcn_std, gcn_distances = analyze("GCN", gcn_reps, gcn_depths)
gat_mean, gat_std, gat_distances = analyze("GAT", gat_reps, gat_depths)
grand_mean, grand_std, grand_distances = analyze("GRAND", grand_reps, grand_depths)

# plot mean distances
plt.subplot(2, 2, 1)
plt.plot(gcn_depths, gcn_mean, 'o-', label='GCN', linewidth=2)
plt.plot(gat_depths, gat_mean, 's-', label='GAT', linewidth=2)
plt.plot(grand_depths, grand_mean, '^-', label='GRAND', linewidth=2)
plt.xlabel('Depth/Layer')
plt.ylabel('Mean Pairwise Distance')
plt.title('Mean Pairwise Distances vs Depth')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot distance decay rates
plt.subplot(2, 2, 2)
models = ['GCN', 'GAT', 'GRAND']
decay_rates = []
for mean_dists in [gcn_mean, gat_mean, grand_mean]:
    if len(mean_dists) > 1:
        decay_rate = (mean_dists[0] - mean_dists[-1]) / mean_dists[0]
        decay_rates.append(decay_rate)
    else:
        decay_rates.append(0)


# plot distance matrices heatmaps for final layer with a uniform color scale
gcn_final_dist = gcn_distances[-1]
grand_final_dist = grand_distances[-1]
gat_final_dist = gat_distances[-1]

# get global min and max distances to create a uniform color scale
vmin = 0
vmax = max(gcn_final_dist.max(), grand_final_dist.max(), gat_final_dist.max())

plt.subplot(2, 2, 3)
plt.imshow(gcn_final_dist, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('GCN Final Layer Distance Matrix')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(grand_final_dist, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('GRAND Final Layer Distance Matrix')
plt.colorbar()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(gcn_final_dist, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('GCN Final Layer Distance Matrix')

plt.subplot(1, 3, 2)
plt.imshow(gat_final_dist, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('GAT Final Layer Distance Matrix')

plt.subplot(1, 3, 3)
plt.imshow(grand_final_dist, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title('GRAND Final Layer Distance Matrix')

plt.subplots_adjust(right=0.85)
cbar_ax = plt.axes([0.87, 0.15, 0.02, 0.7])
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
plt.colorbar(sm, cax=cbar_ax)

plt.savefig('final-distance-matrix.png', dpi=300, bbox_inches='tight')
plt.show()