'''
implementation of the graph diffusion algorithm outlined
in section 3 of GRAND using explicit and implicit euler
'''

import numpy as np
from math import exp, ceil
from scipy.special import softmax

import networkx as nx
import matplotlib.pyplot as plt
import io
from PIL import Image
from IPython.display import Image as IPImage, display


def attention(x, edges):
    n = max(max(u, v) for u, v in edges)
    A = np.zeros((n,n))

    d_k = x.shape[1] # number of columns
    '''
    d_k is a hyperparameter that defines the shape of W_K and W_Q
    which are of dimension d_k x d, where d is the feature dimension of x
    the simplest tuning is to have d_k set to d
    '''

    W_K = np.eye(d_k)
    W_Q = np.eye(d_k) # use identity for the learned matrices

    # calculate similarity matrix
    for i, j in edges:
        i, j = i-1, j-1  # convert to 0 indexing

        Ki = (W_K @ x[i]).T
        Qj = W_Q @ x[j]

        score = (Ki @ Qj) / d_k
        A[i, j] = np.exp(score)
        A[j, i] = A[i, j]

    #softmax
    row_sums = A.sum(axis=1, keepdims=True)
    A /= row_sums
    return A


def fwd_diffusion(edges, x, tau, T):
    n = max(max(u, v) for u, v in edges)
    x_k = x.copy()
    steps = ceil(T / tau)
    I = np.eye(n)

    # initialize a list of states for visualization later
    states = [x_k]

    # diffusion using the forward euler scheme they give
    for step in range(steps):
        A = attention(x_k, edges)
        A_t = A - I
        Q_k = I + tau * A_t
        x_k1 = Q_k @ x_k
        x_k = x_k1
        states.append(x_k)

    return x_k, states

def bwd_diffusion(edges, x, tau, T):
    x_k = x.copy()
    steps = ceil(T / tau)

    n = max(max(u, v) for u, v in edges)
    I = np.eye(n)

    # initialize a list of states for visualization
    states = [x_k]

    # diffusion using backward euler scheme
    for step in range(steps):
        A = attention(x_k, edges)
        A_t = A - I
        B = I - tau * A_t
        x_k1 = np.linalg.solve(B, x_k)
        x_k = x_k1
        states.append(x_k)

    return x_k, states

def create_gif(states, edges, filename='diffusion.gif'):
    """
    Create a GIF of the diffusion process.

    Args:
        states: List of np.array of shape (n, 2) — 2 features per node
            first feature = color, second feature = size
        edges: List of tuples (i,j), undirected edges (1-based indexing)
        filename: str, output GIF filename
    """

    n = max(max(u, v) for u, v in edges)
    G = nx.Graph()
    G.add_nodes_from(range(1, n+1))
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    # Get color feature range across all states for consistent colormap
    all_features = np.concatenate([x[:, 0] for x in states])
    vmin, vmax = all_features.min(), all_features.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.viridis

    # Get size feature range across all states for consistent scaling
    all_size_features = np.concatenate([x[:, 1] for x in states])
    smin, smax = all_size_features.min(), all_size_features.max()
    size_range = smax - smin

    frames = []
    for step, x in enumerate(states):
        # x is (n, 2)
        colors = x[:, 0]
        # Normalize sizes relative to the global min/max of the second feature
        sizes = 300 + 800 * (x[:, 1] - smin) / (size_range + 1e-8)
        # 300–1100 size range

        fig, ax = plt.subplots(figsize=(6, 4))

        # Draw graph with color and size from features
        nx.draw(
            G, pos, ax=ax, with_labels=True,
            node_color=colors, cmap=cmap,
            vmin=vmin, vmax=vmax,
            node_size=sizes,
            font_color='white'
        )

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, label='Feature 1 (Color)')

        plt.title(f'Diffusion Step {step}')

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)

    frames[0].save(filename, format='GIF', append_images=frames[1:],
                   save_all=True, duration=200, loop=0)

    for frame in frames:
        frame.close()

'''
forward diffusion
unstable for tau > 1, as Thm 1 implies
'''

edges = [(1,3), (2,3), (3,4), (5, 6), (2, 6), (3, 7), (1, 6)]
x = np.array([
    [0.1, 3],
    [0.2, 0.4],
    [0.5, 0.6],
    [0.7, 0.1],
    [0.3, 0.8],
    [0.9, 0.5],
    [0.2, 0.7]
])
tau = 0.1
T = 1

x_final, states = fwd_diffusion(edges, x, tau, T)
print('Final states:', x_final)

create_gif(states, edges, 'diffusion.gif')