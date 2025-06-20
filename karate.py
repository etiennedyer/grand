import torch
import torch.nn as nn
from torch.nn.functional import softmax
from math import ceil, sqrt
import networkx as nx

torch.manual_seed(42)

'''
pytorch implementation of GRAND-l (linear) with fwd euler
'''

class GraphAttention(nn.Module):
  '''
  for the full model we need attention to be
  a class instead of a function, as that allows us to
  store WK and WQ as internal states
  '''
  def __init__(self, feature_dim, d_k):
      super().__init__()
      self.feature_dim = feature_dim
      self.d_k = d_k
      self.W_K = nn.Linear(feature_dim, d_k)
      self.W_Q = nn.Linear(feature_dim, d_k)

  def forward(self, x, edges):
      num_nodes = x.shape[0]
      A = torch.zeros((num_nodes, num_nodes))

      W_Kx = self.W_K(x)
      W_Qx = self.W_Q(x)

      # calculate similarity scores
      for u, v in edges:
          score = torch.dot(W_Kx[u], W_Qx[v]) / self.d_k
          A[u, v] = torch.exp(score) # no longer assuming graph is undirected
          '''
          this is the implementation from our last meeting
          but I will replace the for loop with a vectorized operation
          and use a sparse matrix for efficiency
          '''
      row_sums = A.sum(dim=1, keepdim=True)
      A = A / torch.where(row_sums == 0, torch.tensor(1.0), row_sums)
      return A

def fwd_diffusion(edges, x_initial, attention, tau, T):
    num_nodes = x_initial.shape[0]
    x_k = x_initial.clone()
    A = attention(x_initial, edges)
    I = torch.eye(num_nodes)

    update_matrix = I + tau * (A - I)
    steps = ceil(T / tau)

    for step in range(steps):
        x_k = torch.matmul(update_matrix, x_k)

    return x_k
    '''
    this is where we have the difference between linear and nonlinear
    the nl model would update the attention in the for loop
    '''

class GRANDModel(nn.Module):
  def __init__(self, input_dim, hidden_dim, num_classes, d_k):
    super().__init__()
    self.encoder = nn.Linear(input_dim, hidden_dim)
    self.attention = GraphAttention(feature_dim=hidden_dim, d_k=d_k)
    self.decoder = nn.Linear(hidden_dim, num_classes)

  def forward(self, x_in, edges, tau, T):
    x_initial = self.encoder(x_in)
    final_features = fwd_diffusion(
        edges=edges,
        x_initial = x_initial,
        attention = self.attention,
        tau=tau,
        T=T)

    output = self.decoder(final_features)
    return output

'''
prediction on karate club dataset

'''

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

classes = [0, 1]

model = GRANDModel(input_dim=34, hidden_dim = 128, num_classes = 2, d_k = 16)

features = torch.eye(34)
labeled_nodes = torch.tensor([10, 3])
labels = torch.tensor([0, 1])

optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
criterion = torch.nn.CrossEntropyLoss()


tau = 0.1
T = 1.0

for epoch in range(100):
    optimizer.zero_grad()
    output = model(features, edges, tau, T)
    loss = criterion(output[labeled_nodes], labels)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
      print('Epoch:', epoch, 'Loss:', loss.item())

final_out = model(features, edges, tau, T)
predicted_classes = torch.argmax(final_out, dim=1)

ground_truth = torch.tensor([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0,
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

count = (predicted_classes == ground_truth).sum()
print(f"Accuracy: {count.item()/34}")