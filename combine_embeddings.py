from pickletools import optimize
import torch
import torch.nn as nn


class embedding_net(nn.Module):  # LT50 and budbreak
    def __init__(self):
        super(embedding_net, self).__init__()
        self.combinations_weights = nn.Parameter(torch.rand(1,3))
        self.emb = nn.Embedding(3,10).requires_grad_(False)


    def forward(self,x):

        return torch.matmul(self.combinations_weights,self.emb.weight)

model = embedding_net()
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.MSELoss()
with torch.no_grad():
    true_label = 0.1 * model.emb.weight[0] - 0.1 * model.emb.weight[1] + 0.2 * model.emb.weight[2]
    true_label = true_label.unsqueeze(0)
for i in range(1000):
    #print("before",model.combinations_weights)
    optimizer.zero_grad()
    output = model(torch.rand(1,2))
    loss = criterion(output,true_label)
    print("loss",loss)
    loss.backward()
    optimizer.step()
    #print("after",model.emb.weight)
    #print("after",output)
