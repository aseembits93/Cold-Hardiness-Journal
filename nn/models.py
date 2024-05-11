import torch
import torch.nn as nn

class model_selection_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(model_selection_net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class model_selection_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(model_selection_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class aggregate_single_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(aggregate_single_net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class aggregate_single_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(aggregate_single_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class single_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(single_net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class single_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars=1, nonlinear=False):
        super(single_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048

        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next


class multiplicative_embedding_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label).tanh()
        #multiply x, embedding_out
        x = torch.mul(x,embedding_out)
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight).tanh()
        embedding_out = embedding_out.squeeze()
        #multiply x, embedding_out
        x = torch.mul(x,embedding_out.repeat(x.shape[0],x.shape[1],1))

        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_afterL1, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label).tanh()
        #multiply x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = torch.mul(out, embedding_out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune_afterL1, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight).tanh()
        embedding_out = embedding_out.squeeze()
        #multiply x, embedding_out


        out = self.linear1(x).relu()
        out = torch.mul(out,embedding_out.repeat(out.shape[0],out.shape[1],1))
        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_afterL2, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label).tanh()
        #multiply x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        out = torch.mul(out,embedding_out)
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune_afterL2, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight).tanh()
        embedding_out = embedding_out.squeeze()
        #multiply x, embedding_out


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()
        out = torch.mul(out,embedding_out.repeat(out.shape[0],out.shape[1],1))
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_afterL3, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label).tanh()
        #multiply x, embedding_out

        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out = torch.mul(out,embedding_out)
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune_afterL3, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight).tanh()
        embedding_out = embedding_out.squeeze()
        #multiply x, embedding_out


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out
        out = torch.mul(out,embedding_out.repeat(out.shape[0],out.shape[1],1))
        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_afterL4, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label).tanh()
        #multiply x, embedding_out

        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_s = torch.mul(out_s,embedding_out)
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune_afterL4, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight).tanh()
        embedding_out = embedding_out.squeeze()
        #multiply x, embedding_out


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_s = torch.mul(out_s,embedding_out.repeat(out_s.shape[0],out_s.shape[1],1))

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class multiplicative_embedding_net_finetune_scratch(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(multiplicative_embedding_net_finetune_scratch, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.scratch_embedding = nn.Parameter(torch.rand(input_size))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        #multiply x, embedding_out
        x = torch.mul(x,self.scratch_embedding.repeat(x.shape[0],x.shape[1],1).tanh())

        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class mtl_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(mtl_net, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4_list = nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(no_of_cultivars)])  # LT10
        self.linear5_list = nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(no_of_cultivars)])  # LT50
        self.linear6_list = nn.ModuleList([nn.Linear(self.penul, 1) for _ in range(no_of_cultivars)])  # LT90

    def forward(self, x, cultivar_label=None, h=None):
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul

        labels = cultivar_label[:,0] #the whole sequence has the same label
        batch_size = cultivar_label.shape[0]
        out_lt_10 = [self.linear4_list[labels[i]](out_s[i]) for i in range(batch_size)] # LT10
        out_lt_50 = [self.linear5_list[labels[i]](out_s[i]) for i in range(batch_size)] # LT50
        out_lt_90 = [self.linear6_list[labels[i]](out_s[i]) for i in range(batch_size)] # LT90
        return torch.stack(out_lt_10), torch.stack(out_lt_50), torch.stack(out_lt_90), 0, 0

class mtl_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(mtl_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90

    def forward(self, x, cultivar_label=None, h=None):
        out = self.linear1(x).relu()
        out = self.linear2(out).relu()
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        return out_lt_10, out_lt_50, out_lt_90, 0, 0

class additive_embedding_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        x = x + embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        embedding_out = embedding_out.repeat(x.shape[0],x.shape[1],1)
        #add x, embedding_out
        x = x + embedding_out

        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_afterL1, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = out + embedding_out

        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_finetune_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_finetune_afterL1, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()

        #add x, embedding_out


        out = self.linear1(x).relu()

        #out = self.dropout(out)
        embedding_out = embedding_out.repeat(out.shape[0],out.shape[1],1)
        out = out + embedding_out
        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_afterL2, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out

        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        out = out + embedding_out
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_finetune_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_finetune_afterL2, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()
        embedding_out = embedding_out.repeat(out.shape[0],out.shape[1],1)
        #add x, embedding_out
        out = out + embedding_out
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_afterL3, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out

        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out = out + embedding_out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_finetune_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_finetune_afterL3, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out
        embedding_out = embedding_out.repeat(out.shape[0],out.shape[1],1)
        #add x, embedding_out
        out = out + embedding_out
        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_afterL4, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)

        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        #add x, embedding_out
        out_s = out_s + embedding_out
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class additive_embedding_net_finetune_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(additive_embedding_net_finetune_afterL4, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()


        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul
        embedding_out = embedding_out.repeat(out_s.shape[0],out_s.shape[1],1)
        #add x, embedding_out
        out_s = out_s + embedding_out
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next


class concat_embedding_net(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size)
        self.linear1 = nn.Linear(input_size*2, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        x = torch.cat((x,embedding_out),axis=-1)
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_finetune(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, input_size),nn.ReLU(),nn.Linear(input_size, input_size)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, input_size).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size*2, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        #add x, embedding_out
        x = torch.cat((x,embedding_out.repeat(x.shape[0],x.shape[1],1)),axis=-1)

        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_afterL1, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(2048, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = torch.cat((out,embedding_out),axis=-1)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_finetune_afterL1(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune_afterL1, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(2048, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        #add x, embedding_out
        out = self.linear1(x).relu()
        out = torch.cat((out,embedding_out.repeat(out.shape[0],out.shape[1],1)),axis=-1)
        #out = self.dropout(out)

        out = self.linear2(out).relu()

        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next
class concat_embedding_net_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_afterL2, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=4096, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        out = torch.cat((out,embedding_out),axis=-1)
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_finetune_afterL2(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune_afterL2, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=4096, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        #add x, embedding_out
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()
        out = torch.cat((out,embedding_out.repeat(out.shape[0],out.shape[1],1)),axis=-1)
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next
class concat_embedding_net_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_afterL3, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size*2, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul, 1)  # LT10
        self.linear5 = nn.Linear(self.penul, 1)  # LT50
        self.linear6 = nn.Linear(self.penul, 1)  # LT90
        self.linear7 = nn.Linear(self.penul, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out = torch.cat((out,embedding_out),axis=-1)
        out_s = self.linear3(out).relu()  # penul
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_finetune_afterL3(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune_afterL3, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 2048),nn.ReLU(),nn.Linear(2048, 2048)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 2048).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size*2, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        #add x, embedding_out
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out
        out = torch.cat((out,embedding_out.repeat(out.shape[0],out.shape[1],1)),axis=-1)
        out_s = self.linear3(out).relu()  # penul

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next
class concat_embedding_net_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_afterL4, self).__init__()
        self.numLayers = 1
        self.penul = 1024
        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024)
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 2048)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul)  # penul
        self.linear4 = nn.Linear(self.penul*2, 1)  # LT10
        self.linear5 = nn.Linear(self.penul*2, 1)  # LT50
        self.linear6 = nn.Linear(self.penul*2, 1)  # LT90
        self.linear7 = nn.Linear(self.penul*2, 1)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None):
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = self.embedding(cultivar_label)
        #add x, embedding_out
        out = self.linear1(x).relu()
        #out = self.dropout(out)
        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)
        out, h_next = self.rnn(out, h)  # rnn out
        out_s = self.linear3(out).relu()  # penul
        out_s = torch.cat((out_s,embedding_out),axis=-1)

        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90
        out_ph = self.linear7(out_s).sigmoid()  # Budbreak
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next

class concat_embedding_net_finetune_afterL4(nn.Module):  # LT50 and budbreak
    def __init__(self, input_size, no_of_cultivars, nonlinear='no'):
        super(concat_embedding_net_finetune_afterL4, self).__init__()

        self.numLayers = 1

        self.penul = 1024

        self.memory_size = 2048
        self.embedding = nn.Sequential(nn.Embedding(no_of_cultivars, 1024),nn.ReLU(),nn.Linear(1024, 1024)).requires_grad_(False) if nonlinear=='yes' else nn.Embedding(no_of_cultivars, 1024).requires_grad_(False)
        self.combinations_weights = nn.Parameter(torch.rand(1,no_of_cultivars))
        self.linear1 = nn.Linear(input_size, 1024).requires_grad_(False)
        self.linear2 = nn.Linear(1024, 2048).requires_grad_(False)
        self.rnn = nn.GRU(input_size=2048, hidden_size=self.memory_size,
                          num_layers=self.numLayers, batch_first=True).requires_grad_(False)
        #self.dropout = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(self.memory_size, self.penul).requires_grad_(False)  # penul
        self.linear4 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT10
        self.linear5 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT50
        self.linear6 = nn.Linear(self.penul, 1).requires_grad_(False)  # LT90
        self.linear7 = nn.Linear(self.penul, 1).requires_grad_(False)  # Budbreak

    def forward(self, x, cultivar_label=None, h=None): #no cultivar label exists in this case
        batch_dim, time_dim, state_dim = x.shape
        embedding_out = torch.matmul(self.combinations_weights,self.embedding.weight)
        embedding_out = embedding_out.squeeze()
        #add x, embedding_out
        out = self.linear1(x).relu()

        #out = self.dropout(out)

        out = self.linear2(out).relu()
        if h is None:
            h = torch.zeros(self.numLayers, batch_dim,
                            self.memory_size, device=x.device)

        out, h_next = self.rnn(out, h)  # rnn out

        out_s = self.linear3(out).relu()  # penul
        out_s = torch.cat((out_s,embedding_out.repeat(out_s.shape[0],out_s.shape[1],1)),axis=-1)
        out_lt_10 = self.linear4(out_s)  # LT10
        out_lt_50 = self.linear5(out_s)  # LT50
        out_lt_90 = self.linear6(out_s)  # LT90

        out_ph = self.linear7(out_s).sigmoid()  # Budbreak

        # return out_s, out_lt, out_ph, h_next
        return out_lt_10, out_lt_50, out_lt_90, out_ph, h_next