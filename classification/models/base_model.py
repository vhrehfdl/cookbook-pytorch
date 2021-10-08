import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, hidden_dim, vocab_num, embedding_dim, class_num, dropout_p=0.2):
        super(BaseModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_num, embedding_dim)
        self.fcnn = nn.Linear(embedding_dim * 50, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, class_num)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.fcnn(x)
        logit = self.out(x)

        return logit