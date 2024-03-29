import os

import spacy
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext.data import TabularDataset


def load_data(train_dir, test_dir):
    nlp = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in nlp.tokenizer(sent) if x.text != " "]

    text = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    label = data.LabelField()

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('turn1', text), ('turn2', text), ('turn3', text), ('label', label)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('turn1', text), ('turn2', text), ('turn3', text), ('label', label)])

    train_data, valid_data = train_data.split(split_ratio=0.8)

    return train_data, valid_data, test_data, text, label


def pre_processing(train_data, valid_data, test_data, text, label, device, batch_size):
    text.build_vocab(train_data)
    label.build_vocab(train_data)

    train_iter, val_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device, sort_key=lambda x: len(x.turn3), sort_within_batch=False, repeat=False)
    test_iter = data.Iterator(test_data, batch_size=batch_size, device=device, shuffle=False, sort=False, sort_within_batch=False)

    return train_iter, val_iter, test_iter, text, label


class BasicModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p=0.2):
        super(BasicModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.fcnn = nn.Linear(embed_dim * 50, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.fcnn(x)
        logit = self.out(x)
        return logit


def train(model, optimizer, train_iter, device):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.turn3.to(device), batch.label.to(device)
        optimizer.zero_grad()
        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter, device):
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.turn3.to(device), batch.label.to(device)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()

    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def save_model(best_val_loss, val_loss, model, model_dir):
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), model_dir)


def main():
    batch_size = 64
    lr = 0.001
    EPOCHS = 3
    n_classes = 4
    embedding_dim = 300
    hidden_dim = 32

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_dir = "../data/multi_train_data.csv"
    test_dir = "../data/multi_test_data.csv"
    model_dir = "snapshot/text_classification.pt"

    wandb.init(project="pytorch_cookbook_multi_self_trained", config={"dataset": "quora insurance", "type": "baseline"});
    wandb.config.epochs = EPOCHS
    wandb.config.batch_size = batch_size
    wandb.config.learning_rate = lr
    wandb.config.embedding_dim = embedding_dim

    print("1. Load data")
    train_data, valid_data, test_data, text, label = load_data(train_dir, test_dir)

    print("2. Pre processing")
    train_iter, val_iter, test_iter, text, label = pre_processing(train_data, valid_data, test_data, text, label, device, batch_size)

    print("3. Build model")
    model = BasicModel(1, hidden_dim, len(text.vocab), embedding_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb.watch(model)

    print("4. Train")
    best_val_loss = None
    for e in range(1, EPOCHS + 1):
        train(model, optimizer, train_iter, device)
        val_loss, val_accuracy = evaluate(model, val_iter, device)
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        save_model(best_val_loss, val_loss, model, model_dir)

        wandb.log({
            "epoch": e,
            "val_accuracy": val_accuracy,
            "val_loss": val_loss
        })

    model.load_state_dict(torch.load(model_dir))
    test_loss, test_acc = evaluate(model, test_iter, device)
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))


if __name__ == '__main__':
    main()