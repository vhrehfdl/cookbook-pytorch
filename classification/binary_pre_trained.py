import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext.data import TabularDataset
from torchtext.vocab import GloVe
from torchtext.data.utils import get_tokenizer

from models.base_model import BaseModel
from utils.data_helper import pre_processing
from utils.evaluation import Evaluation
from utils import save_model


def load_data(train_dir, test_dir):
    tokenizer = get_tokenizer('basic_english')

    text = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    label = data.LabelField()

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', text), ('label', label)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', text), ('label', label)])

    train_data, valid_data = train_data.split(split_ratio=0.8)

    return train_data, valid_data, test_data, text, label


def training(model, optimizer, train_iter, device):
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(device), batch.label.to(device)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

    return model


def save_model(best_val_loss, val_loss, model, model_dir):
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), model_dir)


def main():
    # Hyper parameter
    batch_size = 64
    lr = 0.001
    epochs = 3
    n_classes = 2
    embedding_dim = 300
    hidden_dim = 32

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Directory
    train_dir = "../data/binary_train_data.csv"
    test_dir = "../data/binary_test_data.csv"
    model_dir = "snapshot/text_classification.pt"

    print("1.Load data")
    train_data, valid_data, test_data, text, label = load_data(train_dir, test_dir)

    print("2.Pre processing")
    train_iter, val_iter, test_iter, text, label = pre_processing(train_data, valid_data, test_data, text, label, device, batch_size)

    print("3.Build model")
    model = BaseModel(
        hidden_dim=32, 
        vocab_num = len(text.vocab), 
        embedding_dim=300, 
        class_num=len(vars(label.vocab)["itos"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("4.Train")
    best_val_loss = None
    for e in range(1, epochs + 1):
        model = training(model, optimizer, train_iter, device)
        val_loss, val_accuracy = Evaluation(model, val_iter, device)
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        save_model(best_val_loss, val_loss, model, model_dir)

    model.load_state_dict(torch.load(model_dir))
    test_loss, test_acc = Evaluation(model, test_iter, device)
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))


if __name__ == '__main__':
    main()