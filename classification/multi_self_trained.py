import os

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from torchtext.data import TabularDataset


def load_data(train_dir, test_dir):
    NLP = spacy.load('en_core_web_sm')
    tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

    TEMP_TEXT = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    TEXT = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    LABEL = data.Field(sequential=False, batch_first=True, use_vocab=True)

    train_data = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('turn1', TEMP_TEXT), ('turn2', TEMP_TEXT), ('turn3', TEXT), ('label', LABEL)])
    test_data = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('turn1', TEMP_TEXT), ('turn2', TEMP_TEXT), ('turn3', TEXT), ('label', LABEL)])

    train_data, valid_data = train_data.split(split_ratio=0.8)

    return train_data, valid_data, test_data, TEXT, LABEL


def data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device, batch_size):
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    print(vars(LABEL))
    print(LABEL.vocab)

    train_iter, val_iter = data.BucketIterator.splits((train_data, valid_data), batch_size=batch_size, device=device,
                                                      sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    test_iter = data.Iterator(test_data, batch_size=batch_size, device=device, shuffle=False, sort=False, sort_within_batch=False)

    return train_iter, val_iter, test_iter, TEXT, LABEL


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
        print(vars(batch))
        print(batch.label)
        x, y = batch.turn3.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()


def evaluate(model, val_iter, device):
    model.eval()
    corrects, total_loss = 0, 0
    print(val_iter)
    for batch in val_iter:
        x, y = batch.turn3.to(device), batch.label.to(device)
        y.data.sub_(1)  # 레이블 값을 0과 1로 변환
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction='sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy


def save_model(best_val_loss, val_loss, model, model_dir):
    # 검증 오차가 가장 적은 최적의 모델을 저장
    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), model_dir)
        best_val_loss = val_loss


def main():
    # 하이퍼파라미터
    batch_size = 64
    lr = 0.001
    EPOCHS = 30
    n_classes = 4
    embedding_dim = 300
    hidden_dim = 32

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    base_dir = ".."
    train_dir = base_dir + "/Data/multi_train_data.csv"
    test_dir = base_dir + "/Data/multi_test_data.csv"
    model_dir = "snapshot/text_classification.pt"

    train_data, valid_data, test_data, TEXT, LABEL = load_data(train_dir, test_dir)
    train_iter, val_iter, test_iter, TEXT, LABEL = data_preprocissing(train_data, valid_data, test_data, TEXT, LABEL, device, batch_size)

    vocab_size = len(TEXT.vocab)
    model = BasicModel(1, hidden_dim, vocab_size, embedding_dim, n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = None
    for e in range(1, EPOCHS + 1):
        train(model, optimizer, train_iter, device)
        val_loss, val_accuracy = evaluate(model, val_iter, device)
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        save_model(best_val_loss, val_loss, model, model_dir)

    model.load_state_dict(torch.load(model_dir))
    test_loss, test_acc = evaluate(model, test_iter, device)
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))


if __name__ == '__main__':
    main()