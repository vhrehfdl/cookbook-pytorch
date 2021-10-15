import torch
import torch.nn.functional as F
from torchtext import data
from torchtext.data import TabularDataset
from torchtext.data.utils import get_tokenizer

from models.base_model import BaseModel
from utils.data_helper import pre_processing
from utils.evaluation import Evaluation
from utils import save_model


def load_data(train_dir, test_dir):
    tokenizer = get_tokenizer('basic_english')

    text = data.Field(sequential=True, batch_first=True, lower=True, fix_length=50, tokenize=tokenizer)
    label = data.LabelField()

    train = TabularDataset(path=train_dir, skip_header=True, format='csv', fields=[('text', text), ('label', label)])
    test = TabularDataset(path=test_dir, skip_header=True, format='csv', fields=[('text', text), ('label', label)])

    train, valid = train.split(split_ratio=0.8)

    return train, valid, test, text, label


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


def main():
    # Directory
    train_dir = "../data/binary_train.csv"
    test_dir = "../data/binary_test.csv"
    model_dir = "./model_save/"


    # Hyper parameter
    batch_size = 64
    lr = 0.001
    epochs = 3


    # Flow
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    print("1.Load data")
    train_data, val_data, test_data, text, label = load_data(train_dir, test_dir)
    
    print("2.Pre processing")
    train_iter, val_iter, test_iter, text, label = pre_processing(train_data, val_data, test_data, text, label, device, batch_size)

    print("3.Build model")
    model = BaseModel(
        hidden_dim=32, 
        vocab_num = len(text.vocab), 
        embedding_dim=300, 
        class_num=len(vars(label.vocab)["itos"])
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("4.Train")
    for e in range(1, epochs + 1):
        model = training(model, optimizer, train_iter, device)
        val_loss, val_accuracy = Evaluation(model, val_iter, device).eval_classification()
        print("[Epoch: %d] val loss : %5.2f | val accuracy : %5.2f" % (e, val_loss, val_accuracy))
        save_model(model, model_dir, e, val_accuracy)

    print("5. Evaluate")
    test_loss, test_acc = Evaluation(model, test_iter, device).eval_classification()
    print('테스트 오차: %5.2f | 테스트 정확도: %5.2f' % (test_loss, test_acc))


if __name__ == '__main__':
    main()