from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from allennlp.data import Instance
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.iterators import BasicIterator
from allennlp.data.iterators import BucketIterator
from allennlp.data.iterators import DataIterator
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules.seq2vec_encoders import PytorchSeq2VecWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.token_embedders import ElmoTokenEmbedder
from allennlp.nn import util as nn_util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.trainer import Trainer
from overrides import overrides
from scipy.special import expit
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.col_name = None
        self.use_gpu = None
        self.target_names = None
        self.label_cols = None
        self.epochs = None
        self.lr = None
        self.batch_size = None
        self.max_seq_len = None
        self.hidden_size = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    batch_size=64,
    lr=3e-4,
    epochs=3,
    hidden_size=64,
    max_seq_len=100,
    use_gpu=torch.cuda.is_available(),
    col_name=["turn3", "label"],
    label_cols=["happy", "angry", "sad", "others"],
    target_names=["0", "1", "2", "3"],
)


class LoadData(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len,
                 col_name: List[str] = None) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len
        self.col_name = col_name

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: np.ndarray = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        labels = np.array([1 if x == config.label_cols.index(labels) else 0 for x in range(0, len(config.label_cols))])
        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, df) -> Iterator[Instance]:
        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row[self.col_name[0]])],
                row[[self.col_name[1]]].values,
            )


class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder, vocab):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), len(config.label_cols))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)
        output = {"class_logits": class_logits, "loss": self.loss(class_logits, label)}

        return output


class Predictor:
    def __init__(self, model: Model, iterator: DataIterator, cuda_device: int = -1) -> None:
        self.model = model
        self.iterator = iterator
        self.cuda_device = cuda_device

    def _extract_data(self, batch) -> np.ndarray:
        out_dict = self.model(**batch)
        return expit(tonp(out_dict["class_logits"]))

    def predict(self, ds: Iterable[Instance]) -> np.ndarray:
        pred_generator = self.iterator(ds, num_epochs=1, shuffle=False)
        self.model.eval()
        pred_generator_tqdm = tqdm(pred_generator, total=self.iterator.get_num_batches(ds))
        preds = []
        with torch.no_grad():
            for batch in pred_generator_tqdm:
                batch = nn_util.move_to_device(batch, self.cuda_device)
                preds.append(self._extract_data(batch))
        return np.concatenate(preds, axis=0)


def tokenizer(x: str):
    return [w.text for w in SpacyWordSplitter(language='en_core_web_sm', pos_tags=False).split_words(x)[:config.max_seq_len]]


def tonp(tsr):
    return tsr.detach().cpu().numpy()


def load_data(train_dir, test_dir):
    token_indexer = ELMoTokenCharactersIndexer()
    reader = LoadData(tokenizer=tokenizer, token_indexers={"tokens": token_indexer}, col_name=config.col_name)

    train_data = pd.read_csv(train_dir)
    test_data = pd.read_csv(test_dir)

    test_y = test_data["label"].tolist()

    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

    train_data = reader.read(train_data)
    test_data = reader.read(test_data)
    val_data = reader.read(val_data)

    return train_data, val_data, test_data, test_y


def build_model(options_file, weight_file):
    vocab = Vocabulary()
    iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
    iterator.index_with(vocab)

    elmo_embedder = ElmoTokenEmbedder(options_file, weight_file)
    word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
    encoder: Seq2VecEncoder = PytorchSeq2VecWrapper(nn.LSTM(word_embeddings.get_output_dim(), config.hidden_size, bidirectional=True, batch_first=True))
    model = BaselineModel(word_embeddings, encoder, vocab)

    return model, iterator, vocab


def train(model, iterator, train_data, val_data):
    if config.use_gpu:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        iterator=iterator,
        train_dataset=train_data,
        validation_dataset=val_data,
        cuda_device=0 if config.use_gpu else -1,
        num_epochs=config.epochs
    )

    trainer.train()

    return model


def evaluate(vocab, model, test_data, test_y):
    seq_iterator = BasicIterator(batch_size=64)
    seq_iterator.index_with(vocab)

    predictor = Predictor(model, seq_iterator, cuda_device=0 if config.use_gpu else -1)
    test_preds = predictor.predict(test_data)

    test_y = [config.label_cols.index(x) for x in test_y]
    y_pred = list(test_preds.argmax(axis=-1))

    print("Accuracy: %.2f%%" % (accuracy_score(test_y, y_pred) * 100.0))
    print(classification_report(test_y, y_pred, target_names=config.target_names))


def main():
    # Directory
    options_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json'
    weight_file = 'https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5'

    train_dir = "../data/multi_train_data.csv"
    test_dir = "../data/multi_test_data.csv"

    print("1.Load Data")
    train_data, val_data, test_data, test_y = load_data(train_dir, test_dir)

    print("2.Build model")
    model, iterator, vocab = build_model(options_file, weight_file)

    print("3.Train")
    model = train(model, iterator, train_data, val_data)

    print("4.Evaluate")
    evaluate(vocab, model, test_data, test_y)


if __name__ == '__main__':
    main()