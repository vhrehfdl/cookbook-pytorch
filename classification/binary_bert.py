from pathlib import Path
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
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.nn import util as nn_util
from allennlp.nn.util import get_text_field_mask
from allennlp.training.trainer import Trainer
from overrides import overrides
from scipy.special import expit
from tqdm import tqdm


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    testing=True,
    seed=1,
    batch_size=64,
    lr=3e-4,
    epochs=2,
    hidden_sz=64,
    max_seq_len=100,  # necessary to limit memory usage
    max_vocab_size=100000,
)


class LoadData(DatasetReader):
    def __init__(self, tokenizer: Callable[[str], List[str]] = lambda x: x.split(),
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_seq_len: Optional[int] = config.max_seq_len) -> None:
        super().__init__(lazy=False)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_seq_len = max_seq_len

    @overrides
    def text_to_instance(self, tokens: List[Token], labels: np.ndarray = None) -> Instance:
        sentence_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": sentence_field}

        if labels is None:
            labels = np.zeros(len(label_cols))

        label_field = ArrayField(array=labels)
        fields["label"] = label_field

        return Instance(fields)

    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)

        if config.testing:
            df = df.head(1000)

        for i, row in df.iterrows():
            yield self.text_to_instance(
                [Token(x) for x in self.tokenizer(row["text"])],
                row[label_cols].values,
            )


class BaselineModel(Model):
    def __init__(self, word_embeddings: TextFieldEmbedder, encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.word_embeddings = word_embeddings
        self.encoder = encoder
        self.projection = nn.Linear(self.encoder.get_output_dim(), 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, tokens: Dict[str, torch.Tensor], label: torch.Tensor) -> torch.Tensor:
        mask = get_text_field_mask(tokens)
        embeddings = self.word_embeddings(tokens)
        state = self.encoder(embeddings, mask)
        class_logits = self.projection(state)

        output = {"class_logits": class_logits}
        output["loss"] = self.loss(class_logits, label)

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



USE_GPU = torch.cuda.is_available()
DATA_ROOT = Path("../") / "Data"
torch.manual_seed(config.seed)
label_cols = ["label"]
from allennlp.data.token_indexers import PretrainedBertIndexer
token_indexer = PretrainedBertIndexer(
    pretrained_model="bert-base-uncased",
    max_pieces=config.max_seq_len,
    do_lowercase=True,
 )
reader = LoadData(tokenizer=tokenizer, token_indexers={"tokens": token_indexer})
train_ds, test_ds = (reader.read(DATA_ROOT / fname) for fname in ["binary_train_data.csv", "binary_test_data.csv"])
vocab = Vocabulary()
iterator = BucketIterator(batch_size=config.batch_size, sorting_keys=[("tokens", "num_tokens")])
iterator.index_with(vocab)
batch = next(iter(iterator(train_ds)))

print(batch["tokens"]["tokens"].shape)


from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased", top_layer_only=True)
word_embeddings: TextFieldEmbedder = BasicTextFieldEmbedder({"tokens": bert_embedder}, allow_unmatched_keys=True)

BERT_DIM = word_embeddings.get_output_dim()


class BertSentencePooler(Seq2VecEncoder):
    def forward(self, embs: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        # extract first token tensor
        return embs[:, 0]

    @overrides
    def get_output_dim(self) -> int:
        return BERT_DIM


encoder = BertSentencePooler(vocab)
model = BaselineModel(word_embeddings, encoder)

if USE_GPU:
    model.cuda()
else:
    model

batch = nn_util.move_to_device(batch, 0 if USE_GPU else -1)
tokens = batch["tokens"]
labels = batch
mask = get_text_field_mask(tokens)
embeddings = model.word_embeddings(tokens)
state = model.encoder(embeddings, mask)
class_logits = model.projection(state)
loss = model(**batch)["loss"]
loss.backward()
[x.grad for x in list(model.encoder.parameters())]
optimizer = optim.Adam(model.parameters(), lr=config.lr)
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    iterator=iterator,
    train_dataset=train_ds,
    cuda_device=0 if USE_GPU else -1,
    num_epochs=config.epochs
)
metrics = trainer.train()
# iterate over the dataset without changing its order
seq_iterator = BasicIterator(batch_size=64)
seq_iterator.index_with(vocab)

predictor = Predictor(model, seq_iterator, cuda_device=0 if USE_GPU else -1)
train_preds = predictor.predict(train_ds)
test_preds = predictor.predict(test_ds)
