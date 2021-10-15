from torchtext import data
import numpy as np


def pre_processing(train, valid, test, text, label, device, batch_size):
    text.build_vocab(train)
    label.build_vocab(train)

    train_iter, val_iter = data.BucketIterator.splits((train, valid), batch_size=batch_size, device=device, sort_key=lambda x: len(x.text), sort_within_batch=False, repeat=False)
    test_iter = data.Iterator(test, batch_size=batch_size, device=device, shuffle=False, sort=False, sort_within_batch=False)

    return train_iter, val_iter, test_iter, text, label



def text_to_vector():
    vocab,embeddings = [],[]
    with open('glove.6B.50d.txt','rt') as fi:
        full_content = fi.read().strip().split('\n')
    for i in range(len(full_content)):
        i_word = full_content[i].split(' ')[0]
        i_embeddings = [float(val) for val in full_content[i].split(' ')[1:]]
        vocab.append(i_word)
        embeddings.append(i_embeddings)

    vocab_npa = np.array(vocab)
    embs_npa = np.array(embeddings)

    #insert '<pad>' and '<unk>' tokens at start of vocab_npa.
    vocab_npa = np.insert(vocab_npa, 0, '<pad>')
    vocab_npa = np.insert(vocab_npa, 1, '<unk>')
    print(vocab_npa[:10])

    pad_emb_npa = np.zeros((1,embs_npa.shape[1]))   #embedding for '<pad>' token.
    unk_emb_npa = np.mean(embs_npa,axis=0,keepdims=True)    #embedding for '<unk>' token.

    #insert embeddings for pad and unk tokens at top of embs_npa.
    embs_npa = np.vstack((pad_emb_npa,unk_emb_npa,embs_npa))
    print(embs_npa.shape)

    return embs_npa