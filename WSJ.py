from tqdm.auto import tqdm
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig
)

class TaggerLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        outputs = outputs.view(-1, self.num_classes)
        labels = labels.view(-1)

        mask = labels != -100

        outputs = outputs[mask]
        labels = labels[mask]

        # F.cross_entropy will apply softmax then negative log likelihood loss
        return F.cross_entropy(outputs, labels)

class Dictionary():
    def __init__(self, min_count=0, unk_token=None, pad_token=None):
        self.cnt = Counter()

        self.min_count = min_count
        self.token2idx = None
        self.idx2token = None

        self.unk_token = unk_token
        self.pad_token = pad_token

    def add_tokens(self, tokens):
        for token in tokens:
            self.cnt[token] += 1

    def finalize(self):
        self.idx2token = [tok for tok, count in self.cnt.items() if count >= self.min_count]

        if self.pad_token is not None:
            self.idx2token = [self.pad_token] + self.idx2token

        if self.unk_token is not None:
            self.idx2token = [self.unk_token] + self.idx2token

        self.token2idx = {tok: idx for idx, tok in enumerate(self.idx2token)}

        if self.pad_token is not None:
            self.pad_token_id = self.token2idx[self.pad_token]

        if self.unk_token is not None:
            self.unk_token_id = self.token2idx[self.unk_token]

    def __getitem__(self, idx):
        if self.idx2token is None: self.finalize()
        return self.idx2token[idx]

    def index(self, token):
        if self.token2idx is None: self.finalize()
        return self.token2idx[token] if self.unk_token == None else self.token2idx.get(token, self.unk_token_id)

    def __len__(self):
        return len(self.cnt)

    def convert_tokens_to_ids(self, tokens):
        return [self.index(token) for token in tokens]

    def convert_ids_to_tokens(self, token_ids):
        return [self[id].decode() for id in token_ids]

def make_dicts(data_file):
    token_dict = Dictionary(min_count=3, unk_token="UNKN", pad_token="[PAD]")
    tag_dict = Dictionary()

    with open(data_file, "rb") as f:
        for line in tqdm(f, desc='Making Dicts'):
            tokens = line.split()
            token_dict.add_tokens(tokens[::2])
            tag_dict.add_tokens(tokens[1::2])

    token_dict.finalize()
    tag_dict.finalize()

    return tag_dict, token_dict

class PartOfSpeechDataset(Dataset): 
    def __init__(self, data_file, token2id, state2id, pad_token_id):
        super(PartOfSpeechDataset, self).__init__()

        self.pad_token_id = pad_token_id

        self.observations = []
        self.labels = []

        with open(data_file, "rb") as f:
            for line in tqdm(f, desc='Loading Dataset'):
                tokens = line.split()
                self.observations.append(token2id(tokens[::2]))
                self.labels.append(state2id(tokens[1::2]))

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        input = self.observations[idx]
        label = self.labels[idx]
        return  torch.Tensor(input).long(), torch.Tensor(label).long()

    def collate(self, examples):
        inputs, labels = zip(*examples)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return inputs, labels

class TestPartOfSpeechDataset(Dataset): 
    def __init__(self, data_file, token2id, pad_token_id):
        super(TestPartOfSpeechDataset, self).__init__()
        self.pad_token_id = pad_token_id
        self.observations = [token2id(line.split()) for line in open(data_file, "rb")]

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        input = self.observations[idx]
        return torch.Tensor(input).long()

    def collate(self, examples):
        inputs = examples
        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad_token_id)    
        return inputs

def load_datasets(token2id, state2id, pad_token_id):
    fullset = PartOfSpeechDataset('wsj1-18.training', token2id, state2id, pad_token_id)
    train_size = int(0.9 * len(fullset))
    valid_size = len(fullset) - train_size
    trainset, validset = torch.utils.data.random_split(fullset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))

    def collate_fn(examples):
        inputs, labels = zip(*examples)

        inputs = pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)
        
        return inputs, labels

    return trainset, validset, collate_fn

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.cls = nn.Linear(input_dim, num_classes)
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.cls.bias = self.bias
    
    def forward(self, x):
        x = self.dropout(x)
        x = self.cls(x)

        return x

class LSTM(nn.Module):
    def __init__(self, tag_dict, token_dict, embedding_dim, hidden_dim):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(len(token_dict), embedding_dim, padding_idx=token_dict.pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.cls = ClassificationHead(hidden_dim, len(tag_dict))

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.cls(out)
        
        return out

def make_lstm():
    tag_dict, token_dict = make_dicts('wsj1-18.training')

    def convert_tokens_to_ids(tokens):
        return token_dict.convert_tokens_to_ids(tokens)

    def convert_tags_to_ids(tags):
        return tag_dict.convert_tokens_to_ids(tags)

    model = LSTM(tag_dict, token_dict, 128, 128)

    return model, convert_tokens_to_ids, convert_tags_to_ids, token_dict.pad_token_id, len(tag_dict)

class Bert(nn.Module):
    def __init__(self, tag_dict):
        super(Bert, self).__init__()
        config = BertConfig(hidden_size = 66,
                            intermediate_size = 66*2,
                            num_hidden_layers = 6,
                            num_attention_heads = 6)
        self.transformer = BertModel(config)
        self.cls = ClassificationHead(66, len(tag_dict))

    def forward(self, x):
        x = self.transformer(x)[0]
        x = self.cls(x)
        return x

def make_bert():
    tag_dict, token_dict = make_dicts('wsj1-18.training')

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def convert_tokens_to_ids(tokens):
        return [tokenizer.encode(token.decode(), add_special_tokens=False)[0] for token in tokens]

    def convert_tags_to_ids(tags):
        return tag_dict.convert_tokens_to_ids(tags)

    model = Bert(tag_dict)

    return model, convert_tokens_to_ids, convert_tags_to_ids, tokenizer.pad_token_id, len(tag_dict)

def TaggerAccuracy(output, label):
    output = output.view(-1, output.shape[-1])
    label = label.view(-1)
    mask = label != -100
    output = output[mask]
    label = label[mask]
    _, preds = torch.max(output, -1)
    batch_correct = torch.sum(preds == label).item()
    return batch_correct / label.shape[0]