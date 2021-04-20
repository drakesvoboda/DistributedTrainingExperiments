import torch
import torch.nn as nn
import torch.functional as F

import pickle

from transformers import (
    BertModel,
    BertTokenizer,
    BertConfig
)

from collections import Counter

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

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
        for line in f:
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
            for line in f:
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

def load_dataset(token2id, state2id, pad_token_id):
    fullset = dataset.PartOfSpeechDataset('wsj1-18.training', token2id, state2id, pad_token_id)
    train_size = int(0.9 * len(fullset))
    valid_size = len(fullset) - train_size
    trainset, validset = torch.utils.data.random_split(fullset, [train_size, valid_size], generator=torch.Generator().manual_seed(42))
    return trainset, validset

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

class Tagger(nn.Module):
    def __init__(self, tag_dict, token_dict=None):
        super(Tagger, self).__init__()
        self.tag_dict = tag_dict
        self.token_dict = token_dict

    def forward(self, x): 
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, model_file, *args, **kwargs):
        loaded = torch.load(model_file)
        tag_dict = pickle.loads(loaded["tag_dict"])
        token_dict = pickle.loads(loaded["token_dict"]) if "token_dict" in loaded else None
        model = cls(tag_dict, token_dict, *args, **kwargs)
        model.load_state_dict(loaded)
        return model

    def state_dict(self):
        state_dict = {
            "model": super().state_dict(),
            "tag_dict": pickle.dumps(self.tag_dict)
        }

        if self.token_dict is not None:
            state_dict["token_dict"] = pickle.dumps(self.token_dict)

        return state_dict

    def load_state_dict(self, state_dict):  
        super().load_state_dict(state_dict["model"])

    def convert_tokens_to_ids(self, tokens):
        return self.token_dict.convert_tokens_to_ids(tokens)

    def convert_tags_to_ids(self, tags):
        return self.tag_dict.convert_tokens_to_ids(tags)

    def convert_ids_to_tags(self, ids):
        return self.tag_dict.convert_ids_to_tokens(ids)

    @property
    def pad_token_id(self):
        return self.token_dict.pad_token_id if self.token_dict is not None else None

class LSTM(Tagger):
    def __init__(self, tag_dict, token_dict, embedding_dim, hidden_dim):
        super(LSTM, self).__init__(tag_dict, token_dict)
        self.embedding = nn.Embedding(len(token_dict), embedding_dim, padding_idx=self.token_dict.pad_token_id)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.cls = ClassificationHead(hidden_dim, len(tag_dict))

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        out = self.cls(out)
        
        return out

class Bert(Tagger):
    def __init__(self, tag_dict):
        super(Bert, self).__init__(tag_dict)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        config = BertConfig(hidden_size = 256,
                            intermediate_size = 256*4,
                            num_hidden_layers = 6,
                            num_attention_heads = 6)
        self.transformer = BertModel()
        self.cls = ClassificationHead(128, len(tag_dict))

    def forward(self, x):
        x = self.transformer(x)[0]
        x = self.cls(x)
        return x

    def convert_tokens_to_ids(self, tokens):
        return [self.tokenizer.encode(token.decode(), add_special_tokens=False)[0] for token in tokens]

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, model_file, *args, **kwargs):
        loaded = torch.load(model_file)
        tag_dict = pickle.loads(loaded["tag_dict"])
        model = cls(tag_dict, *args, **kwargs)
        model.load_state_dict(loaded)
        return model

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels

        self.shortcut_model = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.out_channels))
        
        self.blocks = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(self.out_channels),
        )

        self.activation = nn.ReLU(inplace=True)

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels
    
    def shortcut(self, x):
        if self.should_apply_shortcut: x = self.shortcut_model(x)
        return x

    def forward(self, x):
        x = self.blocks(x) + self.shortcut(x)
        x = self.activation(x)
        return x 

class ResNet(Tagger):
    def __init__(self, tag_dict, token_dict, embedding_dim):
        super(ResNet, self).__init__(tag_dict, token_dict)

        self.embedding = nn.Embedding(len(token_dict), embedding_dim, padding_idx=self.token_dict.pad_token_id)

        self.model = nn.Sequential(
            ResidualBlock(embedding_dim, embedding_dim),
            ResidualBlock(embedding_dim, embedding_dim),
            ResidualBlock(embedding_dim, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512)
        )

        self.cls = ClassificationHead(512, len(tag_dict))

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = self.model(x)
        x = x.permute(0,2,1)
        x = self.cls(x)
        return x

def TaggerAccuracy(output, label):
    output = output.view(-1, output.shape[-1])
    label = label.view(-1)
    mask = label != -100
    output = output[mask]
    label = label[mask]
    _, preds = torch.max(output, -1)
    batch_correct = torch.sum(preds == label).item()
    return batch_correct / label.shape[0]