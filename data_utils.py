import torch
from torch.utils.data import IterableDataset

import os
from datasets import load_dataset

class ShakespeareDataset(IterableDataset):
    def __init__(self, root, train=True, block_size=256):
        self.train = train
        self.block_size = block_size
        with open(os.path.join(root, 'shakespeare/input.txt'), 'r', encoding='utf-8') as f:
            text = f.read()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }

        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(0.9*len(text)) # first 90% will be train, rest validation
        if train:
            self.dataset =  data[:n]
        else:
            self.dataset = data[n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def __iter__(self):
        indices = torch.randperm(len(self.dataset) - self.block_size)
        for i in indices:
            x = self.dataset[i:i+self.block_size].clone()
            y = self.dataset[i+1:i+self.block_size+1].clone()
            yield x, y

    def __len__(self):
        return len(self.dataset) - self.block_size


class TextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, block_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

    def __iter__(self):
        buffer = []
        indices = torch.arange(len(self.dataset))
        for i in indices:
            example = self.dataset[i.unsqueeze(0)]
            if not example or not example.get('text'):
                continue
            assert isinstance(example['text'], list)
            tokens = self.tokenizer.encode(example['text'][0], add_special_tokens=False)
            if not tokens:
                continue
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                x = torch.tensor(buffer[:self.block_size], dtype=torch.long)
                y = torch.tensor(buffer[1:self.block_size+1], dtype=torch.long)
                buffer = buffer[self.block_size:]
                yield x, y

    def __len__(self):
        return len(self.dataset)    # incorrect estimation! (it should be # tokens/block size)

class WikiTextDataset(TextDataset):
    def __init__(self, root, tokenizer, train=True, block_size=256):
        split = 'train' if train else 'validation'
        dataset = load_dataset('wikitext', 'wikitext-103-v1', cache_dir=os.environ.get("HF_HOME"))[split]
        super(WikiTextDataset, self).__init__(dataset, tokenizer, block_size)

class OpenWebTextDataset(TextDataset):
    def __init__(self, root, tokenizer, train=True, block_size=256):
        split = 'train' if train else 'test'
        dataset = load_dataset("openwebtext", cache_dir=os.environ.get("HF_HOME"))
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)[split]
        super(OpenWebTextDataset, self).__init__(dataset, tokenizer, block_size)

class BookCorpusDataset(TextDataset):
    def __init__(self, root, tokenizer, train=True, block_size=256):
        split = 'train' if train else 'test'
        dataset = load_dataset('bookcorpus/bookcorpus', trust_remote_code=True)
        dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)[split]
        super(BookCorpusDataset, self).__init__(dataset, tokenizer, block_size)

class DataUtil:
    def __init__(self, train_loader, eval_loader):
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_loader_iter = iter(train_loader) if train_loader is not None else None
        self.eval_loader_iter = iter(eval_loader) if eval_loader is not None else None

    def get_batch(self, eval=False):
        if not eval or self.eval_loader_iter is None:
            try:
                x, y = next(self.train_loader_iter)
            except StopIteration:
                self.train_loader_iter = iter(self.train_loader)
                x, y = next(self.train_loader_iter)
        else:
            try:
                x, y = next(self.eval_loader_iter)
            except StopIteration:
                self.eval_loader_iter = iter(self.eval_loader)
                x, y = next(self.eval_loader_iter)

        return x, y
    