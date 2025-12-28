import json
import re
import os
import torch
from typing import List, Dict, Tuple

class Vocabulary:
    def __init__(self, data_path: str, min_freq: int = 1):
        self.data_path = data_path
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        data_iter = raw if isinstance(raw, list) else raw.values()

        word_freq: Dict[str, int] = {}
        label_set = set()
        
        for item in data_iter:
            if "review" not in item or "domain" not in item:
                continue
                
            sent = self._preprocess_sentence(item["review"])
            for w in sent.split():
                word_freq[w] = word_freq.get(w, 0) + 1
            
            label_set.add(str(item["domain"]))
        
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        sorted_words = sorted(
            [item for item in word_freq.items() if item[1] >= min_freq], 
            key=lambda x: (-x[1], x[0])
        )
        
        base_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab_words = base_tokens + [w for w, _ in sorted_words]

        self.word2idx: Dict[str, int] = {w: i for i, w in enumerate(vocab_words)}
        self.idx2word: Dict[int, str] = {i: w for w, i in self.word2idx.items()}

        sorted_labels = sorted(label_set)
        self.label2idx: Dict[str, int] = {l: i for i, l in enumerate(sorted_labels)}
        self.idx2label: Dict[int, str] = {i: l for l, i in self.label2idx.items()}

        self.pad_idx = self.word2idx[self.pad_token]
        self.bos_idx = self.word2idx[self.bos_token]
        self.eos_idx = self.word2idx[self.eos_token]
        self.unk_idx = self.word2idx[self.unk_token]

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def num_labels(self) -> int:
        return len(self.label2idx)

    def _preprocess_sentence(self, sentence: str) -> str:
        if not isinstance(sentence, str): return "" # Phòng hờ data null
        s = sentence.lower()
        s = re.sub(r"https?://\S+|www\.\S+", " ", s)
        s = re.sub(r"[^0-9a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        unk = self.word2idx[self.unk_token]
        return [self.word2idx.get(t, unk) for t in tokens]

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.unk_token) for i in ids]

    def encode_sentence(self, sentence: str, add_bos_eos: bool = True) -> List[int]:
        sent = self._preprocess_sentence(sentence)
        tokens = sent.split()
        ids = self.tokens_to_ids(tokens)
        if add_bos_eos:
            bos = self.word2idx[self.bos_token]
            eos = self.word2idx[self.eos_token]
            ids = [bos] + ids + [eos]
        return ids

    def decode_ids(self, ids: List[int], remove_special: bool = True) -> str:
        tokens = self.ids_to_tokens(ids)
        if remove_special:
            specials = {self.pad_token, self.bos_token, self.eos_token}
            tokens = [t for t in tokens if t not in specials]
        return " ".join(tokens)

    def encode_batch(self, sentences: List[str], add_bos_eos: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        seqs = [self.encode_sentence(s, add_bos_eos) for s in sentences]
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long)
        
        if len(lengths) == 0: return torch.empty(0), torch.empty(0)

        max_len = lengths.max().item()
        pad_id = self.word2idx[self.pad_token]

        padded = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        for i, seq in enumerate(seqs):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        return padded, lengths

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        return torch.tensor([self.label2idx[str(l)] for l in labels], dtype=torch.long)

    def decode_labels(self, label_ids: List[int]) -> List[str]:
        return [self.idx2label[i] for i in label_ids]