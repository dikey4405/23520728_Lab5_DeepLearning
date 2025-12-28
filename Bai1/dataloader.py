import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ViOCD_Dataset(Dataset):
    def __init__(self, data_path: str, vocab):
        super().__init__()
        self.vocab = vocab
        
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        
        items = raw if isinstance(raw, list) else raw.values()
        
        self.data = []
        for item in items:
            if "review" in item and "domain" in item:   
                self.data.append({
                    "review": item["review"],
                    "domain": str(item["domain"])
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_ids_list = self.vocab.encode_sentence(item["review"])
        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long)
        
        label_id = self.vocab.label2idx[item["domain"]]
        label_tensor = torch.tensor(label_id, dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "label_ids": label_tensor
        }

def collate_fn(items: list) -> dict:
    """
    Hàm gom batch độc lập.
    items: List các dict trả về từ __getitem__
    """
    input_ids = [item["input_ids"] for item in items]
    
    max_len = max(len(ids) for ids in input_ids)
    
    input_ids = [
        F.pad(
            input,
            pad = (0, max(0, max_len - input.shape[0])), 
            mode = "constant",
            value = 0 
        ).unsqueeze(0) for input in input_ids] 
    
    label_ids = [item["label_ids"].unsqueeze(0) for item in items]

    return {
        "input_ids": torch.cat(input_ids, dim=0), 
        "label_ids": torch.cat(label_ids, dim=0)  
    }