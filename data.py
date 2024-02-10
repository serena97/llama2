from torch.utils.data import Dataset
import transformers as t
import pandas as pd
from datasets import load_dataset

model_dir = "./llama/llama-2-7b-chat-hf"

class TrainDataset(Dataset):
    def __init__(self):
        self.tokenizer = t.AutoTokenizer.from_pretrained(model_dir)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        self.df = self.get_data()
        self.ds = self.df.apply(lambda row: self.tokenize(row), axis=1)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        data = self.ds[idx]
        # print(f"len(data['input_ids']) {len(data['input_ids'])} , idx: {idx}")
        return data
    
    def get_data(self):
        rows = load_dataset("heliosbrahma/mental_health_chatbot_dataset")['train']
        df_new = pd.DataFrame(rows)
        return df_new
    
    def tokenize(self, elm):
        # print(elm['text'])
        res = self.tokenizer(elm['text'])
        res['input_ids'].append(self.tokenizer.eos_token_id)
        res['attention_mask'].append(1)
        res['labels'] = res['input_ids'].copy()
        return res
    
    def max_seq_len(self):
        return max([len(elm('input_ids')) for elm in self.ds])
    
# ds = TrainDataset()
# print(ds.tokenizer.decode(ds[1]['input_ids']))
    