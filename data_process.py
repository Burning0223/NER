from torch.utils.data import Dataset
import torch
from transformers import BertTokenizerFast

class NERDataset(Dataset):
    def __init__(self,dataset_type,label2id,config):
        super().__init__()
        self.label2id=label2id
        self.config=config
        self.sentences,self.labels=self.load_data(dataset_type)
        self.tokenizer=BertTokenizerFast.from_pretrained(config.model_path)


    def load_data(self,dataset_type):
        if dataset_type == "train":
            file_path = f"{self.config.data_path}/train.txt"
        elif dataset_type == "dev":
            file_path = f"{self.config.data_path}/dev.txt"
        elif dataset_type == "test":
            file_path = f"{self.config.data_path}/test.txt"
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        words=[]
        tags=[]
        sentences=[]
        labels=[]
        with open(file_path,'r',encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    if words:
                        sentences.append(words)
                        labels.append(tags)
                        words=[]
                        tags=[]
                    continue

                parts=line.split()
                try:
                    word=parts[0]
                    tag=parts[1]
                    assert isinstance(word,str)
                    assert isinstance(tag,str)
                except Exception:
                    continue
                else:
                    words.append(word)
                    if tag in self.label2id:
                        tag_id=self.label2id[tag]
                    else:
                        print(f"警告：标签{tag}未在label2id中找到,默认为-100")
                        tag_id=-100
                    tags.append(tag_id)
        if words:
            sentences.append(words)
            labels.append(tags)
        return sentences,labels
    
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        sentence=self.sentences[index]
        label=self.labels[index]
        return sentence,label

    def collate_fn(self,batch):
        sentences,labels=zip(*batch)
        encoding=self.tokenizer(sentences,return_tensors='pt',max_length=self.config.max_length,
                                padding=True,truncation=True, is_split_into_words=True)
        batch_labels=[]
        for i,label in enumerate(labels):
            word_ids=encoding.word_ids(batch_index=i)
            get_labels=[]
            previous_word_id=None
            for word_id in word_ids:
                if word_id is None:
                    get_labels.append(-100)
                elif word_id!=previous_word_id:
                    get_labels.append(label[word_id])
                else:
                    get_labels.append(-100)
                previous_word_id=word_id
            batch_labels.append(get_labels)        
        return {
            'input_ids':encoding['input_ids'],
            'attention_mask':encoding['attention_mask'],
            'labels':torch.tensor(batch_labels)
        }