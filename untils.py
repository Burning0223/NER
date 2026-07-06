import json
import os

class NER_Config:
    def __init__(self,config_path):
        self.config_dict=self.load_config(config_path)
        for key,value in self.config_dict.items():
            setattr(self,key,value)

    def load_config(self,config_path):
        if not os.path.exists(config_path):
            raise FileExistsError(f"配置文件不存在: {config_path}")
        with open(config_path,'r',encoding='utf-8') as f:
            return json.load(f)

class LabelManager:
    def __init__(self,data_path):
        self.mapping_path=os.path.join(data_path,'label_mapping.json')
        self.label2id,self.id2label=self.load_labels_mapping()
    def load_labels_mapping(self):
        if not os.path.exists(self.mapping_path):
            raise FileExistsError(f"标签映射文件不存在: {self.mapping_path}")
        with open(self.mapping_path,'r',encoding='utf-8') as f:
            mapping=json.load(f)
        label2id={k:v for k,v in mapping['label2id'].items()}
        id2label={int(k):v for k,v in mapping['id2label'].items()}
        return label2id,id2label

class Metrics:
    def __init__(self,id2label):
        self.id2label=id2label
        self.tp=0
        self.fp=0
        self.fn=0
    
    def get_entities(self,labels):
        if isinstance(labels[0], list):
            raise ValueError("传入了batch，不是单句")
        entity_type=None
        entities=[]
        start=None
        for i,label in enumerate(labels):
            if label.startswith("B-"):
                if entity_type is not None:
                    entities.append((entity_type,start,i-1))
                entity_type=label[2:]
                start=i
            elif label.startswith("I-"):
                continue
            else:
                if entity_type is not None:
                    entities.append((entity_type,start,i-1))
                    entity_type=None
                    start=None
        if entity_type is not None:
            entities.append((entity_type,start,len(labels)-1))

        return entities
    def calculate_tp_fp_fn(self,preds,labels):
        
        p_labels=[
            [self.id2label[int(p)] for p,t in zip(pred_seq,true_seq) if t!=-100]
            for pred_seq,true_seq in zip(preds,labels)
        ]
        t_labels=[
            [self.id2label[int(t)] for p,t in zip(pred_seq,true_seq) if t!=-100]
            for pred_seq,true_seq in zip(preds,labels)
        ]
        for pred_seq,true_seq in zip(p_labels,t_labels):
            pred_entities=self.get_entities(pred_seq)
            true_entities=self.get_entities(true_seq)

        self.tp+=len(set(pred_entities)&set(true_entities))
        self.fp+=len(set(pred_entities)-set(true_entities))
        self.fn+=len(set(true_entities)-set(pred_entities))

    def compute(self):
        eps=1e-8
        precision=self.tp/(self.tp+self.fp+eps)
        recall=self.tp/(self.tp+self.fn+eps)
        f1=2*precision*recall/(precision+recall+eps)
        
        return precision,recall,f1

