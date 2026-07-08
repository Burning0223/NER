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
    def __init__(self,id2label,config):
        self.id2label=id2label
        self.config=config
        self.reset()
        
    def reset(self):
        self.tp=0
        self.fp=0
        self.fn=0
        self.entities_tp={}
        self.entities_fp={}
        self.entities_fn={}
        self.entities_support={}
    
    def get_entities(self,labels):
        if isinstance(labels[0], list):
            raise ValueError("传入了batch,不是单句")
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
            pred_entities=set(self.get_entities(pred_seq))
            true_entities=set(self.get_entities(true_seq))

            self.tp+=len(pred_entities&true_entities)
            self.fp+=len(pred_entities-true_entities)
            self.fn+=len(true_entities-pred_entities)

            for entity in pred_entities:
                entity_type=entity[0]
                if entity in true_entities:
                    self.entities_tp[entity_type]=self.entities_tp.get(entity_type,0)+1
                else:
                    self.entities_fp[entity_type]=self.entities_fp.get(entity_type,0)+1
            for entity in true_entities:
                entity_type=entity[0]
                if entity not in pred_entities:
                    self.entities_fn[entity_type]=self.entities_fn.get(entity_type,0)+1
                self.entities_support[entity_type]=self.entities_support.get(entity_type,0)+1


    def compute(self,tp,fp,fn):
        eps=1e-8
        precision=tp/(tp+fp+eps)
        recall=tp/(tp+fn+eps)
        f1=2*precision*recall/(precision+recall+eps)
        
        return precision,recall,f1


    def report(self):
        print(f"{'Entity':<10}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}")
        sum_p,sum_r,sum_f,count=0,0,0,0
        weighted_p,weighted_r,weighted_f,total_suppot=0,0,0,0
        entity_types=set()
        for label in self.id2label.values():
            if label!="O":
                entity_types.add(label.split("-")[1])
        for entity_type in entity_types:
            tp=self.entities_tp.get(entity_type,0)
            fp=self.entities_fp.get(entity_type,0)
            fn=self.entities_fn.get(entity_type,0)
            precision,recall,f1=self.compute(tp,fp,fn)
            support=self.entities_support.get(entity_type,0)
            print(f"{entity_type:<10}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}{support:<10}")
            
            sum_p+=precision
            sum_r+=recall
            sum_f+=f1
            count+=1

            weighted_p+=precision*support
            weighted_r+=recall*support
            weighted_f+=f1*support
            total_suppot+=support

        micro_avg_p,micro_avg_r,micro_avg_f=self.compute(self.tp,self.fp,self.fn)
        macro_avg_p=sum_p/count
        macro_avg_r=sum_r/count
        macro_avg_f=sum_f/count
        weighted_avg_p=weighted_p/total_suppot
        weighted_avg_r=weighted_r/total_suppot
        weighted_avg_f=weighted_f/total_suppot
        print(f"\n{'micro avg':<10}{micro_avg_p:<15.2f}{micro_avg_r:<15.2f}{micro_avg_f:<15.2f}{total_suppot:<10}")
        print(f"\n{'macro avg':<10}{macro_avg_p:<15.2f}{macro_avg_r:<15.2f}{macro_avg_f:<15.2f}{total_suppot:<10}")
        print(f"\n{'weighted avg':<10}{weighted_avg_p:<15.2f}{weighted_avg_r:<15.2f}{weighted_avg_f:<15.2f}{total_suppot:<10}")



