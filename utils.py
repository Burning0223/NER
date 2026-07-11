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
        self.pred_sum=0
        self.true_sum=0
        self.entities_tp={}
        self.entities_pred={}
        self.entities_true={}
    
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
    def calculate(self,preds,labels):
        
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
            self.pred_sum+=len(pred_entities)
            self.true_sum+=len(true_entities)

            for entity in pred_entities:
                entity_type=entity[0]
                if entity in true_entities:
                    self.entities_tp[entity_type]=self.entities_tp.get(entity_type,0)+1
                self.entities_pred[entity_type]=self.entities_pred.get(entity_type,0)+1
            for entity in true_entities:
                entity_type=entity[0]
                self.entities_true[entity_type]=self.entities_true.get(entity_type,0)+1


    def compute(self,tp,pred_sum,true_sum):
        eps=1e-8
        precision=tp/(pred_sum+eps)
        recall=tp/(true_sum+eps)
        f1=2*precision*recall/(precision+recall+eps)
        
        return precision,recall,f1


    def report(self):
        print(f"{'Entity':<10}{'Precision':<15}{'Recall':<15}{'F1-Score':<15}{'Support':<10}")
        sum_p,sum_r,sum_f,count=0,0,0,0
        weighted_p,weighted_r,weighted_f=0,0,0
        entity_types=set()
        for label in self.id2label.values():
            if label!="O":
                entity_types.add(label.split("-")[1])
        for entity_type in entity_types:
            tp=self.entities_tp.get(entity_type,0)
            pred_num=self.entities_pred.get(entity_type,0)
            true_num=self.entities_true.get(entity_type,0)
            precision,recall,f1=self.compute(tp,pred_num,true_num)
            print(f"{entity_type:<10}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}{true_num:<10}")
            
            sum_p+=precision
            sum_r+=recall
            sum_f+=f1
            count+=1

            weighted_p+=precision*true_num
            weighted_r+=recall*true_num
            weighted_f+=f1*true_num

        micro_avg_p,micro_avg_r,micro_avg_f=self.compute(self.tp,self.pred_sum,self.true_sum)
        macro_avg_p=sum_p/count
        macro_avg_r=sum_r/count
        macro_avg_f=sum_f/count
        weighted_avg_p=weighted_p/self.true_sum
        weighted_avg_r=weighted_r/self.true_sum
        weighted_avg_f=weighted_f/self.true_sum
        print(f"{'micro avg':<10}{micro_avg_p:<15.2f}{micro_avg_r:<15.2f}{micro_avg_f:<15.2f}{self.true_sum:<10}")
        print(f"{'macro avg':<10}{macro_avg_p:<15.2f}{macro_avg_r:<15.2f}{macro_avg_f:<15.2f}{self.true_sum:<10}")
        print(f"{'weighted avg':<10}{weighted_avg_p:<15.2f}{weighted_avg_r:<15.2f}{weighted_avg_f:<15.2f}{self.true_sum:<10}")



