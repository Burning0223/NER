import argparse
import random
import numpy as np
import swanlab
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from utils import NER_Config,LabelManager,Metrics
import os
from data_process import NERDataset
from torch.utils.data import DataLoader
from model import NER_Bert
parser = argparse.ArgumentParser(description='exp config path')
parser.add_argument('exp_config')
args = parser.parse_args()
def random_seed(seed):
    random .seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

class Trainer():
    def __init__(self,config,model,optimizer,metric,scheduler):
        self.config=config
        self.model=model.to(device)
        self.optimizer=optimizer
        self.loss_fn=nn.CrossEntropyLoss(ignore_index=-100)
        self.metric=metric
        self.scheduler=scheduler

        self.experiment_name=f"max_len_{self.config.max_length}_num_epochs_{self.config.num_epochs}_bs_{self.config.batch_size}_lr_{self.config.learning_rate}"
        self.experiment_dir=os.path.join("experiment",self.experiment_name)
        os.makedirs(self.experiment_dir,exist_ok=True)
        swanlab.init(project="NER",
             experiment_name=self.experiment_name,
             config=self.config.config_dict,
             mode="offline")
        
    def train(self,dataloader):
        self.model.train()
        self.metric.reset()
        total_loss=0.0

        for batch in dataloader:
            batch={k:v.to(device) for k,v in batch.items()}
            self.optimizer.zero_grad()
            logits,labels=self.model(**batch)
            preds=torch.argmax(logits,dim=-1)
            loss=self.loss_fn(logits.view(-1,self.config.num_classes),labels.view(-1))
            total_loss+=loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.metric.calculate_tp_fp_fn(preds.cpu().tolist(),labels.cpu().tolist())
        ave_loss=total_loss/len(dataloader)
        train_precision,train_recall,train_f1=self.metric.compute(self.metric.tp,self.metric.fp,self.metric.fn)
        self.metric.report()
        return ave_loss,train_precision,train_recall,train_f1
    
    def dev(self,dataloader):
        self.model.eval()
        self.metric.reset()
        total_loss=0.0
        with torch.no_grad():
            for batch in dataloader:
                batch={k:v.to(device) for k,v in batch.items()}
                logits,labels=self.model(**batch)
                preds=torch.argmax(logits,dim=-1)
                loss=self.loss_fn(logits.view(-1,self.config.num_classes),labels.view(-1))
                total_loss+=loss.item()
                self.metric.calculate_tp_fp_fn(preds.cpu().tolist(),labels.cpu().tolist())
        ave_loss=total_loss/len(dataloader)
        dev_precision,dev_recall,dev_f1=self.metric.compute(self.metric.tp,self.metric.fp,self.metric.fn)
        return ave_loss,dev_precision,dev_recall,dev_f1
    
    def test(self,dataloader,best_model_path):
        if not os.path.exists(best_model_path):
            print("未找到最优模型，无法进行测试！")
            return
        else:
            checkpoint=torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        self.metric.reset()
        total_loss=0.0
        with torch.no_grad():
            for batch in dataloader:
                batch={k:v.to(device) for k,v in batch.items()}
                logits,labels=self.model(**batch)
                preds=torch.argmax(logits,dim=-1)
                loss=self.loss_fn(logits.view(-1,self.config.num_classes),labels.view(-1))
                total_loss+=loss.item()
                self.metric.calculate_tp_fp_fn(preds.cpu().tolist(),labels.cpu().tolist())
        ave_loss=total_loss/len(dataloader)
        test_precision,test_recall,test_f1=self.metric.compute(self.metric.tp,self.metric.fp,self.metric.fn)
        self.metric.report()
        return ave_loss,test_precision,test_recall,test_f1

    def save_checkpoint(self,train_dataloader,dev_dataloader,test_dataloader):
        dev_best_f1=0.0
        for epoch in range(self.config.num_epochs):
            print(f"Epoch{epoch+1}")
            train_loss,train_precision,train_recall,train_f1=self.train(train_dataloader)
            dev_loss,dev_precision,dev_recall,dev_f1=self.dev(dev_dataloader)
            print(f"训练集损失:{train_loss:.4f}")
            print(f"训练集精准率:{train_precision:.4f}")
            print(f"训练集召回率:{train_recall:.4f}")
            print(f"训练集f1分数:{train_f1:.4f}")

            print(f"验证集损失:{dev_loss:.4f}")
            print(f"验证集精准率:{dev_precision:.4f}")
            print(f"验证集召回率:{dev_recall:.4f}")
            print(f"验证集f1分数:{dev_f1:.4f}")

            checkpoint_name=f"check_epoch_{epoch+1}.pt"
            checkpoint_path=os.path.join(self.experiment_dir,checkpoint_name)
            checkpoint={
                'epoch':epoch,
                'model':self.model.state_dict(),
                'loss':dev_loss,
                'f1':dev_f1
            }
            if self.optimizer is not None:
                checkpoint['optimizer']=self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler']=self.scheduler.state_dict()
            if dev_f1>=dev_best_f1:
                dev_best_f1=dev_f1
                best_model_path=checkpoint_path
                print(f"当前模型最优f1分数为:{dev_best_f1}")
                torch.save(checkpoint,checkpoint_path)
                print(f"保存最佳模型：{best_model_path}")

            swanlab.log({
                "epoch":epoch,
                "训练集损失":train_loss,
                "训练集f1分数":train_f1,
                "验证集损失":dev_loss,
                "验证集f1分数":dev_f1
            })
        print(f"模型最优f1分数为:{dev_best_f1}")
        test_loss,test_precision,test_recall,test_f1=self.test(test_dataloader,best_model_path)
        print(f"测试集损失:{test_loss:.4f}")
        print(f"测试集精准率:{test_precision:.4f}")
        print(f"测试集召回率:{test_recall:.4f}")
        print(f"测试集f1分数:{test_f1:.4f}")
        swanlab.log({
            "测试集损失":test_loss,
            "测试集f1分数":test_f1
        })

def main(config_path):
    config=NER_Config(config_path)
    random_seed(config.random)

    label_manager=LabelManager(config.data_path)
    label2id, id2label=label_manager.load_labels_mapping()
    if len(label2id)!=config.num_classes:
        print(f"标签类别数不匹配！实际类别数：{len(label2id)}，配置文件中的类别数：{config.num_classes}")
    train_dataset=NERDataset(dataset_type="train",label2id=label2id,config=config)
    dev_dataset=NERDataset(dataset_type="dev",label2id=label2id,config=config)
    test_dataset=NERDataset(dataset_type="test",label2id=label2id,config=config)

    train_dataloader=DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=train_dataset.collate_fn)
    dev_dataloader=DataLoader(dev_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=dev_dataset.collate_fn)
    test_dataloader=DataLoader(test_dataset,batch_size=config.batch_size,shuffle=False,collate_fn=test_dataset.collate_fn)
    model=NER_Bert(config)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=config.learning_rate
        )
    total_steps = len(train_dataloader) * config.num_epochs
    num_warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)
    metric=Metrics(id2label,config)
    trainer=Trainer(config,model,optimizer,metric,scheduler)
    trainer.save_checkpoint(train_dataloader,dev_dataloader,test_dataloader)

if __name__=="__main__":
    main(args.exp_config)


              





