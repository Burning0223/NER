# 中文命名实体识别（NER）任务
## 项目简介
本项目基于 PyTorch 和 Hugging Face Transformers 框架，实现中文命名实体识别（Named Entity Recognition, NER）任务。
项目分别使用：
- `bert-base-chinese`
- `chinese-bert-wwm`
两种中文预训练 BERT 模型，在：
- Weibo NER
- MSRA NER
两个数据集上进行训练和评估。
## 项目结构
``` 
├── config_exp         # 参数配置文件
│   ├── config_exp1.json
│   ├── config_exp2.json
│   ├── config_exp3.json
│   └── config_exp4.json
├── utils.py          # 功能类
├── model.py           # BERT 模型定义
├── data_process.py    # 数据预处理
├── train.py     # 训练器以及程序入口       
├── requirements.txt   # 环境依赖
└── README.md          # 项目说明
```
## 数据集
### WeiBo
| 标签 | 标签ID | 含义 |
|---|---|---|
| O | 0 | 非实体 |
| B-GPE.NAM | 1 | 地缘政治实体专名开始 |
| B-GPE.NOM | 2 | 地缘政治实体普通名词开始 |
| B-LOC.NAM | 3 | 地点专名开始 |
| B-LOC.NOM | 4 | 地点普通名词开始 |
| B-ORG.NAM | 5 | 组织机构专名开始 |
| B-ORG.NOM | 6 | 组织机构普通名词开始 |
| B-PER.NAM | 7 | 人名专名开始 |
| B-PER.NOM | 8 | 人物普通名词开始 |
| I-GPE.NAM | 9 | 地缘政治实体专名内部 |
| I-GPE.NOM | 10 | 地缘政治实体普通名词内部 |
| I-LOC.NAM | 11 | 地点专名内部 |
| I-LOC.NOM | 12 | 地点普通名词内部 |
| I-ORG.NAM | 13 | 组织机构专名内部 |
| I-ORG.NOM | 14 | 组织机构普通名词内部 |
| I-PER.NAM | 15 | 人名专名内部 |
| I-PER.NOM | 16 | 人物普通名词内部 |
### MSRA
| 标签 | 标签ID | 含义 |
|---|---|---|
| O | 0 | 非实体 |
| B-LOC | 1 | 地点实体开始 |
| B-ORG | 2 | 组织机构实体开始 |
| B-PER | 3 | 人物实体开始 |
| I-LOC | 4 | 地点实体内部 |
| I-ORG | 5 | 组织机构实体内部 |
| I-PER | 6 | 人物实体内部 |
## 对齐策略
(1)特殊标记处理：[CLS]、[SEP]和填充标记的word_idx为None，将其标签设为-100，训练时这些位置会被损失函数忽略。
(2)新词开始：当word_idx变化时，表示进入新词，赋予该词对应的原始标签。
(3)同一词延续：同一词的后续子词标记为-100，避免重复标注。
## 环境依赖
``` 
pip install -r requirements.txt
``` 
## 模型参数设置（MSRA数据集为例）
``` 
    "model_path":"bert-base-chinese",
    "max_length":512,
    "num_epochs":20,
    "batch_size":16,
    "learning_rate":5e-6,
    "num_classes":7,
    "data_path":"./data_path/MSRA/",
    "dropout":0.1,
    "random":42
```
## 实验结果
### 1.MSRA数据集
#### (1)bert-base-chinese
运行命令：
``` 
python train.py config_exp/config_exp1.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
PER       0.93           0.94           0.93           473
LOC       0.87           0.89           0.88           647
ORG       0.83           0.83           0.83           287
micro avg 0.88           0.89           0.89           1407
macro avg 0.88           0.89           0.88           1407
weighted avg0.88           0.89           0.89           1407
```
<img width="2214" height="1135" alt="image" src="https://github.com/user-attachments/assets/c1d27587-76e5-4e8d-a4a4-61fdc74d1104" />

#### (2)chinese-bert-wwm
运行命令：
``` 
python train.py config_exp/config_exp3.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
LOC       0.89           0.89           0.89           647
ORG       0.89           0.84           0.86           287
PER       0.94           0.93           0.93           473
micro avg 0.90           0.89           0.90           1407
macro avg 0.90           0.89           0.90           1407
weighted avg0.90           0.89           0.90           1407
``` 
<img width="2230" height="1142" alt="image" src="https://github.com/user-attachments/assets/9b7f2c93-97aa-4fb6-9df9-99e1b482a903" />

### 2.weibo数据集
#### (1)bert-base-chinese
运行命令：
``` 
python train.py config_exp/config_exp2.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
ORG.NOM   0.00           0.00           0.00           17
GPE.NOM   0.00           0.00           0.00           2
LOC.NAM   0.67           0.11           0.18           19
PER.NOM   0.71           0.74           0.72           170
PER.NAM   0.71           0.80           0.75           111
GPE.NAM   0.71           0.87           0.78           47
ORG.NAM   0.37           0.41           0.39           39
LOC.NOM   0.00           0.00           0.00           9
micro avg 0.67           0.66           0.67           414
macro avg 0.40           0.37           0.35           414
weighted avg0.63           0.66           0.63           414
``` 
<img width="2243" height="1150" alt="image" src="https://github.com/user-attachments/assets/b40b3df6-3b79-4a52-bc45-0abb6f6af04e" />

#### (2)chinese-bert-wwm
运行命令：
``` 
python train.py config_exp/config_exp4.json
```

``` 
Entity    Precision      Recall         F1-Score       Support
ORG.NAM   0.41           0.41           0.41           39
GPE.NAM   0.63           0.85           0.73           47
PER.NOM   0.69           0.72           0.70           170
LOC.NOM   0.33           0.11           0.17           9
ORG.NOM   0.00           0.00           0.00           17
GPE.NOM   0.00           0.00           0.00           2
PER.NAM   0.70           0.79           0.74           111
LOC.NAM   0.50           0.21           0.30           19
micro avg 0.65           0.66           0.65           414
macro avg 0.41           0.39           0.38           414
weighted avg0.61           0.66           0.63           414
``` 
<img width="2197" height="1136" alt="image" src="https://github.com/user-attachments/assets/dd19e2f2-b46a-47b7-8769-b30a0b1e748d" />
