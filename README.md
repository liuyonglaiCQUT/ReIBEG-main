# ReIBEG

**ReIBEG**: Relation-aware Information Bottleneck for Few-shot Knowledge Graph Completion

---

## Key Features

**Relation Aggregator**

**Relation-Based Information Bottleneck**

**EMAttentionPooler**

---

## Environment

- **Python:** 3.8.20
- **Operating System:** Ubuntu 22.04.5
- **GPU:** RTX3090 24GB Memory

---

## Dataset & Checkpoints

### Original Datasets

- **NELL:** [NELL Dataset](https://github.com/xwhan/One-shot-Relational-Learning)
- **FB15K-237:** [FB15K-237 Dataset](https://github.com/SongW-SW/REFORM)

### Processed Dataset

- [Processed Dataset](https://drive.google.com/drive/folders/1QQPn8-lENxaJE-ebrcAJ7-HwqPegGCRw?usp=sharing)

> **Note:** Download and extract the datasets to the root folder of the project.

### Checkpoints are available for download here:

- [Learned Checkpoint](https://drive.google.com/file/d/1zn9TmPCCUjooZsJq3LSntEDcCAT1PVul/view?usp=drive_link))

**Directory Structure Example:**

```bash
ReIBEG
├── NELL
├── FB15K
└── ...
```


## How to Run

### Training

#### For the NELL Dataset

```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix nell_reibeg --device 0  --batch_size 64 --g_batch 1024
```

#### For the FB15K Dataset

```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix fb15k_reibeg --weight_decay 1e-5 --device 1 --batch_size 64 --g_batch 1024
```

#### For the Wiki Dataset
```bash
python main.py --dataset Wiki-One --data_path ./Wiki --few 5 --data_form Pre-Train --prefix wiki_reibeg --device 0 --batch_size 64 --g_batch 1024
```
### Testing

Use pre-trained checkpoints for evaluation.

```bash
ReIBEG
├── state
│   ├── state_dict_fb15k_best_658
│   └── state_dict_nell_best_553
```

#### Testing on the NELL Dataset 

```bash
python main.py --dataset NELL-One --data_path ./NELL --few 5 --data_form Pre-Train --prefix state_dict_nell_best_553 --state_dict_filename state_dict_nell_best_553 --weight_decay 1e-7 --device 0 --batch_size 64 --g_batch 1024  --learning_rate 1e-3 --step test
```

#### Testing on the FB15K Dataset

```bash
python main.py --dataset FB15K-One --data_path ./FB15K --few 5 --data_form Pre-Train --prefix state_dict_fb15k_best_658 --state_dict_filename state_dict_fb15k_best_658 --weight_decay 1e-5 --device 1 --batch_size 64 --g_batch 1024 --learning_rate 5e-3 --step test
```

---

## Experimental Results

### 5-shot FKGC Results

| Dataset       | MRR   | Hits@10 | Hits@5 | Hits@1 |
|---------------|-------|---------|--------|--------|
| **NELL**      | 0.553 | 0.621   | 0.585  | 0.522  |
| **FB15K-237** | 0.658 | 0.858   | 0.801  | 0.526  |



## Acknowledgements

This repository is based on [ReCDAP-FKGC](https://github.com/hou27/ReCDAP-FKGC). We appreciate the efforts of the original authors and thank them for their excellent work.