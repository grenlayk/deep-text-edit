# text-deep-fake design doc

This project aims to implement neural network architecture, described in [Text Style Brush Paper](https://arxiv.org/pdf/2106.08385.pdf).

## File storage

We use Yandex.disk with 1TB storage to store dataset, logs and checkpoints.

## How to run ?

- Choose config file
- Run `python run.py configs/config.py`

## Repo structure

```
├── run.py                  <- [entry point]
│
├── requirements.txt        <- [necessary requirements]
│
├── src                     <- [project source code]
│   ├── training
│   │   ├── simple.py           <- [Template Trainer]
│   │   ├── gan.py
│   │   ├── ...
│   │
│   ├── config 
│   │   ├── simple.py           <- [Template Config]
│   │   ├── gan.py
│   │   ├── ...
│   │
│   ├── losses
│   │   ├── simple.py           <- [Template Loss]
│   │   ├── perceptual.py
│   │   ├── ...
│   │
│   ├── metrics
│   │   ├── simple.py           <- [Template Metric]
│   │   ├── ...
│   │
│   ├── models
│   │   ├── simple.py           <- [Template Model]
│   │   ├── ...
│   │
│   ├── data
│   │   ├── simple.py           <- [Template CustomDataset]
│   │   ├── ...
│   │
│   ├── logger
│   │   ├── storage             <- [models' checkpoints, etc]
│   │   ├── wandb
│   │
│   ├── ...
```

## Classes design


### CustomDataset
```python
class SimpleDataset(Dataset):
    def __init__(remote, local):
        if not local: 
            self.download(remote, local)
        
    def preprocess():
        ...

    def download():
        ...

    def __getitem__():
        ...

    def __len__():
        ...
```

### Trainer
```python
class SimpleTrainer():
    def __init__():
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        self.dataloader = Dataloader(dataset)
    def run():
        # train steps
        # eval steps
```

### Config
```python
class Config():
    def __init__():
        self.model = ...
        self.criterion = ...
        self.optimizer = ...
        self.dataset = Dataset(local)
        self.trainer = SimpleTrainer()
    def run():
        self.trainer.run()
```

### Losses
```python
class Loss():
    def __init__():
        ...
    def forward():
        ...
```

### Metric
```python
class Metric():
    def __init__():
        ...
    def __call__():
        ...
```

### Model
```python
class Model():
    def __init__():
        ...
    def forward():
        ...
```


## Requirements
- PyTorch framework
- Python 3.9
- Type Annotations

## Future plans

### Tests
- CI tests 



