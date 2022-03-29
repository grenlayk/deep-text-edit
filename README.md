# text-deep-fake design doc

This project aims to implement neural network architecture, described in [Text Style Brush Paper](https://arxiv.org/pdf/2106.08385.pdf).

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

### Config
```python
class Config():
    def __init__(): 
        model = Model()             # model from src/models
        criterion = Loss()          # loss func from from src/losses
        optimizer = ...             # some non-custom optimizer 
        storage = Storage()         # storage class func from from src/logger/storage
        logger = Logger()           # logger class func from from src/logger
        train_dataloader = DataLoader(SimpleDataset('path/to/train'))
        val_dataloader = DataLoader(SimpleDataset('path/to/val'))

        self.trainer = SimpleTrainer(<"pass all args from above">) 
    def run():
        self.trainer.run()
```

### Trainer
```python
class SimpleTrainer():
    def __init__():
        self.model = ...            # model from src/models
        self.criterion = ...        # loss func from from src/losses
        self.optimizer = ...         
        self.storage = storage      # storage class func from from src/logger/storage
        self.logger = logger        # logger class func from from src/logger
        self.train_dataloader = ... # dataloader based on custom dataset from src/data
        self.val_dataloader = ...   # dataloader based on custom dataset from src/data
    def run():
        for _ in range(max_epoch):
            self.train()            # train steps
            self.validate()         # validation steps
    def train():
        # train actions per epoch
    def validate():
        # validation actions per epoch
```

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

## File storage

We use Yandex.disk with 1TB storage to store dataset, logs and checkpoints.

## Requirements & restrictions
- PyTorch framework
- Python 3.9
- Type Annotations

## Future plans

### Tests
- CI tests 



