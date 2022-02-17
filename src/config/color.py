import datetime
from pathlib import Path

import torch
from src.data.color import CustomDataset
from src.models.color import Colorization
from src.training.color import ColorizationTrainer


class Config:
    def __init__(self):
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        total_epochs = 20
        model = Colorization(256)
        criterion = torch.nn.MSELoss(reduction='mean').to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(range(0, total_epochs, 5)),
            gamma=0.2
        )

        batch_size = 10

        train_dataset = CustomDataset(Path('data/raw/Coco/train'))
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )

        val_dataset = CustomDataset(Path('data/raw/Coco/val'))
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8
        )

        test_dataset = CustomDataset(Path('data/raw/Coco/test'))
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

        point_batches = 500

        model_save_path = Path('models') / datetime.datetime.utcnow().strftime('%Y-%m-%d %H-%M-%S')
        model_save_path.mkdir(parents=True)

        self.trainer = ColorizationTrainer(
            model,
            criterion,
            optimizer,
            scheduler,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            total_epochs,
            batch_size,
            point_batches
        )

    def run(self):
        self.trainer.run()
