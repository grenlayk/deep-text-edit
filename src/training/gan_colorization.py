import time
import torch
from loguru import logger
from torchvision.utils import save_image
from torch.autograd import Variable


class GANColorizationTrainer:
    def __init__(self,
                 model_G,
                 model_D,
                 criterion,
                 criterion_gan,
                 lambda_L1,
                 optimizer_G,
                 optimizer_D,
                 scheduler_G,
                 train_dataloader,
                 val_dataloader,
                 test_dataloader,
                 total_epochs,
                 logger,
                 storage):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_G = model_G
        self.model_D = model_D
        self.criterion = criterion
        self.criterion_gan = criterion_gan, 
        self.lambda_L1 = lambda_L1
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.scheduler_G = scheduler_G
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.val_dataloader = val_dataloader
        self.total_epochs = total_epochs
        self.logger = logger
        self.storage = storage

    def set_requires_grad(self, net, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            net (network)         -- a  networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

    def train(self):
        self.model.train()

        # inputs: l_img x3
        # target: rgb_img
        for inputs, targets in self.train_dataloader:

            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer_G.zero_grad()
            preds_G = self.model_G(inputs)

            # update D
            self.set_requires_grad(self.model_D, True)
            self.optimizer_D.zero_grad()

            # Fake; stop backprop to the generator by detaching pred_G
            pred_D_fake = self.netD(preds_G.detach())
            fake = Variable(torch.Tensor(pred_D_fake.shape[0], pred_D_fake.shape[1]).fill_(0.0), \
                                                                requires_grad=False).to(self.device)
            loss_D_fake = self.criterion_gan(pred_D_fake, fake)
            # Real
            pred_D_real = self.netD(targets)
            valid = Variable(torch.Tensor(pred_D_real.shape[0], pred_D_real.shape[1]).fill_(1.0), \
                                                                requires_grad=False).to(self.device)
            loss_D_real = self.criterion_gan(pred_D_real, valid) 

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            self.optimizer_D.step() 

            # update G
            self.set_requires_grad(self.model_D, False)
            self.optimizer_G.zero_grad() 

            pred_D_fake = self.model_D(preds_G)
            loss_G_gan = self.criterion_gan(pred_D_fake, valid)
            loss_G_L1 = self.criterion(preds_G, targets)

            loss_G = loss_G_gan + loss_G_L1 * self.lambda_L1
            loss_G.backward()

            self.optimizer_G.step()
            self.scheduler_G.step()

            # Print stats after every point_batches
            self.logger.log_train(
                losses={'loss_G': loss_G.item(), 'loss_D': loss_D.item(), 
                'lr_G': self.scheduler_G.get_last_lr()[0]},
                images={'input': inputs, 'pred': preds_G, 'target': targets},
            )

    def validate(self, epoch: int):
        # Validation Step
        # Intialize Model to Eval Mode for validation
        self.model.eval()
        for inputs, targets in self.val_dataloader:
            # Skip bad data
            if not inputs.ndim:
                continue

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward Propagation
            preds_G = self.model_G(inputs)

            # Loss Calculation
            loss_G_L1 = self.criterion(preds_G, targets)

            self.logger.log_val(
                losses={'val_loss_G_L1': loss_G_L1.item()},
                images={'val_input': inputs, 'val_pred': preds_G, 'val_target': targets},
            )

        return self.logger.end_val()

    def run(self):
        for epoch in range(self.total_epochs):
            start = time.time_ns()
            self.train()
            logger.success(f'Finished training epoch #{epoch} in {(time.time_ns() - start) / 1e9}s')
            with torch.no_grad():
                _, _ = self.validate(epoch)

            # Save the Model to disk
            self.storage.save(
                epoch,
                {'model': self.model, 'optimizer': self.optimizer, 'scheduler': self.scheduler},
                None
            )
            logger.info('Model saved')
