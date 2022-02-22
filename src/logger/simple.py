import time
from torch import Tensor
from loguru import logger

class Logger():
    
    def __init__(self, print_freq: int = 100, image_freq: int = 1000, val_size: int = None, tb_path: str = None):
        self.print_freq: int = print_freq
        self.image_freq: int = image_freq
        self.loss_buff: dict[str, list[float]] = {}
        self.tb_path: str = tb_path
        self.train_iter = 1
        self.val_iter = 1
        # сомнительно 
        self.val_size = val_size # in batches

    def log_train(self, losses: dict[str, float], images: dict[str, Tensor]):
        if self.train_iter == 1:
            self.start_time = time.time()
            logger.info(
                    'Training started'
                )

        for loss_name, loss_value in losses:
            self.loss_buff[loss_name] += [loss_value]
            self.loss_buff[loss_name + ' sum'] += loss_value
        
        if self.train_iter % self.print_freq == 0:
            self.end_time = time.time()
            logger.info(
                    f'Batch: {self.train_iter}',
                    f'Processing time for last {self.print_freq} batches: {self.end_time - self.end_time:.3f}s',
                    *[f'Average {loss_name} over last {self.print_freq} batches: {self.loss_buff[loss_name + " sum"] / self.print_freq}' \
                        for loss_name in losses]
                )
            self.start_time = self.end_time
            self.loss_buff.clear()
        
        if self.train_iter % self.image_freq == 0:
            #TensorBoard magic
            pass

        self.train_iter += 1

    def log_val(self, losses: dict[str, float], images: dict[str, Tensor], metrics: dict[str, float]):
        
        if self.val_iter == 1:
            self.metrics_buff = {}
            self.loss_buff.clear()
            logger.info(
                    'Validation started'
                )

        for loss_name, loss_value in losses:
            self.loss_buff[loss_name] += [loss_value]
            self.loss_buff[loss_name + ' sum'] += loss_value
            self.loss_buff[loss_name + ' sumlast'] += loss_value # сомнительное решение


        for metric_name, metric_value in metrics:
            self.metrics_buff[metric_name] += [metric_value]
            self.metrics_buff[metric_name + ' sum'] += metric_value
            self.metrics_buff[metric_name + ' sumlast'] += metric_value # сомнительное решение


        if self.val_iter % self.print_freq == 0:
            logger.info(
                    f'Batch: {self.val_iter}',
                    *[f'Average {loss_name} over last {self.print_freq} batches: {self.loss_buff[loss_name + " sumlast"] / self.print_freq}' \
                        for loss_name in metrics],
                    *[f'Average {metric_name} over last {self.print_freq} batches: {self.metrics_buff[metric_name + " sumlast"] / self.print_freq}' \
                        for metric_name in losses]
                )
            # следствие сомнительного решения
            for loss_name in losses:
                self.loss_buff[loss_name + ' sumlast'] = 0
            for metric_name in metrics:
                self.metrics_buff[metric_name + ' sumlast'] = 0

        if self.val_iter == self.val_size:
            logger.info(
                    *[f'{loss_name}: {self.loss_buff[loss_name + " sum"] / self.val_size}' \
                        for loss_name in losses]
                )

            logger.info(
                    *[f'{metric_name}: {self.loss_buff[metric_name + " sum"] / self.val_size}' \
                        for metric_name in metrics]
                )


        if self.val_iter % self.image_freq == 0:
            #TensorBoard magic
            pass
