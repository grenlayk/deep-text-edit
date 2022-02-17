from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from loguru import logger
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.files = list(root_dir.iterdir())
        logger.info(f'File[0]: {self.files[0]}, Total Files: {len(self.files)}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            # *** Read the image from file ***
            rgb_img = cv2.imread(str(self.files[index]))

            if rgb_img is None:
                raise Exception

            rgb_img = rgb_img.astype(np.float32)
            rgb_img /= 255.0

            # *** Resize the color image to pass to encoder ***
            rgb_encoder_img = cv2.resize(rgb_img, (224, 224))

            # *** Resize the color image to pass to decoder ***
            rgb_resnet_img = cv2.resize(rgb_img, (300, 300))

            ''' Encoder Images '''
            # *** Convert the encoder color image to normalized lab space ***
            lab_encoder_img = cv2.cvtColor(rgb_encoder_img, cv2.COLOR_BGR2Lab)

            # *** Splitting the lab images into l-channel, a-channel, b-channel ***
            l_encoder_img = lab_encoder_img[:, :, 0]
            a_encoder_img = lab_encoder_img[:, :, 1]
            b_encoder_img = lab_encoder_img[:, :, 2]

            # *** Normalizing l-channel between [-1,1] ***
            l_encoder_img = l_encoder_img / 50.0 - 1.0

            # *** Repeat the l-channel to 3 dimensions ***
            l_encoder_img = torchvision.transforms.ToTensor()(l_encoder_img)
            l_encoder_img = l_encoder_img.expand(3, -1, -1)

            # *** Normalize a and b channels and concatenate ***
            a_encoder_img = (a_encoder_img / 128.0)
            b_encoder_img = (b_encoder_img / 128.0)
            a_encoder_img = torch.stack([torch.Tensor(a_encoder_img)])
            b_encoder_img = torch.stack([torch.Tensor(b_encoder_img)])
            ab_encoder_img = torch.cat([a_encoder_img, b_encoder_img], dim=0)

            # Convert the resnet color image to lab space
            lab_resnet_img = cv2.cvtColor(rgb_resnet_img, cv2.COLOR_BGR2Lab)

            # Extract the l-channel of resnet lab image
            l_resnet_img = lab_resnet_img[:, :, 0] / 50.0 - 1.0

            # Convert the resnet l-image to torch Tensor and stack it in 3 channels
            l_resnet_img = torchvision.transforms.ToTensor()(l_resnet_img)
            l_resnet_img = l_resnet_img.expand(3, -1, -1)

            # return images to data-loader
            rgb_encoder_img = torchvision.transforms.ToTensor()(rgb_encoder_img)
            return l_encoder_img, ab_encoder_img, l_resnet_img, rgb_encoder_img, str(self.files[index].name)

        except Exception as e:
            logger.error(f'Exception at {self.files[index]}, {e}')
            return torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), torch.tensor(-1), 'Error'
