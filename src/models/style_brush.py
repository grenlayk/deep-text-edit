from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

from src.models.embedders import ContentResnet, StyleResnet
from src.models.stylegan import StyleBased_Generator


class StypeBrush(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = ContentResnet(resnet34(ResNet34_Weights.IMAGENET1K_V1))
        self.style = StyleResnet(resnet34(ResNet34_Weights.IMAGENET1K_V1))

        self.generator = StyleBased_Generator(dim_latent=512)

    def forward(self, style, content):
        style_embeds = self.style(style)
        content_embeds = self.content(content)

        results = self.generator(content_embeds, style_embeds)
        return results
