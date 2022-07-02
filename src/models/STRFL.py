# source: https://github.com/ku21fan/STR-Fewer-Labels
# 
# Copyright (c) 2021 Baek JeongHun

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttnLabelConverter:
    """Convert between text-label and text-index"""

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [SOS] (start-of-sentence token) and [EOS] (end-of-sentence token) for the attention decoder.
        list_special_token = [
            "[PAD]",
            "[UNK]",
            "[SOS]",
            "[EOS]",
            " ",
        ]  # [UNK] for unknown character, ' ' for space.
        list_character = list(character)
        self.character = list_special_token + list_character

        self.dict = {}
        for i, char in enumerate(self.character):
            # print(i, char)
            self.dict[char] = i

        print(f"# of tokens and characters: {len(self.character)}")

    def encode(self, word_string, batch_max_length=25):
        """convert word_list (string) into word_index.
        input:
            word_string: word labels of each image. [batch_size]
            batch_max_length: max length of word in the batch. Default: 25
        output:
            word_index : the input of attention decoder. [batch_size x (max_length+2)] +1 for [SOS] token and +1 for [EOS] token.
            word_length : the length of output of attention decoder, which count [EOS] token also. [batch_size]
        """
        word_length = [
            len(word) + 1 for word in word_string
        ]  # +1 for [EOS] at end of sentence.
        batch_max_length += 1

        # additional batch_max_length + 1 for [SOS] at first step.
        word_index = torch.LongTensor(len(word_string), batch_max_length + 1).fill_(
            self.dict["[PAD]"]
        )
        word_index[:, 0] = self.dict["[SOS]"]

        for i, word in enumerate(word_string):
            word = list(word)
            word.append("[EOS]")
            word_idx = [
                self.dict[char] if char in self.dict else self.dict["[UNK]"]
                for char in word
            ]
            word_index[i][1: 1 + len(word_idx)] = torch.LongTensor(
                word_idx
            )  # word_index[:, 0] = [SOS] token

        return (word_index.to(device), torch.IntTensor(word_length).to(device))

    def decode(self, word_index, word_length):
        """convert word_index into word_string"""
        word_string = []
        for idx, length in enumerate(word_length):
            word_idx = word_index[idx, :length]
            word = "".join([self.character[i] for i in word_idx])
            word_string.append(word)
        return word_string


class Options():
    def __init__(self):
        self.batch_max_length: int = 25
        self.img_h: int = 32
        self.img_w: int = 100
        self.character: str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        self.Aug: str = 'None'
        self.semi: str = 'None'
        self.model_name: str = 'TRBA'
        self.num_fiducial: int = 20
        self.input_channel: int = 3
        self.output_channel: int = 512
        self.hidden_size: int = 256
        self.Transformation: str = 'TPS'
        self.FeatureExtraction: str = 'ResNet'
        self.SequenceModeling: str = 'BiLSTM'
        self.Prediction: str = 'Attn'
        self.Converter: AttnLabelConverter = AttnLabelConverter(self.character)
        self.sos_token_index: int = -1
        self.eos_token_index: int = -1
        self.num_class: int = 0
        self.sos_token_index = self.Converter.dict["[SOS]"]
        self.eos_token_index = self.Converter.dict["[EOS]"]
        self.num_class = len(self.Converter.character)


class LocalizationNetwork(nn.Module):
    """Localization Network of RARE, which predicts C' (K x 2) from I (I_width x I_height)"""

    def __init__(self, F, I_channel_num):
        super().__init__()
        self.F = F
        self.I_channel_num = I_channel_num
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.I_channel_num,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 64 x I_height/2 x I_width/2
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 128 x I_height/4 x I_width/4
            nn.Conv2d(128, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # batch_size x 256 x I_height/8 x I_width/8
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),  # batch_size x 512
        )

        self.localization_fc1 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.localization_fc2 = nn.Linear(256, self.F * 2)

        # Init fc2 in LocalizationNetwork
        self.localization_fc2.weight.data.fill_(0)
        # see RARE paper Fig. 6 (a)
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = np.linspace(0.0, -1.0, num=int(F / 2))
        ctrl_pts_y_bottom = np.linspace(1.0, 0.0, num=int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        initial_bias = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        self.localization_fc2.bias.data = (
            torch.from_numpy(initial_bias).float().view(-1)
        )

    def forward(self, batch_I):
        """
        input:     batch_I : Batch Input Image [batch_size x I_channel_num x I_height x I_width]
        output:    batch_C_prime : Predicted coordinates of fiducial points for input batch [batch_size x F x 2]
        """
        batch_size = batch_I.size(0)
        features = self.conv(batch_I).view(batch_size, -1)
        batch_C_prime = self.localization_fc2(self.localization_fc1(features)).view(
            batch_size, self.F, 2
        )
        return batch_C_prime


class GridGenerator(nn.Module):
    """Grid Generator of RARE, which produces P_prime by multipling T with P"""

    def __init__(self, F, I_r_size):
        """Generate P_hat and inv_delta_C for later"""
        super().__init__()
        self.eps = 1e-6
        self.I_r_height, self.I_r_width = I_r_size
        self.F = F
        self.C = self._build_C(self.F)  # F x 2
        self.P = self._build_P(self.I_r_width, self.I_r_height)

        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            # for multi-gpu, you may need register buffer
            self.register_buffer(
                "inv_delta_C",
                torch.tensor(self._build_inv_delta_C(self.F, self.C)).float(),
            )  # F+3 x F+3
            self.register_buffer(
                "P_hat", torch.tensor(self._build_P_hat(self.F, self.C, self.P)).float()
            )  # n x F+3
        else:
            # for fine-tuning with different image width, you may use below instead of self.register_buffer
            self.inv_delta_C = (
                torch.tensor(self._build_inv_delta_C(self.F, self.C)).float().to(device)
            )  # F+3 x F+3
            self.P_hat = (
                torch.tensor(self._build_P_hat(self.F, self.C, self.P))
                .float()
                .to(device)
            )  # n x F+3

    def _build_C(self, F):
        """Return coordinates of fiducial points in I_r; C"""
        ctrl_pts_x = np.linspace(-1.0, 1.0, int(F / 2))
        ctrl_pts_y_top = -1 * np.ones(int(F / 2))
        ctrl_pts_y_bottom = np.ones(int(F / 2))
        ctrl_pts_top = np.stack([ctrl_pts_x, ctrl_pts_y_top], axis=1)
        ctrl_pts_bottom = np.stack([ctrl_pts_x, ctrl_pts_y_bottom], axis=1)
        C = np.concatenate([ctrl_pts_top, ctrl_pts_bottom], axis=0)
        return C  # F x 2

    def _build_inv_delta_C(self, F, C):
        """Return inv_delta_C which is needed to calculate T"""
        hat_C = np.zeros((F, F), dtype=float)  # F x F
        for i in range(0, F):
            for j in range(i, F):
                r = np.linalg.norm(C[i] - C[j])
                hat_C[i, j] = r
                hat_C[j, i] = r
        np.fill_diagonal(hat_C, 1)
        hat_C = (hat_C ** 2) * np.log(hat_C)
        # print(C.shape, hat_C.shape)
        delta_C = np.concatenate(  # F+3 x F+3
            [
                np.concatenate([np.ones((F, 1)), C, hat_C], axis=1),  # F x F+3
                np.concatenate([np.zeros((2, 3)), np.transpose(C)], axis=1),  # 2 x F+3
                np.concatenate([np.zeros((1, 3)), np.ones((1, F))], axis=1),  # 1 x F+3
            ],
            axis=0,
        )
        inv_delta_C = np.linalg.inv(delta_C)
        return inv_delta_C  # F+3 x F+3

    def _build_P(self, I_r_width, I_r_height):
        I_r_grid_x = (
            np.arange(-I_r_width, I_r_width, 2) + 1.0
        ) / I_r_width  # self.I_r_width
        I_r_grid_y = (
            np.arange(-I_r_height, I_r_height, 2) + 1.0
        ) / I_r_height  # self.I_r_height
        P = np.stack(  # self.I_r_width x self.I_r_height x 2
            np.meshgrid(I_r_grid_x, I_r_grid_y), axis=2
        )
        return P.reshape([-1, 2])  # n (= self.I_r_width x self.I_r_height) x 2

    def _build_P_hat(self, F, C, P):
        n = P.shape[0]  # n (= self.I_r_width x self.I_r_height)
        P_tile = np.tile(
            np.expand_dims(P, axis=1), (1, F, 1)
        )  # n x 2 -> n x 1 x 2 -> n x F x 2
        C_tile = np.expand_dims(C, axis=0)  # 1 x F x 2
        P_diff = P_tile - C_tile  # n x F x 2
        rbf_norm = np.linalg.norm(P_diff, ord=2, axis=2, keepdims=False)  # n x F
        rbf = np.multiply(np.square(rbf_norm), np.log(rbf_norm + self.eps))  # n x F
        P_hat = np.concatenate([np.ones((n, 1)), P, rbf], axis=1)
        return P_hat  # n x F+3

    def build_P_prime(self, batch_C_prime):
        """Generate Grid from batch_C_prime [batch_size x F x 2]"""
        batch_size = batch_C_prime.size(0)
        batch_inv_delta_C = self.inv_delta_C.repeat(batch_size, 1, 1)
        batch_P_hat = self.P_hat.repeat(batch_size, 1, 1)
        batch_C_prime_with_zeros = torch.cat(
            (batch_C_prime, torch.zeros(batch_size, 3, 2).float().to(device)), dim=1
        )  # batch_size x F+3 x 2
        batch_T = torch.bmm(
            batch_inv_delta_C, batch_C_prime_with_zeros
        )  # batch_size x F+3 x 2
        batch_P_prime = torch.bmm(batch_P_hat, batch_T)  # batch_size x n x 2
        return batch_P_prime  # batch_size x n x 2


class TPS_SpatialTransformerNetwork(nn.Module):
    """Rectification Network of RARE, namely TPS based STN"""

    def __init__(self, F, I_size, I_r_size, I_channel_num=1):
        """Based on RARE TPS
        input:
            batch_I: Batch Input Image [batch_size x I_channel_num x I_height x I_width]
            I_size : (height, width) of the input image I
            I_r_size : (height, width) of the rectified image I_r
            I_channel_num : the number of channels of the input image I
        output:
            batch_I_r: rectified image [batch_size x I_channel_num x I_r_height x I_r_width]
        """
        super().__init__()
        self.F = F
        self.I_size = I_size
        self.I_r_size = I_r_size  # = (I_r_height, I_r_width)
        self.I_channel_num = I_channel_num
        self.LocalizationNetwork = LocalizationNetwork(self.F, self.I_channel_num)
        self.GridGenerator = GridGenerator(self.F, self.I_r_size)

    def forward(self, batch_I):
        batch_C_prime = self.LocalizationNetwork(batch_I)  # batch_size x K x 2
        # batch_size x n (= I_r_width x I_r_height) x 2
        build_P_prime = self.GridGenerator.build_P_prime(batch_C_prime)
        build_P_prime_reshape = build_P_prime.reshape(
            [build_P_prime.size(0), self.I_r_size[0], self.I_r_size[1], 2]
        )

        if torch.__version__ > "1.2.0":
            batch_I_r = F.grid_sample(
                batch_I,
                build_P_prime_reshape,
                padding_mode="border",
                align_corners=True,
            )
        else:
            batch_I_r = F.grid_sample(
                batch_I, build_P_prime_reshape, padding_mode="border"
            )

        return batch_I_r


class ResNet_FeatureExtractor(nn.Module):
    """FeatureExtractor of FAN (http://openaccess.thecvf.com/content_ICCV_2017/papers/Cheng_Focusing_Attention_Towards_ICCV_2017_paper.pdf)"""

    def __init__(self, input_channel, output_channel=512):
        super(ResNet_FeatureExtractor, self).__init__()
        self.ConvNet = ResNet(input_channel, output_channel, BasicBlock, [1, 2, 5, 3])

    def forward(self, input):
        return self.ConvNet(input)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = self._conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = self._conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def _conv3x3(self, in_planes, out_planes, stride=1):
        "3x3 convolution with padding"
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, input_channel, output_channel, block, layers):
        super(ResNet, self).__init__()

        self.output_channel_block = [
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
            output_channel,
        ]

        self.inplanes = int(output_channel / 8)
        self.conv0_1 = nn.Conv2d(
            input_channel,
            int(output_channel / 16),
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_1 = nn.BatchNorm2d(int(output_channel / 16))
        self.conv0_2 = nn.Conv2d(
            int(output_channel / 16),
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn0_2 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, self.output_channel_block[0], layers[0])
        self.conv1 = nn.Conv2d(
            self.output_channel_block[0],
            self.output_channel_block[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.output_channel_block[0])

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(
            block, self.output_channel_block[1], layers[1], stride=1
        )
        self.conv2 = nn.Conv2d(
            self.output_channel_block[1],
            self.output_channel_block[1],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.output_channel_block[1])

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(
            block, self.output_channel_block[2], layers[2], stride=1
        )
        self.conv3 = nn.Conv2d(
            self.output_channel_block[2],
            self.output_channel_block[2],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(self.output_channel_block[2])

        self.layer4 = self._make_layer(
            block, self.output_channel_block[3], layers[3], stride=1
        )
        self.conv4_1 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=(2, 1),
            padding=(0, 1),
            bias=False,
        )
        self.bn4_1 = nn.BatchNorm2d(self.output_channel_block[3])
        self.conv4_2 = nn.Conv2d(
            self.output_channel_block[3],
            self.output_channel_block[3],
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn4_2 = nn.BatchNorm2d(self.output_channel_block[3])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0_1(x)
        x = self.bn0_1(x)
        x = self.relu(x)
        x = self.conv0_2(x)
        x = self.bn0_2(x)
        x = self.relu(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)

        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size], T = num_steps.
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(
            input
        )  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_char_embeddings=256):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(
            input_size, hidden_size, num_char_embeddings
        )
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.generator = nn.Linear(hidden_size, num_class)
        self.char_embeddings = nn.Embedding(num_class, num_char_embeddings)

    def forward(self, batch_H, text, is_train=True, batch_max_length=25):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [SOS] token. text[:, 0] = [SOS].
        output: probability distribution at each step [batch_size x num_steps x num_class]
        """
        batch_size = batch_H.size(0)
        num_steps = batch_max_length + 1  # +1 for [EOS] at end of sentence.

        output_hiddens = (
            torch.FloatTensor(batch_size, num_steps, self.hidden_size)
            .fill_(0)
            .to(device)
        )
        hidden = (
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
            torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        )

        if is_train:
            for i in range(num_steps):
                char_embeddings = self.char_embeddings(text[:, i])
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_embeddings : f(y_{t-1})
                hidden, alpha = self.attention_cell(hidden, batch_H, char_embeddings)
                output_hiddens[:, i, :] = hidden[
                    0
                ]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = text[0].expand(batch_size)  # should be fill with [SOS] token
            probs = (
                torch.FloatTensor(batch_size, num_steps, self.num_class)
                .fill_(0)
                .to(device)
            )

            for i in range(num_steps):
                char_embeddings = self.char_embeddings(targets)
                hidden, alpha = self.attention_cell(hidden, batch_H, char_embeddings)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_class


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(
            hidden_size, hidden_size
        )  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, prev_hidden, batch_H, char_embeddings):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(
            torch.tanh(batch_H_proj + prev_hidden_proj)
        )  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(
            1
        )  # batch_size x num_channel
        concat_context = torch.cat(
            [context, char_embeddings], 1
        )  # batch_size x (num_channel + num_embedding)
        cur_hidden = self.rnn(concat_context, prev_hidden)
        return cur_hidden, alpha


class TRBA(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial,
            I_size=(opt.img_h, opt.img_w),
            I_r_size=(opt.img_h, opt.img_w),
            I_channel_num=opt.input_channel,
        )

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(
            opt.input_channel, opt.output_channel
        )
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (img_h/16-1) -> 1

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(
                self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
            ),
            BidirectionalLSTM(
                opt.hidden_size, opt.hidden_size, opt.hidden_size
            ),
        )
        self.SequenceModeling_output = opt.hidden_size

        """Prediction"""
        self.Prediction = Attention(
            self.SequenceModeling_output, opt.hidden_size, opt.num_class
        )

    def forward(self, image, text=None, is_train=True):
        """Transformation stage"""
        image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(
            visual_feature
        )  # [b, num_steps, opt.hidden_size]

        """ Prediction stage """
        prediction = self.Prediction(
            contextual_feature.contiguous(),
            text,
            is_train,
            batch_max_length=self.opt.batch_max_length,
        )

        return prediction  # [b, num_steps, opt.num_class]
