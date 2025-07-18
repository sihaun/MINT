import torch
import torch.nn as nn
import torchvision.models as models
from transformers import DistilBertModel, GPT2LMHeadModel

# 개선된 작은 버전
# InvertibleBlock 및 SharedNetwork (더 크게)
class InvertibleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

    def forward(self, x, reverse=False):
        if not reverse:
            return self.act(self.norm(self.linear(x)))
        else:
            W = self.linear.weight
            b = self.linear.bias
            return torch.linalg.solve(W.T, (x - b).T).T

class SharedNetwork(nn.Module):
    def __init__(self, dim=2048, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList([InvertibleBlock(dim) for _ in range(depth)])

    def forward(self, x, reverse=False):
        for block in (reversed(self.blocks) if reverse else self.blocks):
            x = block(x, reverse=reverse)
        return x

# 간단한 ConvDecoder
class ConvDecoder(nn.Module):
    def __init__(self, dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),  # 32x32
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class Mint(nn.Module):
    def __init__(self):
        super().__init__()
        # self.image_encoder = SimpleImageEncoder(output_dim=512)
        self.image_encoder = models.resnet18(weights='DEFAULT')
        self.image_encoder.fc = nn.Linear(512, 512)
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.text_decoder = GPT2LMHeadModel.from_pretrained("distilgpt2")

        # projection to shared space (dim=2048)
        self.to_shared_img = nn.Linear(512, 2048)
        self.to_shared_txt = nn.Linear(768, 2048)

        self.shared = SharedNetwork(dim=2048, depth=6)

        self.to_gpt_embed = nn.Linear(2048, self.text_decoder.config.n_embd)
        self.image_decoder = ConvDecoder(dim=2048)

    def forward_image_to_text(self, image):
        img_feat = self.image_encoder(image)              # (B, 512)
        shared = self.shared(self.to_shared_img(img_feat), reverse=False)
        gpt_input = self.to_gpt_embed(shared).unsqueeze(1)
        return self.text_decoder(inputs_embeds=gpt_input)

    def forward_text_to_image(self, input_ids, attention_mask):
        txt_feat = self.text_encoder(input_ids, attention_mask).last_hidden_state[:, 0]
        shared = self.shared(self.to_shared_txt(txt_feat), reverse=True)
        return self.image_decoder(shared)