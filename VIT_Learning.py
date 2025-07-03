import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torchsummary import summary

# 打开图片
img = Image.open('./OIP-C.jpg')
fig = plt.figure()
# plt.imshow(img)
# plt.show()

# resize to ImageNet size 
transform = Compose([Resize((224, 224)), ToTensor()])
x = transform(img)
# 确保 x 是 tensor
if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)
x = x.unsqueeze(0)  # 主要是为了添加batch这个维度
print(f"Final tensor shape: {x.shape}")

patch_size = 16  # 16 pixels
pathes = rearrange(x, 'b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size)
print(f"Patches shape: {pathes.shape}")

# 可视化 patch
fig = plt.figure(figsize=(10, 10))
for i in range(16):
    ax = fig.add_subplot(4, 4, i + 1)
    # 将patch重塑为图像格式 (16, 16, 3)
    patch_img = pathes[0, i].reshape(16, 16, 3)
    ax.imshow(patch_img)
    ax.axis('off')
plt.show()##1

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # 将原始图像切分为16*16的patch并把它们拉平
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            # 注意这里的隐层大小设置的也是768，可以配置
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )
                
    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x
    
PatchEmbedding()(x).shape