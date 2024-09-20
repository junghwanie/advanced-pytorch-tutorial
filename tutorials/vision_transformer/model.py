import torch.nn as nn
import torch

from utils import *

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads: int=8):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        head_dim = int(emb_size / num_heads)
        self.keys = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(self.num_heads)])
        self.queries = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(self.num_heads)])
        self.values = nn.ModuleList([nn.Linear(head_dim, head_dim) for _ in range(self.num_heads)])
        self.head_dim = head_dim
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.num_heads):
                key = self.keys[head]
                query = self.queries[head]
                value = self.values[head]

                seq = sequence[:, head*self.head_dim:(head+1)*self.head_dim]
                q, k, v = query(seq), key(seq), value(seq)

                attention = self.softmax(q @ k.T / (self.num_heads ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])

# Non-hybrid architecture for transformer encoder
class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                hidden_dim, n_heads, forward_expansion: int=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mha = MultiHeadAttention(hidden_dim, n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, forward_expansion*hidden_dim),
            nn.GELU(),
            nn.Linear(forward_expansion*hidden_dim, hidden_dim),
        )
    def forward(self, x):
        out = x + self.mha(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out

# Model configuration with datasets
batch_size = 1
n_classes = 10

class ViT(nn.Module):
    def __init__(self, curr_dim,
                n_patches: int=8,
                n_blocks: int=6,
                hidden_dim=192,
                n_heads=8,
                n_classes: int=10,):
        super(ViT, self).__init__()

        self.curr_dim = curr_dim
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_classes = n_classes

        # Step 1. Linear projection; Nx(CxPxP)
        self.patch_size = (curr_dim[1]/n_patches, curr_dim[2]/n_patches)
        self.input_dim = int(3 * self.patch_size[0] * self.patch_size[1])
        self.linear_map = nn.Linear(batch_size*self.input_dim, self.hidden_dim)
        
        # Step 2. Classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_dim))

        # Step 3. Positional embedding
        self.positions = nn.Parameter(positional_embeddings(self.n_patches ** 2 + 1, self.hidden_dim).clone().detach())
        
        # Step 4. Vit block, classification head
        self.encoderBlock = nn.ModuleList([TransformerEncoderBlock(hidden_dim, n_heads) for _ in range(n_blocks)])
        self.classificationHead = nn.Sequential(
            nn.Linear(self.hidden_dim, n_classes),
            nn.Softmax(dim=-1),
        )

    def forward(self, images):
        n, _, _, _ = images.shape
        patches = patchify(images, self.n_patches)

        tokens = self.linear_map(patches)
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        out = tokens + self.positions.repeat(n, 1, 1)

        for block in self.encoderBlock:
            out = block(out)
        out = out[:, 0]

        return self.classificationHead(out)