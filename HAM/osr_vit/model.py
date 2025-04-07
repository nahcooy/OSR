# vit_training/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class PositionEmbs(nn.Module):
    def __init__(self, num_patches, emb_dim, dropout_rate=0.1):
        super(PositionEmbs, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        out = x + self.pos_embedding
        if self.dropout:
            out = self.dropout(out)
        return out

class MlpBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.dropout2 = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)
        out = self.fc2(out)
        if self.dropout2:
            out = self.dropout2(out)
        return out

class LinearGeneral(nn.Module):
    def __init__(self, in_dim=(768,), feat_dim=(12, 64)):
        super(LinearGeneral, self).__init__()
        self.weight = nn.Parameter(torch.randn(*in_dim, *feat_dim))
        self.bias = nn.Parameter(torch.zeros(*feat_dim))

    def forward(self, x, dims):
        return torch.tensordot(x, self.weight, dims=dims) + self.bias

class SelfAttention(nn.Module):
    def __init__(self, in_dim, heads=8, dropout_rate=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = in_dim // heads
        self.scale = self.head_dim ** 0.5
        self.query = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.key = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.value = LinearGeneral((in_dim,), (self.heads, self.head_dim))
        self.out = LinearGeneral((self.heads, self.head_dim), (in_dim,))
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x):
        b, n, _ = x.shape
        q = self.query(x, dims=([2], [0])).permute(0, 2, 1, 3)
        k = self.key(x, dims=([2], [0])).permute(0, 2, 1, 3)
        v = self.value(x, dims=([2], [0])).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, v).permute(0, 2, 1, 3)
        out = self.out(out, dims=([2, 3], [0, 1]))
        return out

class EncoderBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, num_heads, dropout_rate=0.1, attn_dropout_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = SelfAttention(in_dim, heads=num_heads, dropout_rate=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
        self.norm2 = nn.LayerNorm(in_dim)
        self.mlp = MlpBlock(in_dim, mlp_dim, in_dim, dropout_rate)

    def forward(self, x):
        residual = x
        out = self.norm1(x)
        out = self.attn(out)
        if self.dropout:
            out = self.dropout(out)
        out += residual
        residual = out
        out = self.norm2(out)
        out = self.mlp(out)
        out += residual
        return out

class Encoder(nn.Module):
    def __init__(self, num_patches, emb_dim, mlp_dim, num_layers=12, num_heads=12, dropout_rate=0.1, attn_dropout_rate=0.0):
        super(Encoder, self).__init__()
        self.pos_embedding = PositionEmbs(num_patches, emb_dim, dropout_rate)
        self.encoder_layers = nn.ModuleList([EncoderBlock(emb_dim, mlp_dim, num_heads, dropout_rate, attn_dropout_rate) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, x):
        out = self.pos_embedding(x)
        for layer in self.encoder_layers:
            out = layer(out)
        out = self.norm(out)
        return out

class VisionTransformer(nn.Module):
    def __init__(self, image_size=(256, 256), patch_size=(16, 16), emb_dim=768, mlp_dim=3072, num_heads=12, num_layers=12, num_classes=1000, attn_dropout_rate=0.0, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw
        num_patches = gh * gw
        self.embedding = nn.Conv2d(3, emb_dim, kernel_size=(fh, fw), stride=(fh, fw))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.transformer = Encoder(num_patches=num_patches, emb_dim=emb_dim, mlp_dim=mlp_dim, num_layers=num_layers, num_heads=num_heads, dropout_rate=dropout_rate, attn_dropout_rate=attn_dropout_rate)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, feat_cls=False):
        emb = self.embedding(x).permute(0, 2, 3, 1).reshape(x.size(0), -1, self.embedding.out_channels)
        cls_token = self.cls_token.repeat(x.size(0), 1, 1)
        emb = torch.cat([cls_token, emb], dim=1)
        feat = self.transformer(emb)
        if feat_cls:
            return feat[:, 0], self.classifier(feat[:, 0])
        return self.classifier(feat[:, 0])

def load_checkpoint(path, new_img=384, patch=16, emb_dim=768, layers=12):
    if path.endswith('npz'):
        keys, values = load_jax(path)
        state_dict = convert_jax_pytorch(keys, values)
    elif path.endswith('pth'):
        if 'deit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))['model']
        elif 'jx' in path or 'vit' in os.path.basename(path):
            state_dict = torch.load(path, map_location=torch.device("cpu"))
        else:
            state_dict = torch.load(path, map_location=torch.device("cpu"))['state_dict']
    else:
        raise ValueError(f"Checkpoint format {path.split('.')[-1]} not supported!")
    
    if 'jx' in path or any(x in os.path.basename(path) for x in ['vit', 'deit']):
        if 'distilled' in path:
            state_dict['distilled_token'] = state_dict.pop('dist_token')
        state_dict['transformer.pos_embedding.pos_embedding'] = state_dict.pop('pos_embed')
        state_dict['embedding.weight'] = state_dict.pop('patch_embed.proj.weight')
        state_dict['embedding.bias'] = state_dict.pop('patch_embed.proj.bias')
        if os.path.basename(path) == 'vit_small_p16_224-15ec54c9.pth':
            state_dict['embedding.weight'] = state_dict['embedding.weight'].reshape(768, 3, 16, 16)
        state_dict['classifier.weight'] = state_dict.pop('head.weight')
        state_dict['classifier.bias'] = state_dict.pop('head.bias')
        state_dict['transformer.norm.weight'] = state_dict.pop('norm.weight')
        state_dict['transformer.norm.bias'] = state_dict.pop('norm.bias')
        posemb = state_dict['transformer.pos_embedding.pos_embedding']
        for i, block_name in enumerate(list(state_dict.keys()).copy()):
            if 'blocks' in block_name:
                new_block = "transformer.encoder_layers." + block_name.split('.', 1)[1]
                state_dict[new_block] = state_dict.pop(block_name)
    else:
        posemb = state_dict['transformer.pos_embedding.pos_embedding']
    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
    model_grid_seq = new_img // patch
    ckpt_grid_seq = int(np.sqrt(posemb_grid.shape[0]))
    if model_grid_seq != ckpt_grid_seq:
        posemb_grid = posemb_grid.reshape(ckpt_grid_seq, ckpt_grid_seq, -1)
        posemb_grid = torch.unsqueeze(posemb_grid.permute(2, 0, 1), dim=0)
        posemb_grid = torch.nn.functional.interpolate(posemb_grid, size=(model_grid_seq, model_grid_seq), mode='bicubic', align_corners=False)
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
        state_dict['transformer.pos_embedding.pos_embedding'] = posemb
        print(f'Resized positional embedding from ({ckpt_grid_seq},{ckpt_grid_seq}) to ({model_grid_seq},{model_grid_seq})')
    return state_dict

def load_jax(path):
    ckpt_dict = np.load(path, allow_pickle=False)
    keys, values = zip(*list(ckpt_dict.items()))
    return keys, values

def convert_jax_pytorch(keys, values):
    state_dict = {}
    for key, value in zip(keys, values):
        names = key.split('/')
        torch_names = replace_names(names)
        torch_key = '.'.join(w for w in torch_names)
        tensor_value = torch.tensor(value, dtype=torch.float)
        num_dim = len(tensor_value.shape)
        if num_dim == 1:
            tensor_value = tensor_value.squeeze()
        elif num_dim == 2 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.T
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] in ['query', 'key', 'value']:
            tensor_value = tensor_value
        elif num_dim == 2 and torch_names[-1] == 'bias' and torch_names[-2] in ['query', 'key', 'value']:
            tensor_value = tensor_value
        elif num_dim == 3 and torch_names[-1] == 'weight' and torch_names[-2] == 'out':
            tensor_value = tensor_value
        elif num_dim == 4 and torch_names[-1] == 'weight':
            tensor_value = tensor_value.permute(3, 2, 0, 1)
        state_dict[torch_key] = tensor_value
    return state_dict

def replace_names(names):
    new_names = []
    for name in names:
        if name == 'Transformer': new_names.append('transformer')
        elif name == 'encoder_norm': new_names.append('norm')
        elif 'encoderblock' in name:
            num = name.split('_')[-1]
            new_names.extend(['encoder_layers', num])
        elif 'LayerNorm' in name:
            num = name.split('_')[-1]
            new_names.append(f'norm{1 if num == "0" else 2}')
        elif 'MlpBlock' in name: new_names.append('mlp')
        elif 'Dense' in name:
            num = name.split('_')[-1]
            new_names.append(f'fc{int(num) + 1}')
        elif 'MultiHeadDotProductAttention' in name: new_names.append('attn')
        elif name in ['kernel', 'scale']: new_names.append('weight')
        elif name == 'bias': new_names.append(name)
        elif name == 'posembed_input': new_names.append('pos_embedding')
        elif name == 'pos_embedding': new_names.append('pos_embedding')
        elif name == 'embedding': new_names.append('embedding')
        elif name == 'head': new_names.append('classifier')
        elif name == 'cls': new_names.append('cls_token')
        else: new_names.append(name)
    return new_names

class OODTransformer(nn.Module):
    def __init__(self, image_size, patch_size, emb_dim, mlp_dim, num_heads, num_layers, num_classes, attn_dropout_rate, dropout_rate):
        super(OODTransformer, self).__init__()
        # 기존 VisionTransformer와 동일한 구조 가정
        self.vit = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            emb_dim=emb_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_classes=emb_dim,  # 임베딩 출력으로 변경
            attn_dropout_rate=attn_dropout_rate,
            dropout_rate=dropout_rate
        )
        # 마지막 분류 레이어 제거 또는 임베딩 출력으로 수정
        self.classifier = nn.Identity()  # 임베딩 그대로 출력

    def forward(self, x):
        return self.vit(x)  # 임베딩 벡터 반환