from math import sqrt

import torch
from torch.functional import F
from torch import nn
from transformers import AutoConfig, AutoTokenizer


def scaled_dot_product_attention(query, key, value):
    dim_k = query.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, value)


class AttentionHead(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        return scaled_dot_product_attention(self.q(hidden_state), self.k(hidden_state), self.v(hidden_state))


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads
        self.heads = nn.ModuleList(
            [AttentionHead(embed_dim=embed_dim, head_dim=head_dim) for _ in range(num_heads)]
        )
        self.output_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        x = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        x = self.output_linear(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear_2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return x


model_ckpt = "bert-base-uncased"
text = "time files like an arrow"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
config = AutoConfig.from_pretrained(model_ckpt, output_attiontoins=True)
multi_head_attention = MultiHeadAttention(config)

inputs = tokenizer(text, return_tensors='pt', add_special_tokens=False)
token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

input_emb = token_emb(inputs.input_ids)
print(input_emb.size())

attn_output = multi_head_attention(input_emb)
print(attn_output.size())
