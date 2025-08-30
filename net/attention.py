import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BCA(nn.Module):
    def __init__(self, vis: bool = False):
        super().__init__()
        self.vis = vis
        self.hidden_size = 512
        self.num_heads = 8
        self.attn_dropout_rate = 0.1

        self.head_dim = self.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        # image projections
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key   = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        # text projections
        self.query_text = nn.Linear(self.hidden_size, self.all_head_size)
        self.key_text   = nn.Linear(self.hidden_size, self.all_head_size)
        self.value_text = nn.Linear(self.hidden_size, self.all_head_size)

        # output heads
        self.out_img   = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_text  = nn.Linear(self.hidden_size, self.hidden_size)

        # dropouts
        self.attn_dropout_img = nn.Dropout(self.attn_dropout_rate)
        self.attn_dropout_txt = nn.Dropout(self.attn_dropout_rate)
        self.attn_dropout_it  = nn.Dropout(self.attn_dropout_rate)
        self.attn_dropout_ti  = nn.Dropout(self.attn_dropout_rate)

        self.proj_dropout_img  = nn.Dropout(self.attn_dropout_rate)
        self.proj_dropout_text = nn.Dropout(self.attn_dropout_rate)
        
        self.concat = nn.Linear(512*2, 512)

        self.softmax = nn.Softmax(dim=-1)
        
        self.attn_pooling = nn.MultiheadAttention(embed_dim=512, num_heads=8)

    # ===== helpers =====
    def _transpose_for_scores(self, x):
        # (B, L, C) -> (B, num_heads, L, head_dim)
        B, L, _ = x.size()
        x = x.view(B, L, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _apply_attn(self, q, k, v, dropout):
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        probs  = self.softmax(scores)
        probs  = dropout(probs)
        ctx    = torch.matmul(probs, v)                       # (B, heads, L_q, head_dim)
        ctx    = ctx.permute(0, 2, 1, 3).contiguous()         # (B, L_q, heads, head_dim)
        ctx    = ctx.view(ctx.size(0), ctx.size(1), -1)       # (B, L_q, C)
        return ctx

    # ===== forward =====
    def forward(self, hidden_states, text):
        # ---- project to QKV ----
        q_img = self._transpose_for_scores(self.query(hidden_states))
        k_img = self._transpose_for_scores(self.key(hidden_states))
        v_img = self._transpose_for_scores(self.value(hidden_states))

        q_txt = self._transpose_for_scores(self.query_text(text))
        k_txt = self._transpose_for_scores(self.key_text(text))
        v_txt = self._transpose_for_scores(self.value_text(text))

        # ---- four attention flows ----
        ctx_img  = self._apply_attn(q_img, k_img, v_img, self.attn_dropout_img)   # I→I
        ctx_txt  = self._apply_attn(q_txt, k_txt, v_txt, self.attn_dropout_txt)   # T→T
        ctx_it   = self._apply_attn(q_img, k_txt, v_txt, self.attn_dropout_it)    # I→T
        ctx_ti   = self._apply_attn(q_txt, k_img, v_img, self.attn_dropout_ti)    # T→I

        # ---- fuse & project ----
        out_img  = self.proj_dropout_img (self.out_img ((ctx_img + ctx_it)  / 2))
        out_text = self.proj_dropout_text(self.out_text((ctx_txt + ctx_ti) / 2))
        out = torch.cat([out_img, out_text], dim=2)
        out = self.concat(out)
        
        out = out.permute(1, 0, 2)
        
        out, _ = self.attn_pooling(out, out, out)
        
        out = out.permute(1, 0, 2)

        return out
