import torch
import torch.nn as nn
import torch.nn.functional as F
from relational_path_gnn import RelationalPathGNN


def bce_ranking_loss(p_score, n_score, y=None):
    """
    p_score: [B, 1, 1]
    n_score: [B, N, 1]
    """
    p_score = p_score.squeeze(-1)  # [B, 1]
    n_score = n_score.squeeze(-1)  # [B, N]

    pos_label = torch.ones_like(p_score)
    neg_label = torch.zeros_like(n_score)

    loss_pos = F.binary_cross_entropy_with_logits(p_score, pos_label)
    loss_neg = F.binary_cross_entropy_with_logits(n_score, neg_label)

    return (loss_pos + loss_neg) / 2

def bpr_hard_negative_loss(p_score, n_score, y=None):
    """
    Args:
        p_score: [B, K, 1] 正样本打分，K 是每个样本有多少个正例
        n_score: [B, K, 1] 每个正样本对应的最难负样本打分
    Returns:
        loss: scalar
    """
    # print(p_score.size(), n_score.size())  torch.Size([64, 3, 1]) torch.Size([64, 3, 1])
    assert p_score.shape == n_score.shape, f"Shape mismatch: p_score {p_score.shape}, n_score {n_score.shape}"

    p_score = p_score.squeeze(-1)
    n_score = n_score.squeeze(-1)

    hardest_n = n_score.max(dim=1).values
    hardest_p = p_score.max(dim=1).values

    # return -F.logsigmoid(hardest_p - hardest_n).mean()
    return (-F.logsigmoid(hardest_p - hardest_n)).mean()

class ScoreCalculator(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.head_encoder = nn.Linear(4 * emb_dim, emb_dim)
        self.tail_encoder = nn.Linear(4 * emb_dim, emb_dim)

    def forward(self, h, t, r, pos_num, z):
        """
        h, t, r : (B, nq+nn, 1, emb_dim)
        z       : (B, nq+nn, embed_dim)
        pos_num : 正样本数量
        return: p_score (B, pos_num), n_score (B, neg_num)
        """
        z_unsq = z.unsqueeze(2)


        h = h + self.head_encoder(z_unsq)
        t = t + self.tail_encoder(z_unsq)

        # L2 norm
        score = -torch.norm(h + r - t, p=2, dim=-1) 

        p_score = score[:, :pos_num] 
        n_score = score[:, pos_num:]
        return p_score, n_score

class ReIBEG(nn.Module):
    def __init__(self, g, dataset, parameter):
        super().__init__()

        self.device = parameter['device']
        self.dropout_p = parameter['dropout_p']
        self.embed_dim = parameter['embed_dim']
        self.margin = parameter['margin']
        self.few = parameter['few']

        self.r_path_gnn = RelationalPathGNN(
            g,
            dataset['ent2id'],
            len(dataset['rel2emb']),
            parameter
        )

        self.relation_learner = GatedCNN_Attn(embed_size=self.embed_dim,
                                              n_hidden=parameter['lstm_hiddendim'],
                                              out_size=self.embed_dim,
                                              layers=parameter['lstm_layers'])
        self.score_calculator = ScoreCalculator(self.embed_dim)

        self.attn_pooler = EMAttentionPooler(embed_dim=self.embed_dim*2, num_bases=8, num_iters=3)

        self.rib_encoder = InformationBottleneckEncoder(
            input_dim=self.embed_dim * 2,
            hidden_dim=self.embed_dim * 4,
            latent_dim=self.embed_dim * 2,
            cond_dim=self.embed_dim * 3
        )

        self.train_loss_func = bpr_hard_negative_loss
        self.eval_loss_func = bce_ranking_loss

    def split_concat(self, positive, negative):
        """
        positive, negative shape: (B, n, 2, embed_dim)
        -> (B, n+n, 1, embed_dim) for head & tail
        """
        pos_neg_e1 = torch.cat([positive[:, :, 0, :],
                                negative[:, :, 0, :]], dim=1).unsqueeze(2)
        pos_neg_e2 = torch.cat([positive[:, :, 1, :],
                                negative[:, :, 1, :]], dim=1).unsqueeze(2)
        return pos_neg_e1, pos_neg_e2

    def eval_reset(self):
        self.eval_query = None
        self.eval_target_r = None
        self.eval_rel = None
        self.is_reset = True

    def eval_support(self, support, support_negative, query):
        support, support_negative, query = self.r_path_gnn(support), self.r_path_gnn(support_negative), self.r_path_gnn(
            query)
        B = support.shape[0]
        support_few = support.view(support.shape[0], self.few, 2, self.embed_dim)
        rel = self.relation_learner(support_few)
        support_pos_r = support.view(B, self.few, -1)
        support_neg_r = support_negative.view(B, self.few, -1)
        target_r = torch.cat([support_pos_r, support_neg_r], dim=1)

        return query, target_r, rel

    def eval_forward(self, task):
        support, support_negative, query, negative = task
        negative = self.r_path_gnn(negative)

        if self.is_reset:
            query, target_r, rel = self.eval_support(support, support_negative, query)
            self.eval_query = query
            self.eval_target_r = target_r
            self.eval_rel = rel
            self.is_reset = False
        else:
            query = self.eval_query
            target_r = self.eval_target_r
            rel = self.eval_rel

        B = negative.shape[0]
        num_q = query.shape[1]  # num of query
        num_n = negative.shape[1]  # num of query negative

        # global_cond: (B, cond_dim)
        pooled_target = self.attn_pooler(target_r)  # (B, embed_dim * 2)
        rel_flat = rel.view(B, -1)  # (B, embed_dim)
        global_cond = torch.cat([rel_flat, pooled_target], dim=-1)  # → (B, embed_dim * 3)

        # 通过信息瓶颈编码器生成压缩表示
        z_sample, mu, logvar = self.rib_encoder(
            target_r, cond=global_cond, deterministic=True
        )

        z_without_pad = z_sample[:, :self.few * 2, :]  # (B, few*2, D)
        z_pos = z_without_pad[:, :self.few]  # (B, few, D)
        z_neg = z_without_pad[:, self.few:]  # (B, few, D)

        z_pos_r = self.attn_pooler(z_pos)  # (B, D)
        z_neg_r = self.attn_pooler(z_neg)  # (B, D)

        z = torch.cat([z_pos_r, z_neg_r], dim=-1)  # (B, 2*D)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)  # (B, nq+nn, 2*D)
        
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)

        p_score, n_score = self.score_calculator(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)

        return p_score, n_score

    def forward(self, task, iseval=False, istest=False):
        """
        task: (support, support_negative, query, negative)
        shape -> self.r_path_gnn -> (B, few, 2, embed_dim) or (B, nq, 2, embed_dim)
        """
        support, support_negative, query, negative = [self.r_path_gnn(t) for t in task]

        B = support.shape[0]
        num_q = query.shape[1]
        num_n = negative.shape[1]

        # relation  (support_few)
        support_few = support.view(B, self.few, 2, self.embed_dim)

        rel = self.relation_learner(support_few)  # (B, 1, 1, embed_dim) shape

        # support_pos_r, support_neg_r : (B, few, embed_dim * 2)
        support_pos_r = support.view(B, self.few, -1)
        support_neg_r = support_negative.view(B, self.few, -1)

        # pos_r -> (B, few, embed_dim * 2)
        pos_r = support_pos_r
        # neg_r -> (B, few, embed_dim * 2)
        neg_r = support_neg_r

        target_r = torch.cat([pos_r, neg_r], dim=1)  # (B, 2 * few, embed_dim * 2)

        # global_cond
        pooled_target = self.attn_pooler(target_r)  # (B, embed_dim*2)
        global_cond = torch.cat([rel.view(B, -1), pooled_target], dim=-1)   # global_cond.shape = [B, embed_dim + embed_dim*2] = [B, 3 * embed_dim]

        # padding
        target_r = F.pad(target_r, (0, 0, 0, 2), mode='constant', value=0.0)
        target_r = target_r.to(self.device)
        if istest or iseval:
            z_sample, mu, logvar = self.rib_encoder(target_r, cond=global_cond, deterministic=True)
        else:
            z_sample, mu, logvar = self.rib_encoder(target_r, cond=global_cond, deterministic=False)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        z_without_pad = z_sample[:, :self.few * 2, :]

        z_pos = z_without_pad[:, :self.few]  # (B, few, embed_dim * 2)
        z_neg = z_without_pad[:, self.few:]  # (B, few, embed_dim * 2)
        z_pos_r = self.attn_pooler(z_pos)  # (B, embed_dim * 2)
        z_neg_r = self.attn_pooler(z_neg)  # (B, embed_dim * 2)
        z = torch.cat([z_pos_r, z_neg_r], dim=-1)  # (B, embed_dim * 2 * 2)
        z_q = z.unsqueeze(1).expand(-1, num_q + num_n, -1)  # (B, num_q + num_n, embed_dim * 2 * 2)

        #  pos/neg triple (query/negative) -> head/tail concat
        #  => (B, nq+nn, 1, emb_dim)
        que_neg_e1, que_neg_e2 = self.split_concat(query, negative)

        # rel : (B, 1, 1, embed_dim) -> (B, nq+nn, 1, embed_dim)
        rel_q = rel.expand(-1, num_q + num_n, -1, -1)

        p_score, n_score = self.score_calculator(que_neg_e1, que_neg_e2, rel_q, num_q, z_q)

        if iseval or istest:
            return p_score, n_score
        else:
            return p_score, n_score, kl_loss

class InformationBottleneckEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, cond_dim=0):
        super().__init__()
        self.cond_dim = cond_dim
        total_input_dim = input_dim + cond_dim

        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, cond=None, deterministic=False):
        """
        x: [B, *, D]
        cond: [B, D']  [B, *, D']
        return: z, mu, logvar
        """
        B, *rest, D = x.shape  # e.g., B x N x D
        x_flat = x.view(-1, D)

        if cond is not None:
            if cond.dim() == 2:  # [B, D'] -> [B, 1, D'] -> [B, N, D']
                cond = cond.unsqueeze(1).expand(-1, int(x.numel() / D / B), -1)
            cond_flat = cond.reshape(-1, cond.shape[-1])
            x_flat = torch.cat([x_flat, cond_flat], dim=-1)  # [B*N, D + D']

        h = self.encoder(x_flat)
        mu = self.fc_mu(h).view(B, *rest, -1)
        logvar = self.fc_logvar(h).view(B, *rest, -1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)

        if deterministic:
            return mu, mu, logvar

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

class EMAttentionPooler(nn.Module):
    def __init__(self, embed_dim=100, num_bases=8, num_iters=3, residual_weight=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_bases = num_bases
        self.num_iters = num_iters
        self.residual_weight = residual_weight

        self.bases = nn.Parameter(torch.randn(num_bases, embed_dim))

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        B, L, D = x.shape
        K = self.num_bases

        # 初始 K 个聚类中心
        mu = self.bases.unsqueeze(0).expand(B, K, D)  # (B, K, D)

        for _ in range(self.num_iters):
            attn_logits = torch.einsum("bld,bkd->blk", x, mu)  # (B, L, K)
            attn = F.softmax(attn_logits, dim=1).transpose(1, 2)  # (B, K, L)

            norm = attn.sum(dim=2, keepdim=True) + 1e-6
            mu = torch.einsum("bkl,bld->bkd", attn, x) / norm  # (B, K, D)

        # 均值聚合
        pooled = mu.mean(dim=1)  # (B, D)
        residual = x[:, -1, :]
        # residual = x[:, 0, :]
        # residual = x.mean(dim=1)
        pooled = pooled + self.residual_weight * residual  # (B, D)
        out = self.mlp(pooled)  # (B, D)
        return out

class GatedCNN_Attn(nn.Module):
    def __init__(self, embed_size=100, n_hidden=200, out_size=100, layers=2, dropout=0.1, num_heads=4):
        super(GatedCNN_Attn, self).__init__()
        self.embed_size = embed_size
        self.hidden_dim = n_hidden
        self.out_size = out_size
        self.layers = layers
        self.dropout = dropout
        self.num_heads = num_heads

        self.input_dim = embed_size * 2

        self.convs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_proj = nn.ModuleList()

        in_dim = self.input_dim
        for _ in range(self.layers):
            self.convs.append(nn.Conv1d(in_dim, self.hidden_dim, kernel_size=3, padding=1))
            self.gates.append(nn.Conv1d(in_dim, self.hidden_dim, kernel_size=3, padding=1))
            self.norms.append(nn.LayerNorm(self.hidden_dim))

            if in_dim != self.hidden_dim:
                self.res_proj.append(nn.Conv1d(in_dim, self.hidden_dim, kernel_size=1))
            else:
                self.res_proj.append(None)
            in_dim = self.hidden_dim

        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=self.num_heads, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, self.out_size)

    def gated_conv(self, x):
        # x: [B, N, D] -> [B, D, N]
        x = x.transpose(1, 2)
        for conv, gate, norm, proj in zip(self.convs, self.gates, self.norms, self.res_proj):
            residual = x  # [B, D, N]
            a = conv(x)
            b = torch.sigmoid(gate(x))
            x = a * b

            if proj is not None:
                residual = proj(residual)
            x = x + residual

            x = x.transpose(1, 2)  # [B, N, D]
            x = norm(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x.transpose(1, 2)  # [B, D, N]
        return x.transpose(1, 2)  # [B, N, D]

    def forward(self, inputs):
        """
        inputs: [B, N, 2, embed_size]
        """
        B, N, _, D = inputs.shape
        x = inputs.contiguous().view(B, N, -1)  # [B, N, 2*D]

        x = self.gated_conv(x)  # [B, N, hidden_dim]

        attn_output, _ = self.attn(x, x, x)  # [B, N, hidden_dim]
        attn_output = F.dropout(attn_output, p=self.dropout, training=self.training)

        score = self.out(attn_output)  # [B, N, out_size]
        weights = F.softmax(score.mean(-1), dim=-1)  # [B, N]
        context = torch.sum(attn_output * weights.unsqueeze(-1), dim=1)  # [B, hidden_dim]

        out = self.out(context)  # [B, out_size]
        return out.view(B, 1, 1, self.out_size)


