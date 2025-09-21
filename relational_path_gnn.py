import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax


class RelationalPathGNN(nn.Module):
    def __init__(self, g, ent2id, num_rel, parameter):
        super(RelationalPathGNN, self).__init__()
        self.ent2id_dict = ent2id
        self.device = parameter['device']
        self.hop = parameter['hop']
        self.es = parameter['embed_dim']
        self.g_batch = parameter['g_batch']
        self.g = g
        self.g.to(self.device)
        self.sampler = dgl.dataloading.MultiLayerFullNeighborSampler(parameter['hop'], prefetch_node_feats=['feat'],    # 构造多层全邻居采样器，支持多跳路径建模（hop层）,预取节点特征 feat、边特征 feat 和 eid（关系ID）
                                                                     prefetch_edge_feats=['feat', 'eid'])
        self.gcn = RPGNN(self.es, self.es * 2, self.es, self.hop, num_rel)      # 初始化 GCN 编码器 RPGNN，输入维度 es，中间维度 2*es，输出维度 es
        self.num_rel = num_rel

    def ent2id(self, triples):  # 将三元组实体映射为 ID
        idx = [[[self.ent2id_dict[t[0]], self.ent2id_dict[t[2]]] for t in batch] for batch in triples]  # 输入是形如 [ [("h1", "r1", "t1"), ...], ... ] 的 batch 三元组
        idx = torch.LongTensor(idx).to(self.device)
        return idx  # B * few * 2   # 返回每对三元组的 (head_id, tail_id)，形状为 [B, few, 2]

    def forward(self, triples):     # 对输入的实体对进行 GCN 编码
        '''
        inputs:
            task: Batch triplets, B * few
        outputs:
            emb: B * few * es
        '''

        idx = self.ent2id(triples)
        batch_size, few_shot = idx.shape[0], idx.shape[1]
        idx = idx.view(-1)
        dataloader = dgl.dataloading.DataLoader(    # 为这些实体对创建 DGL 的子图 batch loader，用于多跳 GCN 编码
            self.g, idx, self.sampler,
            batch_size=self.g_batch,
            shuffle=False,
            drop_last=False,
            device=self.device,
            use_uva=True)
        out_emb = []
        for input_nodes, output_nodes, blocks in dataloader:    # 遍历每一批采样到的图块，调用 RPGNN 编码器进行图卷积
            input_features = blocks[0].srcdata['feat']
            out_features = self.gcn(blocks, input_features)
            out_emb.append(out_features)
        out_emb = torch.cat(out_emb, dim=0)
        out_emb = out_emb.view(batch_size, few_shot, 2, -1)
        return out_emb  # 输出形状 [B, few, 2, es]，表示每对实体路径对的 head / tail 编码

# 用于路径实体对的多层图卷积编码器
class RPGNN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, hop, n_rel):
        super().__init__()
        emb_dim = in_features
        self.conv_in = RPLayer(emb_dim, in_features, hidden_features, n_rel)
        self.conv_out = RPLayer(emb_dim, hidden_features, out_features, n_rel)
        self.hop = hop
        if hop > 2:
            self.conv_hidden = nn.ModuleList(
                [RPLayer(emb_dim, hidden_features, hidden_features, n_rel) for _ in range(hop - 2)])

    def forward(self, blocks, x):   # 通过多层关系感知图卷积提取路径信息,每层用不同的子图 block 表示不同的 hop
        x = F.relu(self.conv_in(blocks[0], x))
        if self.hop > 2:
            for i, conv in enumerate(self.conv_hidden):
                x = F.relu(conv(blocks[i + 1], x))
        x = F.relu(self.conv_out(blocks[-1], x))
        return x

# 基于关系感知的消息传递层
class RPLayer(nn.Module):
    def __init__(self, emb_dim, in_feat, out_feat, num_rels):
        super().__init__()
        self.num_rels = num_rels
        self.linear_r = dgl.nn.pytorch.TypedLinear(in_feat + emb_dim * 2, out_feat, num_rels)
        self.attn_fc = nn.Linear(emb_dim + out_feat, 1, bias=False)
        self.h_bias = nn.Parameter(torch.Tensor(out_feat))
        self.loop_weight = nn.Parameter(torch.Tensor(emb_dim, out_feat))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

    def edge_agg(self, edges):
        """Relation Message Passing
        edges 是 DGL 自动传入的边批处理对象，包含当前子图中所有的边
        每条边包含：

        edges.src['h']：边的源节点的输入特征（message来源）

        edges.dst['feat']：边的目标节点的原始嵌入（用于注意力）

        edges.data['feat']：边本身的特征（可能表示路径上的关系）

        edges.data['eid']：边的关系类型 ID，用于多关系建模
        """
        x = torch.cat([edges.src['h'], edges.data['feat'], edges.dst['feat']], dim=1)   # 这一操作将源节点表示 + 边的特征 + 目标节点原始嵌入拼接成一个大的特征向量 x，用于构造更丰富的消息
        m = self.linear_r(x, edges.data['eid'])                     # linear_r 是 DGL 的 TypedLinear，根据边的 eid（关系ID）选择不同的线性变换矩阵。
        # attn_fc 的输出是标量分数（未归一化），代表该边的重要性 𝛼
        attn = F.leaky_relu(self.attn_fc(torch.cat([edges.dst['feat'], m], dim=1)))     # 把目标节点特征 edges.dst['feat'] 和消息 m 拼接起来，送入一个前馈网络 attn_fc，输出一个注意力得分。
        return {'h': m, 'z': attn}

    def forward(self, g, feat):
        with g.local_scope():
            # Norm
            degs = g.out_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = torch.reshape(norm, shp)
            feat = feat * norm
            g.srcdata['h'] = feat
            g.apply_edges(self.edge_agg)
            e = g.edata.pop('z')
            a = edge_softmax(g, e)
            g.edata['h'] = a * g.edata['h']
            g.update_all(dgl.function.copy_e('h', 'm'), dgl.function.sum('m', 'h'))
            h = g.dstdata['h']
            h = h + g.dstdata['feat'] @ self.loop_weight
            # Norm 
            degs = g.in_degrees().float().clamp(min=1)
            norm = torch.pow(degs, -0.5)
            shp = norm.shape + (1,) * (h.dim() - 1)
            norm = torch.reshape(norm, shp)
            rst = h * norm
            h = rst + self.h_bias

            return h