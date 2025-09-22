import logging
import os
import shutil
import sys
import gc
import dgl
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm
import json
import numpy as np
from model import *
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from torch.cuda.amp import autocast, GradScaler
from typing import Union
import math


class Trainer:
    def __init__(self, data_loaders, dataset, parameter):
        self.parameter = parameter

        self.dataset_name = parameter['dataset']
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/{self.dataset_name}_trainlog_{timestamp}.txt"
        os.makedirs('logs', exist_ok=True)
        with open(self.log_path, 'w') as f:
            f.write(f"# Training Log for dataset: {self.dataset_name}\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"{'Epoch':>6} | {'Total Loss':>12} | {'Ranking Loss':>14} | {'KL Loss':>10} | {'KL Weight':>10}\n")
            f.write("-" * 65 + "\n")

        # data loader
        self.train_data_loader = data_loaders[0]
        self.dev_data_loader = data_loaders[1]
        self.test_data_loader = data_loaders[2]
        # parameters
        self.few = parameter['few']
        self.num_query = parameter['num_query']
        self.batch_size = parameter['batch_size']
        self.eval_batch_size = parameter['eval_batch_size']
        self.learning_rate = parameter['learning_rate']
        self.early_stopping_patience = parameter['early_stopping_patience']
        self.power = parameter['power']
        # epoch
        self.epoch = parameter['epoch']
        self.print_epoch = parameter['print_epoch']
        self.eval_epoch = parameter['eval_epoch']
        self.checkpoint_epoch = parameter['checkpoint_epoch']
        # device
        self.device = parameter['device']

        self.data_path = parameter['data_path']
        self.embed_model = parameter['embed_model']

        self.scaler = GradScaler()  # ---------------------------------------------------------

        self.load_embed()
        self.num_symbols = len(self.symbol2id.keys()) - 1
        self.pad_id = self.num_symbols
        self.ent2id = json.load(open(self.data_path + '/ent2ids'))
        self.rel2id = json.load(open(self.data_path + '/relation2ids'))
        self.num_ents = len(self.ent2id.keys())
        kg = self.build_kg(dataset['ent2emb'], dataset['rel2emb'])

        self.reibeg = ReIBEG(kg, dataset, parameter)
        self.reibeg.to(self.device)
        self.ema = EMAModel(parameters=self.reibeg.parameters(), power=self.power)
        # optimizer
        self.optimizer = torch.optim.Adam(
            self.reibeg.parameters(),
            self.learning_rate,
            weight_decay=parameter['weight_decay']  #  L2
        )  # optimizer

        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=parameter['epoch']
        )

        # tensorboard log writer
        if parameter['step'] == 'train':
            self.writer = SummaryWriter(os.path.join(parameter['log_dir'], parameter['prefix']))
        # dir
        self.state_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'])
        if not os.path.isdir(self.state_dir):
            os.makedirs(self.state_dir)
        self.ckpt_dir = os.path.join(self.parameter['state_dir'], self.parameter['prefix'], 'checkpoint')
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.state_dict_file = ''

        # logging
        logging_dir = os.path.join(
            self.parameter['log_dir'], self.parameter['prefix'])
        if not os.path.exists(logging_dir):
            os.makedirs(logging_dir)
        logging.basicConfig(filename=os.path.join(logging_dir, "res.log"),
                            level=logging.INFO, format="%(asctime)s - %(message)s", force=True)
        logging.info('*' * 100)
        logging.info('*** hyper-parameters ***')
        for k, v in parameter.items():
            logging.info(k + ': ' + str(v))
        logging.info('*' * 100)
        # load state_dict_fb15k_best_658 and params
        if parameter['step'] in ['test', 'dev']:
            self.reload()

    def load_symbol2id(self):
        symbol_id = {}
        rel2id = json.load(open(self.data_path + '/relation2ids'))
        ent2id = json.load(open(self.data_path + '/ent2ids'))
        i = 0
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        symbol_id = {}
        symbol_idinv = {}
        rel2id = json.load(open(self.data_path + '/relation2ids'))
        ent2id = json.load(open(self.data_path + '/ent2ids'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.data_path + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.data_path + '/relation2vec.' + self.embed_model)

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            print(rel_embed.shape[0])
            print(len(rel2id.keys()))
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    symbol_idinv[i] = key
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    symbol_idinv[i] = key
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings
            # print(symbol_idinv)
            # exit(-1)

    # def build_kg(self, ent_emb, rel_emb):
    #     print("Build KG...")
    #     src = []
    #     dst = []
    #     e_feat = []
    #     e_id = []
    #     with open(self.data_path + '/path_graph') as f:
    #         lines = f.readlines()
    #         for line in tqdm(lines):
    #             e1, rel, e2 = line.rstrip().split()
    #             src.append(self.ent2id[e1])
    #             dst.append(self.ent2id[e2])
    #             e_feat.append(rel_emb[self.rel2id[rel]])
    #             e_id.append(self.rel2id[rel])
    #             # Reverse
    #             src.append(self.ent2id[e2])
    #             dst.append(self.ent2id[e1])
    #             e_feat.append(rel_emb[self.rel2id[rel + '_inv']])
    #             e_id.append(self.rel2id[rel + '_inv'])
    #
    #     src = torch.LongTensor(src)
    #     dst = torch.LongTensor(dst)
    #     kg = dgl.graph((src, dst))
    #     kg.ndata['feat'] = torch.FloatTensor(ent_emb)
    #     kg.edata['feat'] = torch.FloatTensor(np.array(e_feat))
    #     kg.edata['eid'] = torch.LongTensor(np.array(e_id))
    #     return kg

    def build_kg(self, ent_emb, rel_emb):
        """
        构建图 (DGL graph)
        - 正向边使用原始关系 embedding
        - 反向边，如果存在 rel+'_inv' 就用对应 embedding，否则用正向 embedding 作为兼容处理
        """
        print("Build KG...")
        src = []
        dst = []
        e_feat = []
        e_id = []

        with open(self.data_path + '/path_graph') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2 = line.rstrip().split()

                # 正向边
                src.append(self.ent2id[e1])
                dst.append(self.ent2id[e2])
                e_feat.append(rel_emb[self.rel2id[rel]])
                e_id.append(self.rel2id[rel])

                # 反向边
                src.append(self.ent2id[e2])
                dst.append(self.ent2id[e1])
                inv_rel = rel + '_inv'
                if inv_rel in self.rel2id:
                    e_feat.append(rel_emb[self.rel2id[inv_rel]])
                    e_id.append(self.rel2id[inv_rel])
                else:
                    # 兼容处理：用正向 embedding 替代
                    e_feat.append(rel_emb[self.rel2id[rel]])
                    e_id.append(self.rel2id[rel])

        # 构建 DGL 图
        src = torch.LongTensor(src)
        dst = torch.LongTensor(dst)
        kg = dgl.graph((src, dst))
        kg.ndata['feat'] = torch.FloatTensor(ent_emb)
        kg.edata['feat'] = torch.FloatTensor(np.array(e_feat))
        kg.edata['eid'] = torch.LongTensor(np.array(e_id))

        return kg

    def reload(self):  
        if self.parameter['eval_ckpt'] is not None:     
            state_dict_file = os.path.join(self.ckpt_dir, 'state_dict_' + self.parameter['eval_ckpt'] + '.ckpt')
        else:
            state_dict_file = os.path.join(self.state_dir, self.parameter['state_dict_filename'])# 'state_dict_fb15k_best_658'
        self.state_dict_file = state_dict_file
        logging.info('Reload state_dict_fb15k_best_658 from {}'.format(state_dict_file))
        print('reload state_dict_fb15k_best_658 from {}'.format(state_dict_file))
        state = torch.load(state_dict_file, map_location=self.device)
        if os.path.isfile(state_dict_file):
            self.reibeg.load_state_dict(state)
        else:
            raise RuntimeError('No state dict in {}!'.format(state_dict_file))

    def save_checkpoint(self, epoch):
        torch.save(self.reibeg.state_dict(), os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt'))

    def del_checkpoint(self, epoch):
        path = os.path.join(self.ckpt_dir, 'state_dict_' + str(epoch) + '.ckpt')
        if os.path.exists(path):
            os.remove(path)
        else:
            raise RuntimeError('No such checkpoint to delete: {}'.format(path))

    def save_best_state_dict(self, best_epoch):
        shutil.copy(os.path.join(self.ckpt_dir, 'state_dict_' + str(best_epoch) + '.ckpt'),
                    os.path.join(self.state_dir, 'state_dict_fb15k_best_658'))

    def write_training_log(self, data, epoch):
        self.writer.add_scalar('Training_Loss', data['Loss'], epoch)
        self.writer.add_scalar('KL_Loss', data['KL'], epoch)

    def write_validating_log(self, data, epoch):
        self.writer.add_scalar('Validating_MRR', data['MRR'], epoch)
        self.writer.add_scalar('Validating_Hits_10', data['Hits@10'], epoch)
        self.writer.add_scalar('Validating_Hits_5', data['Hits@5'], epoch)
        self.writer.add_scalar('Validating_Hits_1', data['Hits@1'], epoch)

    def logging_training_data(self, data, epoch):
        logging.info("Epoch: {}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            epoch, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def logging_eval_data(self, data, state_path, istest=False):
        setname = 'dev set'
        if istest:
            setname = 'test set'
        logging.info("Eval {} on {}".format(state_path, setname))
        logging.info("MRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

    def rank_predict(self, data, x, ranks):
        # query_idx is the idx of positive score
        query_idx = x.shape[0] - 1
        # sort all scores with descending, because more plausible triple has higher score
        _, idx = torch.sort(x, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        ranks.append(rank)
        # update data
        if rank <= 10:
            data['Hits@10'] += 1
        if rank <= 5:
            data['Hits@5'] += 1
        if rank == 1:
            data['Hits@1'] += 1
        data['MRR'] += 1.0 / rank

    # cosine 增长，避免后期太大
    def get_kl_weight(self, epoch, warmup_epoch=1000):
        return 0.5 * (1 - math.cos(min(epoch / warmup_epoch, 1.0) * math.pi))

    def do_one_step(self, epoch, task, iseval=False, curr_rel='', istest=False, batch_eval=False):
        if not iseval:
            with torch.cuda.amp.autocast():
                p_score, n_score, kl_loss = self.reibeg(task, iseval, istest)
                y = torch.ones_like(p_score).to(self.device)
                ranking_loss = self.reibeg.train_loss_func(p_score, n_score, y)
                kl_weight = self.get_kl_weight(self.curr_epoch, warmup_epoch=1000)
                loss = ranking_loss + kl_weight * kl_loss

            return loss, kl_loss, p_score, n_score, ranking_loss, kl_weight

        elif curr_rel != '':
            with torch.no_grad():
                if batch_eval:
                    p_score, n_score = self.reibeg.eval_forward(task)
                else:
                    p_score, n_score = self.reibeg(task, iseval, istest)
                y = torch.ones_like(p_score).to(self.device)
                loss = self.reibeg.eval_loss_func(p_score, n_score, y)
            return loss, p_score, n_score

    def train(self):
        # initialization
        best_epoch = 0
        best_value = 0
        bad_counts = 0

        # training by epoch
        for e in range(self.epoch):
            self.curr_epoch = e

            # sample one batch from data_loader
            self.reibeg.train()
            train_task, curr_rel = self.train_data_loader.next_batch()
            with torch.cuda.amp.autocast(): 
                loss, kl_loss, _, _, ranking_loss, kl_weight = self.do_one_step(epoch=e, task=train_task, iseval=False, curr_rel=curr_rel, istest=False)

            with open(self.log_path, 'a') as f:
                f.write(f"{e:6} | {loss:12.4f} | {kl_loss:14.4f} | {ranking_loss:10.4f} | {kl_weight:10.4f}\n")

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.reibeg.parameters(), max_norm=5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()
            self.ema.step(self.reibeg.parameters())

            # print the loss on specific epoch
            if e % self.print_epoch == 0:
                loss_num = loss.item()
                self.write_training_log({'Loss': loss_num, 'KL': kl_loss.item()}, e)
                for param_group in self.optimizer.param_groups:
                    print("Epoch: {}\tLoss: {:.4f}\tLR: {:.6f}".format(e, loss_num, param_group['lr']))
            # save checkpoint on specific epoch
            if e % self.checkpoint_epoch == 0 and e != 0:

                print('Epoch  {} has finished, saving...'.format(e))
                self.save_checkpoint(e)
            # do evaluation on specific epoch
            if e % self.eval_epoch == 0 and e != 0:
                print('Epoch  {} has finished, validating...'.format(e))

                valid_data = self.eval(istest=False, epoch=e)
                self.write_validating_log(valid_data, e)
                metric = self.parameter['metric']

                # early stopping checking
                if valid_data[metric] > best_value:
                    best_value = valid_data[metric]
                    best_epoch = e
                    print('\tBest model | {0} of valid set is {1:.3f}'.format(metric, best_value))
                    bad_counts = 0
                    # save current best
                    self.save_checkpoint(best_epoch)
                    test_data = self.eval(istest=True)
                else:
                    print('\tBest {0} of valid set is {1:.3f} at {2} | bad count is {3}'.format(
                        metric, best_value, best_epoch, bad_counts))
                    bad_counts += 1

                if bad_counts >= self.early_stopping_patience:
                    print('\tEarly stopping at epoch %d' % e)
                    break
            # torch.cuda.empty_cache()
            # gc.collect()

        print('Training has finished')
        print('\tBest epoch is {0} | {1} of valid set is {2:.3f}'.format(best_epoch, metric, best_value))
        self.save_best_state_dict(best_epoch)
        print('Finish')

    def eval(self, istest=False, epoch=None):
        self.reibeg.eval()

        if istest:
            data_loader = self.test_data_loader
        else:
            data_loader = self.dev_data_loader
        data_loader.curr_tri_idx = 0

        # initial return data of validation
        data = {'MRR': 0, 'Hits@1': 0, 'Hits@5': 0, 'Hits@10': 0}
        ranks = []

        t = 0
        temp = dict()
        while True:
            if t > 6100:
                self.eval_batch_size = 128
            else:
                self.eval_batch_size = 0
            # sample all the eval tasks
            eval_task, curr_rel = data_loader.next_one_on_eval()
            # at the end of sample tasks, a symbol 'EOT' will return
            if eval_task == 'EOT':
                break
            t += 1

            if self.eval_batch_size > 0:
                def batch(iterable, n=1):
                    l = len(iterable)
                    for ndx in range(0, l, n):
                        yield [iterable[ndx:min(ndx + n, l)]]

                score_list = []
                support_triples, support_negative_triples, query_triple, negative_triples = eval_task
                self.reibeg.eval_reset()
                for neg_batch in batch(negative_triples[0], self.eval_batch_size):
                    eval_task_batch = [support_triples, support_negative_triples, query_triple, neg_batch]
                    _, p_score, n_score = self.do_one_step(epoch, eval_task_batch, iseval=True, curr_rel=curr_rel,
                                                           istest=istest, batch_eval=True)
                    score_list.append(n_score.detach().cpu())
                score_list.append(p_score.detach().cpu())
                x = torch.cat(score_list, 1).squeeze()
            else:
                _, p_score, n_score = self.do_one_step(epoch, eval_task, iseval=True, curr_rel=curr_rel, istest=istest)
                x = torch.cat([n_score, p_score], 1).squeeze()

            self.rank_predict(data, x, ranks)

            # print current temp data dynamically
            for k in data.keys():
                temp[k] = data[k] / t
            sys.stdout.write("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
                t, temp['MRR'], temp['Hits@10'], temp['Hits@5'], temp['Hits@1']))
            sys.stdout.flush()
            # if t>50:
            #    break

        # print overall evaluation result and return it
        for k in data.keys():
            data[k] = round(data[k] / t, 3)

        if self.parameter['step'] == 'train':
            self.logging_training_data(data, epoch)
        else:
            self.logging_eval_data(data, self.state_dict_file, istest)

        print("{}\tMRR: {:.3f}\tHits@10: {:.3f}\tHits@5: {:.3f}\tHits@1: {:.3f}\r".format(
            t, data['MRR'], data['Hits@10'], data['Hits@5'], data['Hits@1']))

        return data

