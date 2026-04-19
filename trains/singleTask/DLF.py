import copy
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import optim
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..utils import MetricsTop, dict_to_str

logger = logging.getLogger('MMSA')


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}


class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()
        self.metrics = MetricsTop().getMetics(args.dataset_name)

        # Loss weights (paper-friendly defaults, configurable via json)
        self.w_task = getattr(args, 'loss_w_task', 1.0)
        self.w_orth = getattr(args, 'loss_w_orth', 0.05)
        self.w_sep = getattr(args, 'loss_w_sep', 0.05)
        self.w_pair = getattr(args, 'loss_w_pair', 0.05)
        self.pair_margin = getattr(args, 'loss_pair_margin', 0.2)
        self.ema_decay = getattr(args, 'ema_decay', 0.999)
        self.warmup_ratio = getattr(args, 'warmup_ratio', 0.1)

    def _build_optimizer_and_scheduler(self, model):
        base_lr = self.args.learning_rate
        bert_lr = getattr(self.args, 'bert_learning_rate', base_lr)
        min_lr = getattr(self.args, 'min_learning_rate', base_lr * 0.1)
        weight_decay = getattr(self.args, 'weight_decay', 0.0)

        no_decay_terms = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm.weight', 'layer_norm.bias']
        groups = {
            'bert_decay': {'params': [], 'lr': bert_lr, 'weight_decay': weight_decay},
            'bert_nodecay': {'params': [], 'lr': bert_lr, 'weight_decay': 0.0},
            'base_decay': {'params': [], 'lr': base_lr, 'weight_decay': weight_decay},
            'base_nodecay': {'params': [], 'lr': base_lr, 'weight_decay': 0.0},
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            is_bert = 'text_model' in name
            is_no_decay = any(nd in name for nd in no_decay_terms)
            if is_bert and is_no_decay:
                groups['bert_nodecay']['params'].append(param)
            elif is_bert:
                groups['bert_decay']['params'].append(param)
            elif is_no_decay:
                groups['base_nodecay']['params'].append(param)
            else:
                groups['base_decay']['params'].append(param)

        param_groups = [g for g in groups.values() if len(g['params']) > 0]
        optimizer = optim.AdamW(param_groups)

        total_epochs = max(1, self.args.update_epochs)
        warmup_epochs = max(1, int(total_epochs * self.warmup_ratio))
        min_lr_ratio = max(0.0, min(1.0, min_lr / base_lr))

        def lr_lambda(epoch_idx):
            if epoch_idx < warmup_epochs:
                return float(epoch_idx + 1) / float(warmup_epochs)
            progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return optimizer, scheduler

    @staticmethod
    def _pooled_cosine_loss(x, y):
        x_pool = x.mean(dim=1)
        y_pool = y.mean(dim=1)
        return 1.0 - F.cosine_similarity(x_pool, y_pool, dim=-1).mean()

    @staticmethod
    def _fro_dot_loss(x, y):
        # x,y: [B, T, D], encourage orthogonality via ||X^T Y||_F^2
        x_t = x.transpose(1, 2)
        cross = torch.bmm(x_t, y)
        return (cross ** 2).mean()

    def _compute_losses(self, output, labels, reg_scale=1.0):
        # 1) Task loss (final + global + local branch)
        loss_task_main = self.criterion(output['output_logit'], labels)
        loss_task_s = self.criterion(output['logits_s'], labels)
        loss_task_c = self.criterion(output['logits_c'], labels)
        loss_task = loss_task_main + 0.5 * (loss_task_s + loss_task_c)

        # 2) Disentanglement constraints
        p_l, p_a, p_v = output['p_l'], output['p_a'], output['p_v']
        s_g = output['s_g']
        s_la, s_lv, s_av = output['s_la'], output['s_lv'], output['s_av']

        # A) private and global shared orthogonality / decorrelation
        loss_orth = self._fro_dot_loss(p_l, s_g) + self._fro_dot_loss(p_a, s_g) + self._fro_dot_loss(p_v, s_g)

        # B) pairwise shared and global shared redundancy reduction
        loss_sep = (
            self._fro_dot_loss(s_la, s_g)
            + self._fro_dot_loss(s_lv, s_g)
            + self._fro_dot_loss(s_av, s_g)
        )

        # C) pairwise shared should approach related two-modal common semantics and repel the unrelated modality
        target_la = 0.5 * (p_l + p_a)
        target_lv = 0.5 * (p_l + p_v)
        target_av = 0.5 * (p_a + p_v)

        s_la_pool = s_la.mean(dim=1)
        s_lv_pool = s_lv.mean(dim=1)
        s_av_pool = s_av.mean(dim=1)
        t_la_pool = target_la.mean(dim=1)
        t_lv_pool = target_lv.mean(dim=1)
        t_av_pool = target_av.mean(dim=1)
        p_l_pool = p_l.mean(dim=1)
        p_a_pool = p_a.mean(dim=1)
        p_v_pool = p_v.mean(dim=1)

        pos_la = 1.0 - F.cosine_similarity(s_la_pool, t_la_pool, dim=-1)
        pos_lv = 1.0 - F.cosine_similarity(s_lv_pool, t_lv_pool, dim=-1)
        pos_av = 1.0 - F.cosine_similarity(s_av_pool, t_av_pool, dim=-1)

        neg_la = F.relu(
            F.cosine_similarity(s_la_pool, p_v_pool, dim=-1)
            - F.cosine_similarity(s_la_pool, t_la_pool, dim=-1)
            + self.pair_margin
        )
        neg_lv = F.relu(
            F.cosine_similarity(s_lv_pool, p_a_pool, dim=-1)
            - F.cosine_similarity(s_lv_pool, t_lv_pool, dim=-1)
            + self.pair_margin
        )
        neg_av = F.relu(
            F.cosine_similarity(s_av_pool, p_l_pool, dim=-1)
            - F.cosine_similarity(s_av_pool, t_av_pool, dim=-1)
            + self.pair_margin
        )

        loss_pair = (
            pos_la.mean() + pos_lv.mean() + pos_av.mean()
            + neg_la.mean() + neg_lv.mean() + neg_av.mean()
        )

        total_loss = (
            self.w_task * loss_task
            + (self.w_orth * reg_scale) * loss_orth
            + (self.w_sep * reg_scale) * loss_sep
            + (self.w_pair * reg_scale) * loss_pair
        )

        loss_items = {
            'task': loss_task.detach().item(),
            'orth': loss_orth.detach().item(),
            'sep': loss_sep.detach().item(),
            'pair': loss_pair.detach().item(),
        }
        return total_loss, loss_items

    def do_train(self, model, dataloader, return_epoch_results=False):
        optimizer, scheduler = self._build_optimizer_and_scheduler(model)
        ema = ModelEMA(model, decay=self.ema_decay)

        best_valid = float('inf')
        best_epoch = 0
        best_state_dict = copy.deepcopy(model.state_dict())

        train_losses, val_losses = [], []
        train_has0_acc, val_has0_acc = [], []
        train_non0_acc, val_non0_acc = [], []
        train_has0_f1, val_has0_f1 = [], []
        train_non0_f1, val_non0_f1 = [], []
        train_mult5, val_mult5 = [], []
        train_mult7, val_mult7 = [], []
        train_mae, val_mae = [], []

        for epoch in range(self.args.update_epochs):
            y_pred, y_true = [], []
            model.train()
            train_loss = 0.0
            epoch_parts = {'task': 0.0, 'orth': 0.0, 'sep': 0.0, 'pair': 0.0}

            with tqdm(dataloader['train'], desc=f"Epoch {epoch + 1}") as td:
                for batch_data in td:
                    optimizer.zero_grad()

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    output = model(text, audio, vision)
                    reg_warmup_epochs = max(1, int(self.args.update_epochs * 0.3))
                    reg_scale = min(1.0, float(epoch + 1) / float(reg_warmup_epochs))
                    combined_loss, loss_items = self._compute_losses(output, labels, reg_scale=reg_scale)
                    combined_loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_norm_(list(model.parameters()), self.args.grad_clip)

                    optimizer.step()
                    ema.update(model)

                    train_loss += combined_loss.item()
                    for k in epoch_parts:
                        epoch_parts[k] += loss_items[k]
                    y_pred.append(output['output_logit'].detach().cpu())
                    y_true.append(labels.cpu())

            train_loss = train_loss / len(dataloader['train'])
            for k in epoch_parts:
                epoch_parts[k] = epoch_parts[k] / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epoch + 1} TRAIN -({self.args.model_name}) [{epoch + 1}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} | "
                f"task: {epoch_parts['task']:.4f} orth: {epoch_parts['orth']:.4f} "
                f"sep: {epoch_parts['sep']:.4f} pair: {epoch_parts['pair']:.4f} "
                f"{dict_to_str(train_results)}"
            )

            val_results = self.do_test(model, dataloader['valid'], mode="VAL", ema=ema)
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_results['Loss'])

            train_has0_acc.append(train_results['Has0_acc_2'])
            val_has0_acc.append(val_results['Has0_acc_2'])

            train_non0_acc.append(train_results['Non0_acc_2'])
            val_non0_acc.append(val_results['Non0_acc_2'])

            train_has0_f1.append(train_results['Has0_F1_score'])
            val_has0_f1.append(val_results['Has0_F1_score'])

            train_non0_f1.append(train_results['Non0_F1_score'])
            val_non0_f1.append(val_results['Non0_F1_score'])

            train_mult5.append(train_results['Mult_acc_5'])
            val_mult5.append(val_results['Mult_acc_5'])

            train_mult7.append(train_results['Mult_acc_7'])
            val_mult7.append(val_results['Mult_acc_7'])

            train_mae.append(train_results['MAE'])
            val_mae.append(val_results['MAE'])

            if val_results['Loss'] < best_valid:
                best_valid = val_results['Loss']
                best_epoch = epoch + 1
                ema.apply_shadow(model)
                best_state_dict = copy.deepcopy(model.state_dict())
                ema.restore(model)

            if (epoch + 1) - best_epoch >= self.args.early_stop:
                logger.info(f"Early stop at epoch {epoch + 1}")
                break

        best_path = str(self.args.model_save_path)
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        torch.save(best_state_dict, best_path)

        model.load_state_dict(best_state_dict)
        best_test = self.do_test(model, dataloader['test'], mode="TEST")
        logger.info(f"Best Epoch: {best_epoch} | Best Test: {dict_to_str(best_test)}")

        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(14, 18))

        plt.subplot(5, 1, 1)
        plt.plot(epochs, train_losses, label='Train Loss')
        plt.plot(epochs, val_losses, label='Val Loss')
        plt.title('Loss')
        plt.legend()
        plt.grid()

        plt.subplot(5, 1, 2)
        plt.plot(epochs, train_has0_acc, label='Train Has0 Acc')
        plt.plot(epochs, val_has0_acc, label='Val Has0 Acc')
        plt.plot(epochs, train_non0_acc, '--', label='Train Non0 Acc')
        plt.plot(epochs, val_non0_acc, '--', label='Val Non0 Acc')
        plt.title('Binary Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(5, 1, 3)
        plt.plot(epochs, train_has0_f1, label='Train Has0 F1')
        plt.plot(epochs, val_has0_f1, label='Val Has0 F1')
        plt.plot(epochs, train_non0_f1, '--', label='Train Non0 F1')
        plt.plot(epochs, val_non0_f1, '--', label='Val Non0 F1')
        plt.title('Binary F1 Score')
        plt.legend()
        plt.grid()

        plt.subplot(5, 1, 4)
        plt.plot(epochs, train_mult5, label='Train Acc@5')
        plt.plot(epochs, val_mult5, label='Val Acc@5')
        plt.plot(epochs, train_mult7, '--', label='Train Acc@7')
        plt.plot(epochs, val_mult7, '--', label='Val Acc@7')
        plt.title('Multi-class Accuracy')
        plt.legend()
        plt.grid()

        plt.subplot(5, 1, 5)
        plt.plot(epochs, train_mae, label='Train MAE')
        plt.plot(epochs, val_mae, label='Val MAE')
        plt.title('Regression Metrics')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300)
        plt.close()

        return best_test

    def do_test(self, model, dataloader, mode="VAL", ema=None):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0

        if ema is not None:
            ema.apply_shadow(model)

        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device).view(-1, 1)

                    output = model(text, audio, vision)
                    loss = self.criterion(output['output_logit'], labels)#L1损失作为评判最优模型的标准
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        if ema is not None:
            ema.restore(model)

        eval_results = self.metrics(pred, true)#遍历完数据集后，这里计算分类指标和回归指标
        eval_results['Loss'] = round(eval_loss, 4)#新增Loss作为key，eval_loss作为值
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        return eval_results
