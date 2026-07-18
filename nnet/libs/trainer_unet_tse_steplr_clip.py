# wujian@2018

import os
import sys
import time

# from itertools import permutations
from collections import defaultdict

import torch as th
import torch.nn.functional as F
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_

from .utils import get_logger
import matplotlib.pyplot as plt

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()
        # self.loss_writer = SummaryWriter('/Share/hzl/sDPCCN/tensorboard_loss')
    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))
            # self.loss_writer.add_scalar('Loss/train', avg, N)
    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }

class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=1.0,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=100):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))      
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr

        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))

            # ====== 加 strip_module ======
            def strip_module(state_dict):
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith("module.") else k
                    new_state_dict[name] = v
                return new_state_dict
            nnet.load_state_dict(strip_module(cpt["model_state_dict"]))
            # ==============================


            # ====== 改这里 ======
            self.nnet = th.nn.DataParallel(nnet.to(self.device), device_ids=self.gpuid)
            # ====================
            self.optimizer = self.create_optimizer(
                optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])

        else:
            # ====== 改这里 ======
            self.nnet = th.nn.DataParallel(nnet.to(self.device), device_ids=self.gpuid)
            # ====================
            self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer,
        #     mode="min",
        #     factor=factor,
        #     patience=patience,
        #     min_lr=min_lr,
        #     verbose=True)
        self.scheduler1 = StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.scheduler2 = StepLR(self.optimizer, step_size=1, gamma=0.9)
        
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm > 0:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))
        
    def save_every_checkpoint(self, idx):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(cpt, os.path.join(self.checkpoint,
                "{0}.pt.tar".format(str(idx))))   

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)

        # 用于收集指标
        sisnr_losses = []
        align_losses = []
        gap_means = []
        weight_means = []
        scale_factors = []

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device) 
            
            self.optimizer.zero_grad()
            outputs = self.compute_loss(egs)

            # 根据训练或评估，outputs是tuple或tensor
            if isinstance(outputs, tuple):
                loss = outputs[0]
                sisnr_losses.append(outputs[1])
                align_losses.append(outputs[2])
                gap_means.append(outputs[3])
                weight_means.append(outputs[4])
                scale_factors.append(outputs[5])
            else:
                loss = outputs
            
            
            loss.backward()

            if self.clip_norm > 0:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()

            reporter.add(loss.item())
            

        if self.cur_epoch % 5 == 0 and sisnr_losses:
            self.logger.info(
                f"[Epoch {self.cur_epoch}] SISNR_loss={sum(sisnr_losses)/len(sisnr_losses):.4f}, "
                f"Align_loss={sum(align_losses)/len(align_losses):.4f}, "
                f"gap.mean={sum(gap_means)/len(gap_means):.4f}, "
                f"weight.mean={sum(weight_means)/len(weight_means):.4f}, "
                f"scale_factor={sum(scale_factors)/len(scale_factors):.1f}"
            )

        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
             
        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                # loss = self.compute_loss(egs)

                outputs = self.compute_loss(egs)
                if isinstance(outputs, tuple):
                    loss = outputs[0]
                else:
                    loss = outputs
                reporter.add(loss.item())

                # reporter.add(loss.item())
        # return reporter.report(details=True)
        return reporter.report()

    def run(self, train_loader, dev_loader, num_epochs=120):
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        with th.cuda.device(self.gpuid[0]):
            stats = dict()
            # ====== 只去掉 eval，保留 save ======
            self.save_checkpoint(best=False)
            best_loss = float('inf')
            self.logger.info("START FROM EPOCH {:d}, LOSS = inf".format(
                self.cur_epoch))
            # =====================================
            no_impr = 0

            # ====== resume 时从 CSV 加载历史数据 ======
            csv_path = os.path.join(self.checkpoint, 'loss_history.csv')
            train_losses = []
            dev_losses = []
            if self.cur_epoch > 0 and os.path.exists(csv_path):
                import csv
                with open(csv_path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        train_losses.append(float(row[1]))
                        dev_losses.append(float(row[2]))
                self.logger.info(f"Loaded {len(train_losses)} epochs of loss history from CSV")
            # =============================================

            while self.cur_epoch < num_epochs:
                self.cur_epoch += 1
                cur_lr = self.optimizer.param_groups[0]["lr"]
                stats[
                    "title"] = "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                        cur_lr, self.cur_epoch)
                tr = self.train(train_loader)
                train_losses.append(tr["loss"])
                stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                    tr["loss"], tr["cost"], tr["batches"])
                cv = self.eval(dev_loader)
                dev_losses.append(cv["loss"])
                stats["cv"] = "dev = {:+.4f}({:.2f}m/{:d})".format(
                    cv["loss"], cv["cost"], cv["batches"])
                stats["scheduler"] = ""
                if cv["loss"] > best_loss:
                    no_impr += 1
                    stats["scheduler"] = "| no impr, best = {:.4f}".format(
                        cv["loss"])
                else:
                    best_loss = cv["loss"]
                    no_impr = 0
                    self.save_checkpoint(best=True)
                if self.cur_epoch >= 100:
                    self.save_every_checkpoint(self.cur_epoch)
                self.logger.info(
                    "{title} {tr} | {cv} {scheduler}".format(**stats))
                if self.cur_epoch <= 100:
                    self.scheduler1.step()
                else:
                    self.scheduler2.step()
                sys.stdout.flush()
                self.save_checkpoint(best=False)

                if no_impr == self.no_impr:
                    self.logger.info(
                        "Stop training cause no impr for {:d} epochs".format(
                            no_impr))
                    break

                # ========== 保存 loss 数据到 CSV ==========
                self._save_loss_csv(train_losses, dev_losses)
                # ========== 画图 ==========
                self._plot_loss_curve(train_losses, dev_losses, best_loss)

            self.logger.info("Training for {:d}/{:d} epoches done!".format(
                self.cur_epoch, num_epochs))


    def _save_loss_csv(self, train_losses, dev_losses):
        """保存每个 epoch 的 train/val loss 到 CSV，方便后续多实验联合画图"""
        import csv
        csv_path = os.path.join(self.checkpoint, 'loss_history.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'dev_loss'])
            for i, (tr, dv) in enumerate(zip(train_losses, dev_losses)):
                writer.writerow([i + 1, tr, dv])

    def _plot_loss_curve(self, train_losses, dev_losses, best_loss):
        """美化版 loss 曲线图"""
        import numpy as np

        epochs = np.arange(1, len(train_losses) + 1)
        train_arr = np.array(train_losses)
        dev_arr = np.array(dev_losses)

        # --- 移动平均平滑（窗口=5） ---
        def moving_avg(x, w=5):
            if len(x) < w:
                return x, np.arange(1, len(x) + 1)
            kernel = np.ones(w) / w
            smoothed = np.convolve(x, kernel, mode='valid')
            return smoothed, np.arange(w, len(x) + 1)

        tr_smooth, tr_ep = moving_avg(train_arr, w=5)
        cv_smooth, cv_ep = moving_avg(dev_arr, w=5)

        # --- 找 best epoch ---
        best_epoch = int(np.argmin(dev_arr)) + 1
        best_val = float(np.min(dev_arr))

        # --- 画图 ---
        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        # 原始曲线（半透明）
        ax.plot(epochs, train_arr, color='#4A90D9', alpha=0.2, linewidth=0.8)
        ax.plot(epochs, dev_arr, color='#E8724A', alpha=0.2, linewidth=0.8)

        # 平滑曲线
        ax.plot(tr_ep, tr_smooth, color='#4A90D9', linewidth=2.0, label='Train Loss')
        ax.plot(cv_ep, cv_smooth, color='#E8724A', linewidth=2.0, label='Dev Loss')

        # 标注 best 点
        ax.scatter([best_epoch], [best_val], color='#E8724A', s=60, zorder=5,
                edgecolors='white', linewidths=1.2)
        ax.annotate(f'best={best_val:.4f}\n(epoch {best_epoch})',
                    xy=(best_epoch, best_val),
                    xytext=(best_epoch + max(2, len(epochs) * 0.05), best_val),
                    fontsize=8.5, color='#C05030',
                    arrowprops=dict(arrowstyle='->', color='#C05030', lw=1.0),
                    bbox=dict(boxstyle='round,pad=0.3', fc='#FFF5F0', ec='#C05030', alpha=0.9))

        # 坐标轴 & 样式
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training & Validation Loss', fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.set_xlim(1, len(epochs))

        # 去掉上方和右侧边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        fig.savefig(os.path.join(self.checkpoint, 'loss_vs_epoch.png'))
        plt.close(fig)


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

    def sisnr(self, x, s, eps=1e-8): 
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def mask_by_length(self, xs, lengths, fill=0):
        """
        Mask tensor according to length 
        """
        assert xs.size(0) == len(lengths)
        ret = xs.data.new(*xs.size()).fill_(fill) 
        for i, l in enumerate(lengths):
            ret[i, :l] = xs[i, :l]
        return ret 

    def compute_loss(self, egs):
        N = egs["mix"].size(0)
        
        # spks x n x S
        # ====== 去掉 DataParallel，直接用 self.nnet ======
        ests = self.nnet(egs["mix"], egs["aux"])
        # =================================================
       
        refs = egs['ref']
        # N = egs["mix"].size(0)
        valid_len = egs["valid_len"] 
        ests = self.mask_by_length(ests, valid_len)
        refs = self.mask_by_length(refs, valid_len) 
        sisnr_loss = -th.sum(self.sisnr(ests, refs)) / N

        return sisnr_loss
