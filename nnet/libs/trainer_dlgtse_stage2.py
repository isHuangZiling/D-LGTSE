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
    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))
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
                 nnet_se,
                 checkpoint="checkpoint",
                 pre_pse_cpt="pre_pse_cpt",
                 pre_se_cpt="pre_se_cpt",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=1.0,
                 pretrained=False,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 no_impr=100,
                 freeze_se=True,):
        self.freeze_se = freeze_se 
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )
        self.device = th.device("cuda:{}".format(gpuid[0]))      
        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.pre_pse_cpt = pre_pse_cpt
        self.pre_se_cpt = pre_se_cpt
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.optimizer_name = optimizer
        self.optimizer_kwargs = optimizer_kwargs
        # ====== 加一个工具函数 ======
        def strip_module(state_dict):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith("module.") else k
                new_state_dict[name] = v
            return new_state_dict
        # ============================
        
        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # ====== 加 strip_module ======
            nnet.load_state_dict(strip_module(cpt["model_state_dict"]))
            nnet_se.load_state_dict(strip_module(cpt["model_se_state_dict"]))
            # ==============================

            self.nnet = th.nn.DataParallel(nnet.to(self.device), device_ids=self.gpuid)
            self.nnet_se = th.nn.DataParallel(nnet_se.to(self.device), device_ids=self.gpuid)
            self.logger.info(f"Models wrapped with DataParallel on GPUs {self.gpuid}")

            self.optimizer = self.create_optimizer(
                self.nnet, self.optimizer_name, self.optimizer_kwargs, state=cpt["optim_state_dict"])
        else:
            if pretrained:
                pse_cpt = th.load(pre_pse_cpt, map_location="cpu")
                se_cpt = th.load(pre_se_cpt, map_location="cpu")
                self.logger.info(f"load pretrained PSE SE checkpoint {pre_pse_cpt} {pre_se_cpt}: epoch {pse_cpt['epoch']} {se_cpt['epoch']}")
                # ====== 加 strip_module ======
                nnet.load_state_dict(strip_module(pse_cpt["model_state_dict"]))
                nnet_se.load_state_dict(strip_module(se_cpt["model_state_dict"]))
                # ==============================
            else:
                se_cpt = th.load(pre_se_cpt, map_location="cpu")
                self.logger.info(f"load pretrained SE checkpoint {pre_se_cpt}: epoch {se_cpt['epoch']}")
                # ====== 加 strip_module ======
                nnet_se.load_state_dict(strip_module(se_cpt["model_state_dict"]))
                # ==============================

                self.nnet = th.nn.DataParallel(nnet.to(self.device), device_ids=self.gpuid)
                self.nnet_se = th.nn.DataParallel(nnet_se.to(self.device), device_ids=self.gpuid)
                self.logger.info(f"Models wrapped with DataParallel on GPUs {self.gpuid}")
                self.optimizer = self.create_optimizer(self.nnet, self.optimizer_name, self.optimizer_kwargs)

        self.scheduler1 = StepLR(self.optimizer, step_size=2, gamma=0.98)
        self.scheduler2 = StepLR(self.optimizer, step_size=1, gamma=0.9)
 
        self.num_params_nnet = sum(param.nelement() for param in nnet.parameters()) / 1e6
        self.num_params_nnet_se = sum(param.nelement() for param in nnet_se.parameters()) / 1e6
        self.num_params = self.num_params_nnet + self.num_params_nnet_se

        self.logger.info("Model summary (Main model):\n{}".format(nnet))
        self.logger.info("Model summary (SE model):\n{}".format(nnet_se))
        self.logger.info("Loading model to GPUs: {}, #params: main = {:.2f}M, se = {:.2f}M, total = {:.2f}M".format(
            gpuid, self.num_params_nnet, self.num_params_nnet_se, self.num_params))
                
        if clip_norm > 0:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "model_se_state_dict": self.nnet_se.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }
        
        filename = "best" if best else "last"
        th.save(cpt, os.path.join(self.checkpoint, f"{filename}.pt.tar"))
       
    def save_every_checkpoint(self, idx):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "model_se_state_dict": self.nnet_se.state_dict(),
            "optim_state_dict": self.optimizer.state_dict(),
        }      
        th.save(cpt, os.path.join(self.checkpoint,
                "{0}.pt.tar".format(str(idx))))   

    def create_optimizer(self,  opt_nnet, optimizer, kwargs, state=None):           
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
        opt = supported_optimizer[optimizer](opt_nnet.parameters(), **kwargs)
        
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs):
        raise NotImplementedError

    # def train(self, data_loader):
    #     self.logger.info("Set train mode...")
    #     self.nnet.train()

    def train(self, data_loader):
        self.logger.info("Set train mode...")
        self.nnet.train()
        self.nnet_se.eval()   # ← 加这行，冻结 BN 的 running statistics
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        # ... 后面不变

        # reporter = ProgressReporter(self.logger, period=self.logging_period)

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)             
            self.optimizer.zero_grad()
            loss = self.compute_loss(egs)
            loss.backward()
            if self.clip_norm > 0:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()
            reporter.add(loss.item())
        return reporter.report()

    def eval(self, data_loader):
        self.logger.info("Set eval mode...")
        self.nnet.eval()
        self.nnet_se.eval()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
             
        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)
                loss = self.compute_loss(egs)
                reporter.add(loss.item())
        return reporter.report(details=True)

    def run(self, train_loader, dev_loader, num_epochs=120):
            if self.freeze_se:
                self.logger.info("SE model parameters are initially FROZEN!!!")
            reporter = ProgressReporter(self.logger, period=self.logging_period)
            with th.cuda.device(self.gpuid[0]):
                stats = dict()
                self.save_checkpoint(best=False)
                best_loss = float('inf')
                self.logger.info("START FROM EPOCH {:d}, LOSS = {:.4f}".format(
                    self.cur_epoch, best_loss))
                no_impr = 0

                while self.cur_epoch < num_epochs:
                    self.cur_epoch += 1
                    cur_lr1 = self.optimizer.param_groups[0]["lr"]
                    stats["title"] = f"Loss(time/N, {cur_lr1}) - Epoch {self.cur_epoch:2d}:"
                    tr = self.train(train_loader)
                    stats["tr"] = "train = {:+.4f}({:.2f}m/{:d})".format(
                        tr["loss"], tr["cost"], tr["batches"])
                    cv = self.eval(dev_loader)
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

                self.logger.info("Training for {:d}/{:d} epoches done!".format(
                    self.cur_epoch, num_epochs))



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
        N = egs["mix_both"].size(0)
        
        with th.no_grad():
            both_spec = th.stft(egs["mix_both"], 256, 64, 256, 
                            th.hann_window(256).pow(0.5).to(egs["mix_both"].device), 
                            return_complex=False)
            se_out_spec = self.nnet_se(both_spec)  
            seout_complexspec = th.complex(se_out_spec[..., 0], se_out_spec[..., 1])
            se_out = th.istft(seout_complexspec, 256, 64, 256, 
                            th.hann_window(256).pow(0.5).to(egs["mix_both"].device), 
                            return_complex=False)
        
        enhanced_2spkn = self.nnet(egs["mix_both"], se_out, egs["aux"])  
        
        clean_target = egs["ref"]       
        valid_len = egs["valid_len"] 
        
        enhanced_2spkn = self.mask_by_length(enhanced_2spkn, valid_len)
        clean_target = self.mask_by_length(clean_target, valid_len)
    
        loss1 = -th.sum(self.sisnr(enhanced_2spkn, clean_target)) / N

        return loss1