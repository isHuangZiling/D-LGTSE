#!/usr/bin/env python3

import os
import time
import argparse
import torch as th
import numpy as np
from mir_eval.separation import bss_eval_sources
from pesq import pesq as pesq2
from pypesq import pesq as pesq1
from pystoi.stoi import stoi
from model_CIENet_mDPTNet_2stage import FreqNet
# from model_SEF_PNet_2stage_sesim import DenseUNet
from gtcrn import GTCRN
from libs.utils import load_json, get_logger
from libs.dataset_tse_dlgtse import Dataset
from torch.utils.data import DataLoader

def evaluate(args, model_file, logger):
    start = time.time()
    total_SISNR = 0
    total_SISNRi = 0
    total_PESQ = 0
    total_PESQi = 0
    total_PESQ2 = 0
    total_PESQi2 = 0
    total_STOI = 0
    total_STOIi = 0
    total_SDR = 0
    total_cnt = 0

    # Load model
    nnet = FreqNet()

    # nnet_conf = load_json(args.checkpoint, "mdl.json")
    # nnet = DenseUNet(**nnet_conf)

    nnet2 = GTCRN()
    cpt_fname = os.path.join(args.checkpoint, model_file)
    cpt = th.load(cpt_fname, map_location="cpu")
    logger.info(f"Loading checkpoint: {model_file}")
    if "epoch" in cpt:
        logger.info(f"Checkpoint epoch: {cpt['epoch']}")
    else:
        logger.info("No 'epoch' key in checkpoint")

    # ====== strip_module ======
    def strip_module(state_dict):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            new_state_dict[name] = v
        return new_state_dict
    nnet.load_state_dict(strip_module(cpt["model_state_dict"]))
    nnet2.load_state_dict(strip_module(cpt["model_se_state_dict"]))
    # ==============================

    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet2 = nnet2.to(device) if args.gpuid >= 0 else nnet2
    nnet.eval()
    nnet2.eval()

    # Load data
    dataset = Dataset(mix_both_scp=args.mix_both_scp, mix_clean_scp=args.mix_clean_scp,ref_scp=args.ref_scp, aux_scp=args.aux_scp, sample_rate=8000)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=lambda x: x[0])

    with th.no_grad():
        for data in dataloader:
            mix = th.tensor(data['mix_both'], dtype=th.float32, device=device).unsqueeze(0)
            mix_clean = th.tensor(data['mix_clean'], dtype=th.float32, device=device).unsqueeze(0)
            aux = th.tensor(data['aux'], dtype=th.float32, device=device).unsqueeze(0)

            # Forward
            ref = data['ref']
            key = data['key']
            both_spec = th.stft(mix, 256, 64, 256,
                            th.hann_window(256).pow(0.5).to(device),
                            return_complex=False)
            se_out_spec = nnet2(both_spec)
            seout_complexspec = th.complex(se_out_spec[..., 0], se_out_spec[..., 1])
            se_out = th.istft(seout_complexspec, 256, 64, 256,
                            th.hann_window(256).pow(0.5).to(device),
                            return_complex=False)
            ests = nnet(mix, se_out, aux)
            ests = ests.squeeze(0).cpu().numpy()
            mix_np = mix.squeeze(0).cpu().numpy()
            if ests.size != ref.size:
                end = min(ests.size, ref.size)
                ests = ests[:end]
                ref = ref[:end]
                mix_np = mix_np[:end]

            # Compute metrics
            if args.cal_sdr == 1:
                SDR, sir, sar, popt = bss_eval_sources(ref, ests)
                total_SDR += SDR[0]
            SISNR, delta = cal_SISNRi(ests, ref, mix_np)
            PESQ, PESQi, PESQ2, PESQi2 = cal_PESQi(ests, ref, mix_np)
            STOI, STOIi = cal_STOIi(ests, ref, mix_np)
            if args.cal_sdr == 1:
                logger.info("Key={} | Utt={:d} | SDR={:.3f} | SI-SNR={:.3f} | SI-SNRi={:.3f} | PESQ={:.3f} | PESQi={:.3f}| PESQ2={:.3f} | PESQi2={:.3f} | STOI={:.2f} | STOIi={:.2f}".format(
                    key, total_cnt+1, SDR[0], SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))
            else:
                logger.info("Key={} | Utt={:d} | SI-SNR={:.2f} | SI-SNRi={:.2f} | PESQ={:.2f} | PESQi={:.2f} | PESQ2={:.2f} | PESQi2={:.2f} | STOI={:.2f} | STOIi={:.2f}".format(
                    key, total_cnt+1, SISNR, delta, PESQ, PESQi, PESQ2, PESQi2, STOI, STOIi))

            total_SISNR += SISNR
            total_SISNRi += delta
            total_PESQ += PESQ
            total_PESQi += PESQi
            total_PESQ2 += PESQ2
            total_PESQi2 += PESQi2
            total_STOI += STOI
            total_STOIi += STOIi
            total_cnt += 1
    end = time.time()

    logger.info('Time Elapsed: {:.1f}s'.format(end-start))
    if args.cal_sdr == 1:
        logger.info("Average SDR: {0:.3f}".format(total_SDR / total_cnt))
    logger.info("Average SI-SNR: {:.3f}".format(total_SISNR / total_cnt))
    logger.info("Average SI-SNRi: {:.3f}".format(total_SISNRi / total_cnt))
    logger.info("Average PESQ: {:.3f}".format(total_PESQ / total_cnt))
    logger.info("Average PESQi: {:.3f}".format(total_PESQi / total_cnt))
    logger.info("Average PESQ2: {:.3f}".format(total_PESQ2 / total_cnt))
    logger.info("Average PESQi2: {:.3f}".format(total_PESQi2 / total_cnt))
    logger.info("Average STOI: {:.2f}".format(total_STOI / total_cnt))
    logger.info("Average STOIi: {:.2f}".format(total_STOIi / total_cnt))

def cal_SISNR(est, ref, eps=1e-8):
    assert len(est) == len(ref)
    est_zm = est - np.mean(est)
    ref_zm = ref - np.mean(ref)
    t = np.sum(est_zm * ref_zm) * ref_zm / (np.linalg.norm(ref_zm)**2 + eps)
    return 20 * np.log10(eps + np.linalg.norm(t) / (np.linalg.norm(est_zm - t) + eps))

def cal_SISNRi(est, ref, mix, eps=1e-8):
    assert len(est) == len(ref) == len(mix)
    sisnr1 = cal_SISNR(est, ref)
    sisnr2 = cal_SISNR(mix, ref)
    return sisnr1, sisnr1 - sisnr2

def cal_PESQ(est, ref):
    assert len(est) == len(ref)
    mode = 'nb'
    p = pesq1(ref, est, 8000)
    p2 = pesq2(8000, ref, est, mode)
    return p, p2

def cal_PESQi(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    pesq1_v, pesq12 = cal_PESQ(est, ref)
    pesq2_v, pesq22 = cal_PESQ(mix, ref)
    return pesq1_v, pesq1_v - pesq2_v, pesq12, pesq12 - pesq22

def cal_STOI(est, ref):
    assert len(est) == len(ref)
    p = stoi(ref, est, 8000)
    return p

def cal_STOIi(est, ref, mix):
    assert len(est) == len(ref) == len(mix)
    stoi1 = cal_STOI(est, ref) * 100
    stoi2 = cal_STOI(mix, ref) * 100
    return stoi1, stoi1 - stoi2

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluate separation performance')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model directory containing checkpoints')
    parser.add_argument('--gpuid', type=int, default=0,
                        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument('--mix_both_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--mix_clean_scp', type=str, required=True,
                        help='mix clean scp')
    parser.add_argument('--ref_scp', type=str, required=True,
                        help='ref scp')
    parser.add_argument('--aux_scp', type=str, required=True,
                        help='aux scp')
    parser.add_argument('--cal_sdr', type=int, default=None,
                        help='Whether calculate SDR')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--eval_epochs', type=str, default=None,
                        help='Comma separated epochs to evaluate, e.g. "100,110,120"')

    args = parser.parse_args()


    best_model_file = "best.pt.tar"
    best_log_file = os.path.join(args.checkpoint, "eval_best.log")
    best_logger = get_logger(best_log_file, file=True)
    best_logger.info(f"Evaluating model: {best_model_file}")
    evaluate(args, best_model_file, best_logger)

    # eval epochs
    if args.eval_epochs:
        epochs = [int(e) for e in args.eval_epochs.split(',')]
    else:
        epochs = range(100, 121)

    for epoch in epochs:
        model_file = f"{epoch}.pt.tar"
        model_path = os.path.join(args.checkpoint, model_file)
        if not os.path.exists(model_path):
            continue
        log_file = os.path.join(args.checkpoint, f"eval_{epoch}.log")
        logger = get_logger(log_file, file=True)
        logger.info(f"Evaluating model: {model_file}")
        evaluate(args, model_file, logger)
