#!/usr/bin/env python3

import os
import time
import argparse
import torch as th
import numpy as np
from gtcrn import GTCRN
from libs.utils import load_json, get_logger
from libs.audio import write_wav
from libs.dataset_se import Dataset 

def run(args):
    start = time.time()
    logger = get_logger(
            os.path.join(args.checkpoint, 'separate.log'), file=True)
    dataset = Dataset(mix_scp=args.mix_scp, ref_scp=args.ref_scp, sample_rate=args.fs)
    
    # Load model
    # nnet_conf = load_json(args.checkpoint, "mdl.json")
    nnet = GTCRN()
    cpt_fname = os.path.join(args.checkpoint, "best.pt.tar")
    cpt = th.load(cpt_fname, map_location="cpu")
    nnet.load_state_dict(cpt["model_state_dict"]) 
    logger.info("Load checkpoint from {}, epoch {:d}".format(
        cpt_fname, cpt["epoch"]))
    
    device = th.device(
        "cuda:{}".format(args.gpuid)) if args.gpuid >= 0 else th.device("cpu")
    nnet = nnet.to(device) if args.gpuid >= 0 else nnet
    nnet.eval()
    
    with th.no_grad():
        total_cnt = 0
        for i, data in enumerate(dataset):    
            mix = th.tensor(data['mix'], dtype=th.float32, device=device)
            key = data['key']
            if args.gpuid >= 0:
                mix = mix.unsqueeze(0).to(device)
              
            # Forward   
            mix_spec = th.stft(mix, 256, 64, 256, th.hann_window(256).pow(0.5).to(mix.device), return_complex=False)          
            ests_spec = nnet(mix_spec)
            ests_complexspec = th.complex(ests_spec[..., 0], ests_spec[..., 1])
            ests = th.istft(ests_complexspec , 256, 64, 256, th.hann_window(256).pow(0.5).to(mix.device), return_complex=False) 
            ests = ests.squeeze(0).cpu().numpy() 
            # norm = np.linalg.norm(mix.cpu().numpy(), np.inf)
            ests = ests[:mix.shape[-1]] 
            # for each utts
            logger.info("Separate Utt{:d} key:{}".format(total_cnt + 1, key))
            # norm
            # ests = ests*norm/np.max(np.abs(ests)) 
            
            fname = key + '.wav'
            write_wav(os.path.join(args.dump_dir, fname),
                      ests, fs=args.fs)
            total_cnt += 1   
    
    end = time.time()
    logger.info('Utt={:d} | Time Elapsed: {:.1f}s'.format(total_cnt, end-start))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Separating speech...')
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Directory of checkpoint")
    parser.add_argument("--gpuid", type=int, default=-1, 
                        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument('--mix_scp', type=str, required=True,
                        help='mix scp')
    parser.add_argument('--ref_scp', type=str, required=True,
                        help='ref scp')
    parser.add_argument('--fs', type=int, default=16000, 
                        help="Sample rate for mixture input")
    parser.add_argument('--dump-dir', type=str, default="",
                        help="Directory to dump separated results out")
    args = parser.parse_args()
    run(args)
