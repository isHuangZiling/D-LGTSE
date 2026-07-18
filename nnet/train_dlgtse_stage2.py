import pprint 
import argparse 
from libs.trainer_dlgtse_stage2 import SiSnrTrainer
from libs.dataset_tse_dlgtse import make_dataloader
from libs.utils import dump_json, get_logger
from model_SEF_PNet_2stage_sesim import DenseUNet
# from model_CIENet_mDPTNet_2stage import FreqNet
from gtcrn import GTCRN
from conf_unet_tse_dlgtse import trainer_conf, nnet_conf, train_data, dev_data, chunk_size

logger = get_logger(__name__) 

def run(args): 
    gpuids = tuple(map(int, args.gpus.split(",")))
    nnet = DenseUNet(**nnet_conf)  
    # nnet = FreqNet()
    nnet_se = GTCRN()
    trainer = SiSnrTrainer(nnet,
                           nnet_se,  
                           gpuid=gpuids,
                           checkpoint=args.checkpoint,
                           pre_pse_cpt=args.pretrained_pse_cpt,
                           pre_se_cpt=args.pretrained_se_cpt,
                           resume=args.resume,
                           **trainer_conf)
    
    data_conf = {  
        "train": train_data,
        "dev": dev_data,
        "chunk_size": chunk_size
    }
    
    for conf, fname in zip([nnet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, args.checkpoint, fname)
    
    train_loader = make_dataloader(train=True,
                                   data_kwargs=train_data,
                                   batch_size=args.batch_size,
                                   chunk_size=chunk_size,
                                   num_workers=args.num_workers)
    dev_loader = make_dataloader(train=False,
                                 data_kwargs=dev_data,
                                 batch_size=args.batch_size,
                                 chunk_size=chunk_size,
                                 num_workers=args.num_workers)
    trainer.run(train_loader, dev_loader, num_epochs=args.epochs) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="1",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        default='',
                        help="Directory to dump models")
    parser.add_argument("--pretrained_pse_cpt",
                        type=str,
                        default=None,
                        help="Directory to dump models")
    parser.add_argument("--pretrained_se_cpt",
                        type=str,
                        default='/node/hzl/expriment2025/project_se/gtcrn/best.pt.tar',
                        help="Directory to dump models")
    parser.add_argument("--resume", 
                        type=str,
                        default=None,
                        help="Exist model to resume training from")
    parser.add_argument("--batch-size",
                        type=int,
                        default=2,
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers", 
                        type=int,
                        default=2,
                        help="Number of workers used in data loader")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
   
    run(args)
    print("train Done!")