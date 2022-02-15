import argparse, random, os, random, sys, glob, re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

import numpy as np

from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from model import VAE, customLoss
import utils 

from datasets import BaselineDataset

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-config',           
                        default=None,          
                        type=str,     
                        help='path to config file')

    opts = parser.parse_args()

    args = OmegaConf.load(opts.config)

    if not torch.cuda.is_available():
        args.device = 'cpu'
    else:
        args.device = 'cuda'

    device = torch.device(args.device)

    Datasets = BaselineDataset(dataset=args.dataset, root_dir=args.data_dir,
                               model_name=args.vae.model_name, mode='train')
    dataset = Datasets.get_dataset()

    model_save_dir = os.path.join(args.model_dir, args.dataset)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    model_name = 'n_epoch_{}__btchsz_{}__hsize_{}_{}__ldim_{}__e_layers_{}' + \
                 '__d_layers_{}__opt_{}__lr_{}__wdecay_{}__eps_{}__momentum_{}__' + \
                 'n_tracin_lyrs_{}__stepszCP_{}__m_{}__l_{}'
    args.model_name = model_name.format(args.vae.n_epochs,
                                        args.vae.batch_size,
                                        args.vae.hidden_sizes[0],
                                        args.vae.hidden_sizes[1],
                                        args.vae.latent_dim, 
                                        args.vae.n_encoder_layers,
                                        args.vae.n_decoder_layers,
                                        args.vae.optimizer,
                                        args.vae.lr,
                                        args.vae.weight_decay,
                                        args.vae.eps,
                                        args.vae.momentum,
                                        args.vae.tracin_layers,
                                        args.vae.step_size_CP,
                                        args.vae.n_random_train_sample,
                                        args.vae.reconstruct_num)


    model_save_dir = os.path.join(model_save_dir, args.model_name)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)

    args.model_dir = model_save_dir

    for iter in range(args.n_iters):

        seed = args.seed + iter

        # check whether training is necessary
        name_list = glob.glob(os.path.join(args.model_dir,
                                           "model_epoch_*_seed_{}.pth".format(seed)))

        if len(name_list) > 0:
            epoch_list = []
            for name in name_list:
                s = re.findall(r'\d+', os.path.basename(name))[0]
                epoch_list.append(int(s))

            epoch_list.sort()
            epoch_st = epoch_list[-1]
            ## if the last checkpoint is equal to the number of epoch, 
            ## skip training for seed
            if epoch_st >= args.vae.n_epochs:
                print('CP already exist for seed {}'.format(seed))
                continue


        # Set seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        dataloader = DataLoader(dataset, batch_size=args.vae.batch_size, num_workers=0,
                                shuffle=True, drop_last=True)

        model = VAE(D_in=dataset.num_features(), hidden_sizes=args.vae.hidden_sizes, 
                            latent_dim=args.vae.latent_dim, n_encoder_layers=args.vae.n_encoder_layers,
                            n_decoder_layers=args.vae.n_decoder_layers)
        model.to(device)

        optimizer = utils.get_optimizer(model.parameters(), args.vae.optimizer, args.vae)
        scaler = GradScaler(enabled=args.fp16_precision)

        loss_mse = customLoss()
        epoch_saver = args.vae.step_size_CP

        while model.epoch < args.vae.n_epochs:

            start_time = datetime.now()
            model.epoch += 1

            model.train()
            outputs = []

            for iteration, batchs in enumerate(dataloader, 1):
                
                total_iter = (model.epoch - 1) * len(dataloader) + iteration
                
                batchs = batchs.to(device)

                Loss = 0

                ### clear gradient
                optimizer.zero_grad()
                with autocast(enabled=args.fp16_precision):

                    if args.fp16_precision:
                        recon_batch, mu, logvar = model(batchs.half())

                        Loss += loss_mse(recon_batch, batchs.half(), mu, logvar)
                    else:
                        recon_batch, mu, logvar = model(batchs.float())

                        Loss += loss_mse(recon_batch, batchs.float(), mu, logvar)
                    
                scaler.scale(Loss).backward()
                scaler.step(optimizer)
                scaler.update()

            end_time = datetime.now()
            print("Elapsed Time for epoch {}/{}: {}".format(model.epoch, args.vae.n_epochs,
                                                            end_time - start_time),
                  "Loss: {}".format(Loss/len(dataloader)))

            if model.epoch==epoch_saver:
                utils.save_model(model, optimizer, None, args, seed)
                epoch_saver += args.vae.step_size_CP
            #end epoch

           

