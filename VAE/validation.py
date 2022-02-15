import argparse, random, os, random, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import statsmodels.stats.api as sms

from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from model import VAE, customLoss
import utils
import tracin_utils
from datasets import BaselineDataset

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
    date_file = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    Datasets = BaselineDataset(dataset=args.dataset, root_dir=args.data_dir,
                               model_name=args.vae.model_name, mode='val')
    dataset = Datasets.get_dataset()

    args.model_name = 'n_epoch_{}__btchsz_{}__hsize_{}_{}__ldim_{}__e_layers_{}' + \
                      '__d_layers_{}__opt_{}__lr_{}__wdecay_{}__eps_{}__momentum_{}__' + \
                      'n_tracin_lyrs_{}__stepszCP_{}__m_{}__l_{}'
    args.model_name = args.model_name.format(args.vae.n_epochs,
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

    args.model_dir = os.path.join(args.model_dir, args.dataset, args.model_name)

    f1_scores_r_error = []
    precision_r_error = []
    recall_r_error= []
    ap_score_r_error = []
    auc_score_r_error = []

    f1_scores_influence = []
    precision_influence = []
    recall_influence = []
    ap_score_influence = []
    auc_score_influence = []

    f1_scores_aug_r_error = []
    precision_aug_r_error = []
    recall_aug_r_error = []
    ap_score_aug_r_error = []
    auc_score_aug_r_error = []

    for iter in range(args.n_iters):

        seed = args.seed + iter
        print('Seed: {}'.format(seed))

        # Set seed for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

        dataloader = DataLoader(dataset, batch_size=args.vae.batch_size, num_workers=0,
                                shuffle=True, drop_last=False)

        model = VAE(D_in=dataset.num_features(), hidden_sizes=args.vae.hidden_sizes, 
                        latent_dim=args.vae.latent_dim, n_encoder_layers=args.vae.n_encoder_layers,
                        n_decoder_layers=args.vae.n_decoder_layers)

        optimizer = utils.get_optimizer(model.parameters(), args.vae.optimizer, args.vae)

        epoch = 0
        loss_mse = customLoss(reduction='none')
        self_score = pd.DataFrame({'influence':np.zeros(len(dataset.val_labels)),
                                   'labels':dataset.val_labels})

        model, _, _ = utils.load_model(model, optimizer, None, args, args.vae.n_epochs, seed, device, verbose=False)
        model = model.to(device)
                
        indexs = []
        scores = []
    

        while epoch < args.vae.n_epochs:

            epoch += args.vae.step_size_CP
            model, _, _ = utils.load_model(model, optimizer, None, args, epoch, seed, device, verbose=False)

            model.to(device)
            print('Computing Influence for CP:{}/{}'.format(epoch, args.vae.n_epochs))

            ## selecting n_random_train_sample random normal samples in the training set to evaluate its impact on the val samples
            rand_train_batch = torch.tensor(dataset.train_data[np.random.choice(dataset.train_data.shape[0],
                                                                                    args.vae.n_random_train_sample,
                                                                                    replace=False),:])

            rand_train_batch = rand_train_batch.to(device)
            
            grad_x_train = tracin_utils.grad_batch(rand_train_batch, args.vae.tracin_layers, 
                                                   model, loss_mse, 
                                                   args.vae.model_name,
                                                   reconstruct_num=args.vae.reconstruct_num)
            
            grad_x_train = [torch.stack(x) for x in list(zip(*grad_x_train))]

            for batchs in tqdm(dataloader):

                batch, index, _ = batchs
                batch = batch.to(device)

                grad_x_val = tracin_utils.grad_batch(batch, args.vae.tracin_layers, 
                                                     model, loss_mse, 
                                                     args.vae.model_name,
                                                     reconstruct_num=args.vae.reconstruct_num)
                
                grad_x_val = [torch.stack(x) for x in list(zip(*grad_x_val))]

                grad_dot = [torch.mean(torch.mm(torch.flatten(val_grad, start_dim=1),
                                                torch.flatten(train_grad, start_dim=1).transpose(0,1)), dim=1) 
                            for val_grad, train_grad in zip(grad_x_val, grad_x_train)]

                grad_dot_product = torch.mean(torch.stack(grad_dot), dim=0).detach().cpu().numpy()

                # add gradient dot product to influences
                self_score.loc[index,"influence"] += grad_dot_product * args.vae.lr / args.vae.n_random_train_sample

        self_score['influence'] = (self_score['influence']-self_score['influence'].mean())/self_score['influence'].std()

        # Influence Score
        (f1_score, precision, recall,
        ap_score, auc_score, thresh) = utils.f_score(self_score['influence'],
                                                 self_score['labels'])
        y_pred = (self_score['influence'] >= thresh).astype(int)
        cm = confusion_matrix(self_score['labels'], y_pred)
        tn, fp, fn, tp = cm.ravel()

        print("Score:", "Influence Score only"
              "\n\tNumber of frauds in validation set:",
              dataset.val_labels.sum(),
              "\n\tShare of frauds in validation set:",
              dataset.val_labels.sum() / len(dataset.val_data),
              "\n\tF1-score: ", f1_score,
              "\n\tPrecision: ", precision,
              "\n\tRecall: ", recall,
              "\n\tAverage Precision: ", ap_score,
              "\n\tAUC Score: ", auc_score,
              "\n\tTrue Negative:", tn,
              "\n\tFalse Positive:", fp,
              "\n\tFalse Negative:", fn,
              "\n\tTrue Positive:", tp)

        f1_scores_influence.append(f1_score)
        precision_influence.append(precision)
        recall_influence.append(recall)
        ap_score_influence.append(ap_score)
        auc_score_influence.append(auc_score)

    stats_influence = np.array([f1_scores_influence, precision_influence, recall_influence, ap_score_influence, auc_score_influence]).T
    mean_stats_influence = np.mean(stats_influence, axis=0)
    std_stats_influence = np.std(stats_influence, axis=0)
    conf_ic_influence = np.array( [sms.DescrStatsW(x).tconfint_mean() for x in [f1_scores_influence, 
                                                                              precision_influence, 
                                                                              recall_influence, 
                                                                              ap_score_influence, 
                                                                              auc_score_influence] ] )


    np.savetxt(os.path.join(args.save_dir, 'stats_{}.txt'.format(date_file)), [args.model_name], fmt="%s")
    with open(os.path.join(args.save_dir, 'stats_{}.txt'.format(date_file)), 'ab') as file:
        file.write(b'Influence Score')
        file.write(b'\nF1 Precision Recall AUPRC AUROC\n')
        np.savetxt(file, stats_influence, delimiter=' ', fmt='%1.3f')

    np.savetxt(os.path.join(args.save_dir, 'stats_mean_std_{}.txt'.format(date_file)), [args.model_name], fmt="%s")
    with open(os.path.join(args.save_dir, 'stats_mean_std_{}.txt'.format(date_file)), 'ab') as file:
        file.write(b'Influence Score')
        file.write(b'\nF1 Precision Recall AUPRC AUROC\n')
        np.savetxt(file, mean_stats_influence[None], delimiter=' ', fmt='%1.3f')
        file.write(b'\nstd\n')
        np.savetxt(file, std_stats_influence[None], delimiter=' ', fmt='%1.3f')
        file.write(b'\nIC\n')
        np.savetxt(file, conf_ic_influence, delimiter=' ', fmt='%1.3f')

