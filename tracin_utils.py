import torch
from torch import nn

from torch.autograd import grad

def grad_batch(batchs:torch.Tensor,
               num_layers:int, model:nn.Module,
               Loss:nn.Module=None, model_name:str='VAE',
               reconstruct_num:int=16, c:torch.Tensor=None, 
               ) -> list:
    
    grads_encoder=None
    grads_decoder=None

    model.eval()

    if model_name == 'VAE':

        loss = 0
        if reconstruct_num > 1:
            model.train()
            for _ in range(reconstruct_num):
                recon_batch, mu, logvar = model(batchs.float())
                loss += Loss(recon_batch, batchs.float(), mu, logvar)
            loss /= reconstruct_num
        else:
            model.eval()
            recon_batch, mu, logvar = model(batchs.float())
            loss = Loss(recon_batch, batchs.float(), mu, logvar)

        loss = torch.mean(loss, dim=1)

        encoder_params = [ p for p in model.encoder.parameters() if (p.requires_grad) ]
        decoder_params = [ p for p in model.decoder.parameters() if (p.requires_grad) ]

        if num_layers > len(decoder_params):
            grads_encoder = list( list( grad( x, encoder_params, create_graph=True) ) for x in loss )
            # removing batchnorm grad
            grads_encoder = [ [ p for p in loss_ if ('BatchNorm' not in str(p.grad_fn)) ] for loss_ in grads_encoder ]
            grads_encoder = [ p[num_layers - len(decoder_params):] for p in grads_encoder]

            grads_decoder = list( list( grad( x, decoder_params, create_graph=True) ) for x in loss )
            grads_decoder = [ [ p for p in loss_ if ('BatchNorm' not in str(p.grad_fn)) ] for loss_ in grads_decoder]

            # concatenating each sublist
            grads = [grad_e + grad_d for grad_e, grad_d in zip(grads_encoder,grads_decoder)]
        else :
            grads_decoder = list( list( grad( x, decoder_params, create_graph=True) ) for x in loss )
            grads_decoder = [ [ p for p in loss_ if ('BatchNorm' not in str(p.grad_fn)) ] for loss_ in grads_decoder]
            grads = [ p[-num_layers:] for p in grads_decoder]

    return grads