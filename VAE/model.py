import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.init as init

class VAE(nn.Module):
    def __init__(self, D_in:int, hidden_sizes:list=[50,12], latent_dim:int=3,
                 n_encoder_layers:int=6, n_decoder_layers:int=6):

        super(VAE,self).__init__()
        self.epoch = 0
        self.encoder = []
        self.decoder = []

        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers

        self.s = None
        self.v = None
        self.mu = None
        
        self.encoder.append(self.__make_layer__(D_in, hidden_sizes[0]))
        self.encoder.append(self.__make_layer__(hidden_sizes[0],hidden_sizes[1]))
        for i in range(self.n_encoder_layers-2):
            self.encoder.append(self.__make_layer__(hidden_sizes[1],hidden_sizes[1]))
        self.encoder = nn.Sequential(*self.encoder)
        
        # Latent vectors mu and sigma
        self.fc1 = nn.Linear(hidden_sizes[1], latent_dim)
        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)
        self.fc21 = nn.Linear(latent_dim, latent_dim)
        self.fc22 = nn.Linear(latent_dim, latent_dim)

        # Sampling vector
        self.fc3 = nn.Linear(latent_dim, latent_dim)
        self.fc_bn3 = nn.BatchNorm1d(latent_dim)
        self.fc4 = nn.Linear(latent_dim, hidden_sizes[1])
        self.fc_bn4 = nn.BatchNorm1d(hidden_sizes[1])

        for i in range(self.n_decoder_layers-2):
            self.decoder.append(self.__make_layer__(hidden_sizes[1],hidden_sizes[1]))
        self.decoder.append(self.__make_layer__(hidden_sizes[1],hidden_sizes[0]))
        self.decoder.append(self.__make_layer__(hidden_sizes[0], D_in))
        self.decoder = nn.Sequential(*self.decoder)
        
        self.relu = nn.ReLU()

        self.weight_init()

    
    def weight_init(self):
        for block in self._modules:
            if not isinstance(self._modules[block], nn.Sequential):
                kaiming_init(block)
            else:
                for m in self._modules[block]:
                    kaiming_init(m)
            
    def encode(self, x):

        x = self.encoder(x)

        fc1 = self.relu(self.bn1(self.fc1(x)))

        r1 = self.fc21(fc1)
        r2 = self.fc22(fc1)
        return r1, r2
    
    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu
        
    def decode(self, z):
        fc3 = self.relu(self.fc_bn3(self.fc3(z)))
        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))

        out = self.decoder(fc4)
        return out
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __make_layer__(self, in_channel, out_channel, bias=True,):   
        
        layers = []
        layers.append(nn.Linear(in_channel, out_channel, bias=bias))
        layers.append(nn.BatchNorm1d(out_channel))

        return nn.Sequential(*layers)


def kaiming_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)

class customLoss(nn.Module):
    def __init__(self, reduction:str="sum", verbose:bool=False):
        super(customLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.verbose = verbose
    
    def forward(self, x_recon, x, mu, logvar):

        loss = self.mse_loss(x_recon, x)
        
        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        if not self.verbose:
            return loss + loss_KLD 
        else:
            loss, loss_KLD