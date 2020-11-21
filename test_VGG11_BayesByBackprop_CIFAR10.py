from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.VGG11 import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Convolutional Neural Net on MNIST with Variational Inference')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Local_Reparam',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=200,
                    help='How many epochs to train. Default: 200.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=10,
                    help='How many MC samples to take when approximating the ELBO. Default: 10.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BBP_VGG11_models',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_VGG11_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='BBP_VGG11_results',
                    help='Where to save learnt training plots. Default: \'BBP_VGG11_results\'.')
args = parser.parse_args()



# ------------------------------------------------------------------------------------------------------
# train config
NTrainPointsCIFAR10 = 50000
batch_size = 32
nb_epochs = args.epochs
log_interval = 1


# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data

# data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3801,0.3801))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3801,0.3801))
])

use_cuda = torch.cuda.is_available()

trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
valset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_test)

if use_cuda:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=True,
                                            num_workers=3)

else:
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=False,
                                              num_workers=3)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                            num_workers=3)

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration
########################################################################################


net = BBP_Bayes_VGG11_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsCIFAR10 / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
net.load('./BBP_VGG11_models/theta_best.dat')



## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTest:')

print('  init cost variables:')

cost_dev = 0
err_dev = 0


# We draw more samples on the first epoch in order to ensure convergence

ELBO_samples = nsamples



# ---- dev
net.set_mode_train(False)
nb_samples = 0
for j, (x, y) in enumerate(valloader):
    cost, err, probs = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

    cost_dev += cost
    err_dev += err
    nb_samples += len(x)

cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev.long()/nb_samples, err_dev.long()/nb_samples))



cost_dev = 0
err_dev = 0

net.set_mode_train(False)
nb_samples = 0
for j, (x, y) in enumerate(valloader):
    cost, err, probs = net.sample_eval(x, y, ELBO_samples)  # This takes the expected weights to save time, not proper inference

    cost_dev += cost
    err_dev += err
    nb_samples += len(x)

cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev.long()/nb_samples, err_dev.long()/nb_samples))

