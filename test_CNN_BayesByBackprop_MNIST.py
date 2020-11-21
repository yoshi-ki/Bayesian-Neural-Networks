from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.conv_model import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Convolutional Neural Net on MNIST with Variational Inference')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=200,
                    help='How many epochs to train. Default: 200.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=10,
                    help='How many MC samples to take when approximating the ELBO. Default: 10.')
parser.add_argument('--models_dir', type=str, nargs='?', action='store', default='BBP_conv_models',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_conv_models\'.')
parser.add_argument('--results_dir', type=str, nargs='?', action='store', default='BBP_conv_results',
                    help='Where to save learnt training plots. Default: \'BBP_conv_results\'.')
args = parser.parse_args()




# ------------------------------------------------------------------------------------------------------
# train config
NTrainPointsMNIST = 60000
batch_size = 100
nb_epochs = args.epochs
log_interval = 1


# ------------------------------------------------------------------------------------------------------
# dataset
cprint('c', '\nData:')

# load data

# data augmentation
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

use_cuda = torch.cuda.is_available()

trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform_train)
valset = datasets.MNIST(root='../data', train=False, download=True, transform=transform_test)

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


net = BBP_Bayes_Conv_Net(lr=lr, channels_in=1, side_in = 28, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsMNIST / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))
net.load('./BBP_conv_models/theta_best.dat')


## ---------------------------------------------------------------------------------------------------------------------
# test
epoch = 0
cprint('c', '\nTest:')

print('  init cost variables:')

cost_dev = 0
err_dev = 0


ELBO_samples = nsamples

# ---- normal evaluation
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

print('sample evaluation')
cprint('g', '    Jdev = %f, err = %f\n' % (cost_dev.long()/nb_samples, err_dev.long()/nb_samples))


