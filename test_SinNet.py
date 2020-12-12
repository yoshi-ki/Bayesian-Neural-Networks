from __future__ import division, print_function
import time
import torch.utils.data
from torchvision import transforms, datasets
import argparse
import matplotlib
from src.Bayes_By_Backprop.SinNet import *
from src.Bayes_By_Backprop_Local_Reparametrization.model import *

matplotlib.use('Agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Train Bayesian Convolutional Neural Net on MNIST with Variational Inference')
parser.add_argument('models_dir', type=str, nargs='?', action='store', default='BBP_SinNet_models',
                    help='Where to save learnt weights and train vectors. Default: \'BBP_SinNet_models\'.')
parser.add_argument('results_dir', type=str, nargs='?', action='store', default='BBP_SinNet_results',
                    help='Where to save learnt training plots. Default: \'BBP_SinNet_results\'.')
parser.add_argument('--model', type=str, nargs='?', action='store', default='Local_Reparam',
                    help='Model to run. Options are \'Gaussian_prior\', \'Laplace_prior\', \'GMM_prior\','
                         ' \'Local_Reparam\'. Default: \'Local_Reparam\'.')
parser.add_argument('--prior_sig', type=float, nargs='?', action='store', default=0.1,
                    help='Standard deviation of prior. Default: 0.1.')
parser.add_argument('--epochs', type=int, nargs='?', action='store', default=200,
                    help='How many epochs to train. Default: 200.')
parser.add_argument('--lr', type=float, nargs='?', action='store', default=1e-3,
                    help='learning rate. Default: 1e-3.')
parser.add_argument('--n_samples', type=float, nargs='?', action='store', default=3,
                    help='How many MC samples to take when approximating the ELBO. Default: 3.')
args = parser.parse_args()



# Where to save models weights
models_dir = args.models_dir
# Where to save plots and error, accuracy vectors
results_dir = args.results_dir

mkdir(models_dir)
mkdir(results_dir)
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
    # transforms.Normalize(mean=(0,0,0), std=(1,1,1))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,0.1307,0.1307), std=(0.3081,0.3801,0.3801))
    # transforms.Normalize(mean=(0,0,0), std=(1,1,1))
])

use_cuda = torch.cuda.is_available()

## ---------------------------------------------------------------------------------------------------------------------
# net dims
cprint('c', '\nNetwork:')

lr = args.lr
nsamples = int(args.n_samples)  # How many samples to estimate ELBO with at each iteration
########################################################################################


net = BBP_Bayes_Sin_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=(NTrainPointsCIFAR10 / batch_size), prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))

train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.17,0.30,0.35,0.4,0.5,0.9,0.95])
train_y = torch.sin(4*train_x) * torch.cos(14*train_x)

textsize = 15
marker = 5

plt.figure()
x = np.arange(-1,1,0.01)
y = np.sin(4*x)*np.cos(14*x)
plt.plot(x,y)
plt.scatter(train_x,train_y)
plt.savefig('THESIS_DATA/5-2/5-2.png')


## ---------------------------------------------------------------------------------------------------------------------
# train
epoch = 0
cprint('c', '\nTrain:')

print('  init cost variables:')
kl_cost_train = np.zeros(nb_epochs)
pred_cost_train = np.zeros(nb_epochs)
err_train = np.zeros(nb_epochs)

cost_dev = np.zeros(nb_epochs)
err_dev = np.zeros(nb_epochs)
best_err = np.inf

nb_its_dev = 1

tic0 = time.time()
for i in range(epoch, nb_epochs):
    # We draw more samples on the first epoch in order to ensure convergence
    if i == 0:
        ELBO_samples = nsamples
    else:
        ELBO_samples = nsamples

    net.set_mode_train(True)
    tic = time.time()
    nb_samples = 0
    for j in range(len(train_x)):
        x = torch.Tensor([train_x[j]])
        y = train_y[j]
        cost_dkl, cost_pred, err = net.fit(x, y, samples=ELBO_samples)
        err_train[j] += err
        kl_cost_train[j] += cost_dkl
        pred_cost_train[j] += cost_pred
        nb_samples += len(x)

    # for x, y in trainloader:
    #     cost_dkl, cost_pred, err = net.fit(x, y, samples=ELBO_samples)
    #     err_train[i] += err
    #     kl_cost_train[i] += cost_dkl
    #     pred_cost_train[i] += cost_pred
    #     nb_samples += len(x)

    kl_cost_train[i] /= nb_samples  # Normalise by number of samples in order to get comparable number to the -log like
    pred_cost_train[i] /= nb_samples
    err_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_KL = %f, Jtr_pred = %f, err = %f, " % (
    i, nb_epochs, kl_cost_train[i], pred_cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        for j in range(len(train_x)):
            x = torch.Tensor([train_x[j]])
            y = train_y[j]
            cost = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

            cost_dev[i] += cost
            # err_dev[i] += err
            nb_samples += len(x)

        cost_dev[i] /= nb_samples
        # err_dev[i] /= nb_samples

        cprint('g', '    square root error = %f' % cost_dev[i])

        if cost_dev[i] < best_err:
            best_err = cost_dev[i]
            cprint('b', 'best test error')
            net.save(models_dir + '/theta_best.dat')

toc0 = time.time()
runtime_per_it = (toc0 - tic0) / float(nb_epochs)
cprint('r', '   average time: %f seconds\n' % runtime_per_it)

net.save(models_dir + '/theta_last.dat')

## ---------------------------------------------------------------------------------------------------------------------
# results
cprint('c', '\nRESULTS:')
nb_parameters = net.get_nb_parameters()
best_cost_dev = np.min(cost_dev)
best_cost_train = np.min(pred_cost_train)
err_dev_min = err_dev[::nb_its_dev].min()

print('  cost_dev: %f (cost_train %f)' % (best_cost_dev, best_cost_train))
print('  nb_parameters: %d (%s)' % (nb_parameters, humansize(nb_parameters)))
print('  time_per_it: %fs\n' % (runtime_per_it))

## Save results for plots
# np.save('results/test_predictions.npy', test_predictions)
np.save(results_dir + '/KL_cost_train.npy', kl_cost_train)
np.save(results_dir + '/pred_cost_train.npy', pred_cost_train)
np.save(results_dir + '/cost_dev.npy', cost_dev)
np.save(results_dir + '/err_train.npy', err_train)


## ---------------------------------------------------------------------------------------------------------------------
# fig cost vs its

textsize = 15
marker = 5

plt.figure()
x = np.arange(-1,1,0.01)
y = np.sin(4*x)*np.cos(14*x)
plt.plot(x,y)
plt.scatter(train_x,train_y)
plt.savefig('THESIS_DATA/5-2/5-2.png')