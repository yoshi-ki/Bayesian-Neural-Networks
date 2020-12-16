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
parser.add_argument('--lr', type=float, nargs='?', action='store', default=0.005,
                    help='learning rate. Default: 0.005.')
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

train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.17,0.30,0.35,0.4,0.5,0.9,0.95])
train_y = torch.sin(4*train_x) * torch.cos(14*train_x)

#TODO: 学習サイズを変える時はここも変えなければならないことに注意
net = BBP_Bayes_Sin_Net(lr=lr, channels_in=3, side_in = 32, cuda=use_cuda, classes=10, batch_size=batch_size, Nbatches=1, prior_instance=isotropic_gauss_prior(mu=0, sigma=args.prior_sig))


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
    # データ一点一点で学習する方法
    train_x = train_x.data.new(train_x.size()).uniform_(-1,1)
    train_y = torch.sin(4*train_x) * torch.cos(14*train_x)
    for j in range(len(train_x)):
        x = torch.Tensor([train_x[j]])
        y = torch.Tensor([train_y[j]])
        # y = y + 0.1 * y.data.new(y.size()).normal_()
        cost_dkl, err = net.fit(x, y, samples=ELBO_samples)
        kl_cost_train[i] += cost_dkl
        err_train[i] += err
        nb_samples += len(x)

    # データを一気に与えて学習する方法
    # train_x = train_x.data.new(train_x.size()).uniform_(-1,1)
    # train_y = torch.sin(4*train_x) * torch.cos(14*train_x)
    # x = torch.Tensor(train_x)
    # y = torch.Tensor(train_y)
    # x = train_x
    # y = train_y
    # y = y + 0.1 * y.data.new(y.size()).normal_()
    # cost_dkl, err = net.fit(x, y, samples=ELBO_samples)
    # err_train[i] += err
    # kl_cost_train[i] += cost_dkl
    # nb_samples += len(x)


    # kl_cost_train[i] /= nb_samples  # Normalise by number of samples in order to get comparable number to the -log like
    # err_train[i] /= nb_samples

    toc = time.time()
    net.epoch = i
    # ---- print
    print("it %d/%d, Jtr_KL = %f, err = %f, " % (
    i, nb_epochs, kl_cost_train[i], err_train[i]), end="")
    cprint('r', '   time: %f seconds\n' % (toc - tic))

    # ---- dev
    if i % nb_its_dev == 0:
        net.set_mode_train(False)
        nb_samples = 0
        out = np.zeros(20)
        # train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.17,0.30,0.35,0.4,0.5,0.9,0.95])
        train_x = torch.Tensor([-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95])
        train_y = torch.sin(4*train_x) * torch.cos(14*train_x)
        for j in range(len(train_x)):
            x = torch.Tensor([train_x[j]])
            y = torch.Tensor([train_y[j]])
            cost, out[j] = net.eval(x, y)  # This takes the expected weights to save time, not proper inference

            cost_dev[i] += cost
            # err_dev[i] += err
            nb_samples += len(x)
        # train_x = torch.Tensor([-0.55,-0.4,-0.38,-0.35,-0.3,-0.25,-0.21,-0.15,-0.05,-0.03,0.0,0.05,0.06,0.17,0.30,0.35,0.4,0.5,0.9,0.95])
        # train_y = torch.sin(4*train_x) * torch.cos(14*train_x)
        # x = train_x
        # y = train_y
        # cost, out = net.eval(x,y)
        # cost_dev[i] = cost

        # cost_dev[i] /= nb_samples
        # err_dev[i] /= nb_samples

        cprint('g', '    square root error = %f' % cost_dev[i])

        if cost_dev[i] < best_err:
            best_err = cost_dev[i]
            print(out,train_y)
            cprint('b', 'best test error')
            net.save(models_dir + '/theta_best.dat')

    if i % 100 == 0:
      print(out)
      textsize = 15
      marker = 5

      plt.figure(dpi=100)
      fig, ax1 = plt.subplots()
      ax1.plot(err_train[:i], 'r--')
      ax1.plot(range(0, i, nb_its_dev), cost_dev[:i], 'b-')
      ax1.set_ylabel('Squared error')
      plt.xlabel('epoch')
      plt.grid(b=True, which='major', color='k', linestyle='-')
      plt.grid(b=True, which='minor', color='k', linestyle='--')
      lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
      ax = plt.gca()
      plt.title('regression costs')
      for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                  ax.get_xticklabels() + ax.get_yticklabels()):
          item.set_fontsize(textsize)
          item.set_weight('normal')
      plt.savefig(results_dir + '/pred_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

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

# textsize = 15
# marker = 5

# plt.figure(dpi=100)
# fig, ax1 = plt.subplots()
# ax1.plot(pred_cost_train, 'r--')
# ax1.plot(range(0, nb_epochs, nb_its_dev), cost_dev[::nb_its_dev], 'b-')
# ax1.set_ylabel('Squared error')
# plt.xlabel('epoch')
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='k', linestyle='--')
# lgd = plt.legend(['train error', 'test error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
# ax = plt.gca()
# plt.title('regression costs')
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(textsize)
#     item.set_weight('normal')
# plt.savefig(results_dir + '/pred_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

# plt.figure()
# fig, ax1 = plt.subplots()
# ax1.plot(kl_cost_train, 'r')
# ax1.set_ylabel('nats?')
# plt.xlabel('epoch')
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='k', linestyle='--')
# ax = plt.gca()
# plt.title('DKL (per sample)')
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(textsize)
#     item.set_weight('normal')
# plt.savefig(results_dir + '/KL_cost.png', bbox_extra_artists=(lgd,), bbox_inches='tight')

# plt.figure(dpi=100)
# fig2, ax2 = plt.subplots()
# ax2.set_ylabel('% error')
# ax2.semilogy(range(0, nb_epochs, nb_its_dev), 100 * err_dev[::nb_its_dev], 'b-')
# ax2.semilogy(100 * err_train, 'r--')
# plt.xlabel('epoch')
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='k', linestyle='--')
# ax2.get_yaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
# ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# lgd = plt.legend(['test error', 'train error'], markerscale=marker, prop={'size': textsize, 'weight': 'normal'})
# ax = plt.gca()
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#              ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(textsize)
#     item.set_weight('normal')
# plt.savefig(results_dir + '/err.png', bbox_extra_artists=(lgd,), box_inches='tight')
