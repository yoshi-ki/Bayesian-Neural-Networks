from src.priors import *
from src.base_net import *
import torch.nn.functional as F
import torch.nn as nn
import copy


def sample_weights(W_mu, b_mu, W_p, b_p):
    """Quick method for sampling weights and exporting weights"""
    eps_W = W_mu.data.new(W_mu.size()).normal_()
    # sample parameters
    std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
    W = W_mu + 1 * std_w * eps_W

    if b_mu is not None:
        std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)
        eps_b = b_mu.data.new(b_mu.size()).normal_()
        b = b_mu + 1 * std_b * eps_b
    else:
        b = None

    return W, b

def KL_gaussian(p_mu, p_sigma, q_mu, q_sigma):
    # compute KL(p||q)
    return (torch.log(q_sigma/p_sigma) + (p_sigma ** 2 + (p_mu - q_mu) ** 2) / (2 * q_sigma ** 2) - 1/2).sum()

def inverse_softplus(x, beta = 1, threshold=20):
    if x >= threshold :
      return x
    else :
      return np.log(np.exp(beta * x) - 1) / beta




class BayesLinear_Normalq(nn.Module):
    """Linear Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, n_in, n_out, prior_class):
        super(BayesLinear_Normalq, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.zeros(self.n_in, self.n_out))
        self.W_p = nn.Parameter(torch.full_like(torch.zeros(self.n_in, self.n_out), inverse_softplus(np.sqrt(2/self.n_in)) ))

        self.b_mu = nn.Parameter(torch.zeros(self.n_out))
        self.b_p = nn.Parameter(torch.full_like(torch.zeros(self.n_out), inverse_softplus(1e-6)) )

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False):
        #         print(self.training)

        if not self.training and not sample:  # When training return MLE of w for quick validation
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        else:

            # Tensor.new()  Constructs a new tensor of the same data type as self tensor.
            # the same random sample is used for every element in the minibatch
            eps_W = Variable(self.W_mu.data.new(self.W_mu.size()).normal_())
            eps_b = Variable(self.b_mu.data.new(self.b_mu.size()).normal_())

            # sample parameters
            std_w = 1e-6 + F.softplus(self.W_p, beta=1, threshold=20)
            std_b = 1e-6 + F.softplus(self.b_p, beta=1, threshold=20)


            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b

            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

            return output, KL_gaussian(self.W_mu, std_w, self.prior.mu, self.prior.sigma), KL_gaussian(self.b_mu, std_b, self.prior.mu, self.prior.sigma)



class bayes_Sin_Net(nn.Module):
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, input_dim, in_channels, output_dim, prior_instance):
        super(bayes_Sin_Net, self).__init__()

        self.prior_instance = prior_instance

        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.bfc1 = BayesLinear_Normalq(1,512, self.prior_instance)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.bfc2 = BayesLinear_Normalq(512,1024, self.prior_instance)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.bfc3 = BayesLinear_Normalq(1024,512, self.prior_instance)
        self.batchnorm3 = nn.BatchNorm1d(512)
        self.bfc4 = BayesLinear_Normalq(512, 1, self.prior_instance)

        self.act = nn.ReLU()

    def forward(self, x, sample=False):
        tklw = 0
        tklb = 0

        # print(x,x.shape)
        x = x.view(x.shape[0],-1)
        # -----------------
        x, klw, klb = self.bfc1(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.batchnorm1(x)
        x = self.act(x)
        x, klw, klb = self.bfc2(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.batchnorm2(x)
        x = self.act(x)
        x, klw, klb = self.bfc3(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.batchnorm3(x)
        x = self.act(x)
        y, klw, klb = self.bfc4(x, sample)
        # print(y)
        tklw = tklw + klw
        tklb = tklb + klb

        return y, tklw, tklb

    def sample_predict(self, x, Nsamples):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        tklw_vec = np.zeros(Nsamples)
        tklb_vec = np.zeros(Nsamples)

        for i in range(Nsamples):
            y, tklw, tklb = self.forward(x, sample=True)
            predictions[i] = y
            tklw_vec[i] = tklw
            tklb_vec[i] = tklb

        return predictions, tklw_vec, tklb_vec

class BBP_Bayes_Sin_Net(BaseNet):
    """Full network wrapper for Bayes By Backprop nets with methods for training, prediction and weight prunning"""
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, Nbatches=0, prior_instance=laplace_prior(mu=0, b=0.1)):
        super(BBP_Bayes_Sin_Net, self).__init__()
        cprint('y', ' Creating Net!! ')
        self.lr = lr
        self.schedule = None  # [] #[50,200,400,600]
        self.cuda = cuda
        self.channels_in = channels_in
        self.classes = classes
        self.batch_size = batch_size
        self.Nbatches = Nbatches
        self.prior_instance = prior_instance
        self.side_in = side_in
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = bayes_Sin_Net(input_dim=self.channels_in * self.side_in * self.side_in,in_channels=self.channels_in, output_dim=self.classes, prior_instance=self.prior_instance)
        if self.cuda:
            self.model.cuda()
        #             cudnn.benchmark = True

        print('    Total params: %.2fM' % (self.get_nb_parameters() / 1000000.0))

    def create_opt(self):
        #         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08,
        #                                           weight_decay=0)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0)

    #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
    #         self.sched = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=10, last_epoch=-1)

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        self.optimizer.zero_grad()
        loss = 0
        for i in range(samples):
            # print("fit")
            out, tklw, tklb = self.model(x)
            y = y.reshape(out.shape)
            # mlpdw = (out-y)*(out-y)/ self.Nbatches
            mlpdw = nn.functional.mse_loss(out, y)
            Edkl = (tklw + tklb) / self.Nbatches
            loss = loss + Edkl/1000 + mlpdw
        loss = loss / samples
        loss.backward()
        # for i, param in enumerate(self.model.parameters()):
        #     param_name = list(self.model.state_dict().keys())[i]
        #     print(param_name)
        #     print(param.grad.max())
        self.optimizer.step()


        # # 1sampleにつき1回学習させる方法
        # for i in range(samples):
        #     self.optimizer.zero_grad()
        #     loss = 0
        #     out, tklw, tklb = self.model(x)
        #     y = y.reshape(out.shape)
        #     mlpdw = nn.functional.mse_loss(out, y) / self.Nbatches
        #     Edkl = (tklw + tklb) / self.Nbatches
        #     loss = loss + Edkl /10000 + mlpdw
        #     loss.backward()
        #     # for i, param in enumerate(self.model.parameters()):
        #     #     param_name = list(self.model.state_dict().keys())[i]
        #     #     print(param_name)
        #     #     print(param.grad.max())
        #     self.optimizer.step()


        # out: (batch_size, out_channels, out_caps_dims)

        return Edkl.data, mlpdw.data

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y), cuda=self.cuda)

        # print("eval")
        # print(x)
        out, _, _ = self.model(x)
        # print(out)
        y = y.reshape(out.shape)
        loss = nn.functional.mse_loss(out, y)
        loss = loss

        # probs = F.softmax(out, dim=1).data.cpu()

        # pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        # err = pred.ne(y.data).sum()

        # return loss.data, err, probs
        return loss.data, out

    def inference(self, x, train=False):
        x = x.reshape(1)
        x = to_variable(var=(x), cuda=self.cuda)
        x = x[0].reshape(1)
        out, _, _ = self.model(x)
        return out.data

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        """Prediction, only returining result with weights marginalised"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model.sample_predict(x, Nsamples)

        if logits:
            mean_out = out.mean(dim=0, keepdim=False)
            loss = F.cross_entropy(mean_out, y, reduction='sum')
            probs = F.softmax(mean_out, dim=1).data.cpu()

        else:
            mean_out = F.softmax(out, dim=2).mean(dim=0, keepdim=False)
            probs = mean_out.data.cpu()

            log_mean_probs_out = torch.log(mean_out)
            loss = F.nll_loss(log_mean_probs_out, y, reduction='sum')

        pred = mean_out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def all_sample_eval(self, x, y, Nsamples):
        """Returns predictions for each MC sample"""
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model.sample_predict(x, Nsamples)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out

    def get_weight_samples(self, Nsamples=10):
        state_dict = self.model.state_dict()
        weight_vec = []

        for i in range(Nsamples):
            previous_layer_name = ''
            for key in state_dict.keys():
                layer_name = key.split('.')[0]
                if layer_name != previous_layer_name:
                    previous_layer_name = layer_name

                    W_mu = state_dict[layer_name + '.W_mu'].data
                    W_p = state_dict[layer_name + '.W_p'].data

                    #                 b_mu = state_dict[layer_name+'.b_mu'].cpu().data
                    #                 b_p = state_dict[layer_name+'.b_p'].cpu().data

                    W, b = sample_weights(W_mu=W_mu, b_mu=None, W_p=W_p, b_p=None)

                    for weight in W.cpu().view(-1):
                        weight_vec.append(weight)

        return np.array(weight_vec)

    def get_weight_SNR(self, thresh=None):
        state_dict = self.model.state_dict()
        weight_SNR_vec = []

        if thresh is not None:
            mask_dict = {}

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                W_mu = state_dict[layer_name + '.W_mu'].data
                W_p = state_dict[layer_name + '.W_p'].data
                sig_W = 1e-6 + F.softplus(W_p, beta=1, threshold=20)

                b_mu = state_dict[layer_name + '.b_mu'].data
                b_p = state_dict[layer_name + '.b_p'].data
                sig_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                W_snr = (torch.abs(W_mu) / sig_W)
                b_snr = (torch.abs(b_mu) / sig_b)

                if thresh is not None:
                    mask_dict[layer_name + '.W'] = W_snr > thresh
                    mask_dict[layer_name + '.b'] = b_snr > thresh

                else:

                    for weight_SNR in W_snr.cpu().view(-1):
                        weight_SNR_vec.append(weight_SNR)

                    for weight_SNR in b_snr.cpu().view(-1):
                        weight_SNR_vec.append(weight_SNR)

        if thresh is not None:
            return mask_dict
        else:
            return np.array(weight_SNR_vec)

    def get_weight_KLD(self, Nsamples=20, thresh=None):
        state_dict = self.model.state_dict()
        weight_KLD_vec = []

        if thresh is not None:
            mask_dict = {}

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                W_mu = state_dict[layer_name + '.W_mu'].data
                W_p = state_dict[layer_name + '.W_p'].data
                b_mu = state_dict[layer_name + '.b_mu'].data
                b_p = state_dict[layer_name + '.b_p'].data

                std_w = 1e-6 + F.softplus(W_p, beta=1, threshold=20)
                std_b = 1e-6 + F.softplus(b_p, beta=1, threshold=20)

                KL_W = W_mu.new(W_mu.size()).zero_()
                KL_b = b_mu.new(b_mu.size()).zero_()
                for i in range(Nsamples):
                    W, b = sample_weights(W_mu=W_mu, b_mu=b_mu, W_p=W_p, b_p=b_p)
                    # Note that this will currently not work with slab and spike prior
                    KL_W += isotropic_gauss_loglike(W, W_mu, std_w,
                                                    do_sum=False) - self.model.prior_instance.loglike(W,
                                                                                                      do_sum=False)
                    KL_b += isotropic_gauss_loglike(b, b_mu, std_b,
                                                    do_sum=False) - self.model.prior_instance.loglike(b,
                                                                                                      do_sum=False)

                KL_W /= Nsamples
                KL_b /= Nsamples

                if thresh is not None:
                    mask_dict[layer_name + '.W'] = KL_W > thresh
                    mask_dict[layer_name + '.b'] = KL_b > thresh

                else:

                    for weight_KLD in KL_W.cpu().view(-1):
                        weight_KLD_vec.append(weight_KLD)

                    for weight_KLD in KL_b.cpu().view(-1):
                        weight_KLD_vec.append(weight_KLD)

        if thresh is not None:
            return mask_dict
        else:
            return np.array(weight_KLD_vec)

    def mask_model(self, Nsamples=0, thresh=0):
        '''
        Nsamples is used to select SNR (0) or KLD (>0) based masking
        '''
        original_state_dict = copy.deepcopy(self.model.state_dict())
        state_dict = self.model.state_dict()

        if Nsamples == 0:
            mask_dict = self.get_weight_SNR(thresh=thresh)
        else:
            mask_dict = self.get_weight_KLD(Nsamples=Nsamples, thresh=thresh)

        n_unmasked = 0

        previous_layer_name = ''
        for key in state_dict.keys():
            layer_name = key.split('.')[0]
            if layer_name != previous_layer_name:
                previous_layer_name = layer_name

                state_dict[layer_name + '.W_mu'][1 - mask_dict[layer_name + '.W']] = 0
                state_dict[layer_name + '.W_p'][1 - mask_dict[layer_name + '.W']] = -1000

                state_dict[layer_name + '.b_mu'][1 - mask_dict[layer_name + '.b']] = 0
                state_dict[layer_name + '.b_p'][1 - mask_dict[layer_name + '.b']] = -1000

                n_unmasked += mask_dict[layer_name + '.W'].sum()
                n_unmasked += mask_dict[layer_name + '.b'].sum()

        return original_state_dict, n_unmasked