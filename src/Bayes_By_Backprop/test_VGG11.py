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
        # self.W_mu = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-0.1, 0.1))
        self.W_p = nn.Parameter(torch.full_like(torch.zeros(self.n_in, self.n_out), inverse_softplus(np.sqrt(2/self.n_in)) ))
        # self.W_p = nn.Parameter(torch.Tensor(self.n_in, self.n_out).uniform_(-3, -2))

        self.b_mu = nn.Parameter(torch.zeros(self.n_out))
        # self.b_mu = nn.Parameter(torch.Tensor(self.n_out).uniform_(-0.1, 0.1))
        # self.b_mu = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.n_out)))
        self.b_p = nn.Parameter(torch.full_like(torch.zeros(self.n_out), inverse_softplus(1e-6)) )
        # self.b_p = nn.Parameter(torch.Tensor(self.n_out).uniform_(-3, -2))
        # self.b_p = nn.Parameter(nn.init.xavier_normal_(torch.empty(self.n_out)))

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

            # # inference only part : randomly mask the weight value's std
            # thre = 0.8
            # mask = torch.nn.init.uniform_(torch.zeros(std_w.size()))
            # mask = torch.where(mask < thre, torch.ones(mask.size()), torch.zeros(mask.size()))
            # mask = mask.to(device='cuda')
            # std_w = std_w * mask

            W = self.W_mu + 1 * std_w * eps_W
            b = self.b_mu + 1 * std_b * eps_b
            # W = self.W_mu
            # b = self.b_mu

            output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)

            return output, KL_gaussian(self.W_mu, std_w, self.prior.mu, self.prior.sigma), KL_gaussian(self.b_mu, std_b, self.prior.mu, self.prior.sigma)

class BayesConv_Normalq(nn.Module):
    """Conv Layer where weights are sampled from a fully factorised Normal with learnable parameters. The likelihood
     of the weight samples under the prior and the approximate posterior are returned with each forward pass in order
     to estimate the KL term in the ELBO.
    """
    def __init__(self, in_channels, out_channels, kernel_size, prior_class, stride=1, padding=0):
        super(BayesConv_Normalq, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.prior = prior_class

        # Learnable parameters -> Initialisation is set empirically.
        self.W_mu = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        self.W_p = nn.Parameter(torch.full_like(torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size), inverse_softplus(np.sqrt(2/(self.in_channels*self.kernel_size*self.kernel_size))) ))

        self.b_mu = nn.Parameter(torch.zeros(self.out_channels))

        self.b_p = nn.Parameter(torch.full_like(torch.zeros(self.out_channels), inverse_softplus(1e-6)) )

        self.lpw = 0
        self.lqw = 0

    def forward(self, X, sample=False, act_drop=False, first_layer=False):
        #         print(self.training)

        if not self.training and not sample:  # When training return MLE of w for quick validation
            # output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            output = F.conv2d(X, self.W_mu, bias = self.b_mu, padding = self.padding)
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

            if not(act_drop):
              output = F.conv2d(X.to(device='cuda'), W, bias = b, padding=self.padding)
            else :
              if(first_layer):
                alpha = 2

                # # percent that satisfies the condition
                # cond_num = torch.where(torch.abs(X)<alpha,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
                # print( torch.sum(cond_num)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]) )

                X_1 = torch.where(torch.abs(X) < alpha, X, torch.zeros(X.size()).to(device='cuda'))
                X_2 = torch.where(torch.abs(X) < alpha,torch.zeros(X.size()).to(device='cuda'),X)
                output1 = F.conv2d(X_1, self.W_mu, bias = b, padding=self.padding)
                output2 = F.conv2d(X_2, W, padding=self.padding)
                output = output1 + output2


              else:
                alpha = 0.1

                # # percent that satisfies the condition
                # cond_num = torch.where(torch.abs(X)<alpha,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
                # print( torch.sum(cond_num)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3]) )

                X_1 = torch.where(X < alpha, X, torch.zeros(X.size()).to(device='cuda'))
                X_2 = torch.where(X < alpha,torch.zeros(X.size()).to(device='cuda'),X)
                output1 = F.conv2d(X_1, self.W_mu, bias = b, padding=self.padding)
                output2 = F.conv2d(X_2, W, padding=self.padding)
                output = output1 + output2


            return output, KL_gaussian(self.W_mu, std_w, self.prior.mu, self.prior.sigma), KL_gaussian(self.b_mu, std_b, self.prior.mu, self.prior.sigma)





class bayes_VGG11(nn.Module):
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, input_dim, in_channels, output_dim, prior_instance, act_drop):
        super(bayes_VGG11, self).__init__()

        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        # prior_instance = spike_slab_2GMM(mu1=0, mu2=0, sigma1=0.135, sigma2=0.001, pi=0.5)
        # prior_instance = isotropic_gauss_prior(mu=0, sigma=0.1)
        self.prior_instance = prior_instance

        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = BayesConv_Normalq(self.in_channels,64,3,self.prior_instance,padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = BayesConv_Normalq(64,128,3,self.prior_instance,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = BayesConv_Normalq(128,256,3,self.prior_instance,padding=1)
        self.conv4 = BayesConv_Normalq(256,256,3,self.prior_instance,padding=1)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv5 = BayesConv_Normalq(256,512,3,self.prior_instance,padding=1)
        self.conv6 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv7 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.conv8 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.pool5 = nn.MaxPool2d(2,2)

        self.bfc1 = BayesLinear_Normalq(512,4096, self.prior_instance)
        self.bfc2 = BayesLinear_Normalq(4096,4096, self.prior_instance)
        self.bfc3 = BayesLinear_Normalq(4096, self.output_dim, self.prior_instance)

        self.count = 0

        # choose your non linearity
        # self.act = nn.Tanh()
        # self.act = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)
        # self.act = nn.ELU(inplace=True)
        # self.act = nn.SELU(inplace=True)

        self.act_drop = act_drop

    def forward(self, x, sample=False):
        tklw = 0
        tklb = 0

        # -----------------
        x1, klw, klb = self.conv1(x, sample, self.act_drop, first_layer=True)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x1)
        x = self.pool1(x)
        # -----------------
        x2, klw, klb = self.conv2(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x2)
        x = self.pool2(x)
        # -----------------
        x3, klw, klb = self.conv3(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x3)
        x4, klw, klb = self.conv4(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x4)
        x = self.pool3(x)
        # -----------------
        x5, klw, klb = self.conv5(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x5)
        x6, klw, klb = self.conv6(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x6)
        x = self.pool4(x)
        # -----------------
        x7, klw, klb = self.conv7(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x7)
        x8, klw, klb = self.conv8(x, sample, self.act_drop)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x8)
        x = self.pool5(x)
        # -----------------
        x = x.view(x.shape[0],-1)
        # -----------------
        x, klw, klb = self.bfc1(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x)
        x, klw, klb = self.bfc2(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb
        x = self.act(x)
        y, klw, klb = self.bfc3(x, sample)
        tklw = tklw + klw
        tklb = tklb + klb


        if(self.count == 22):
          self.conv_out = [x1,x2,x3,x4,x5,x6,x7,x8]
        self.count = self.count + 1

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

class BBP_Bayes_VGG11_Net(BaseNet):
    """Full network wrapper for Bayes By Backprop nets with methods for training, prediction and weight prunning"""
    eps = 1e-6

    def __init__(self, lr=1e-3, channels_in=3, side_in=28, cuda=True, classes=10, batch_size=128, Nbatches=0, prior_instance=laplace_prior(mu=0, b=0.1), act_drop=False):
        super(BBP_Bayes_VGG11_Net, self).__init__()
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
        self.act_drop = act_drop
        self.create_net()
        self.create_opt()
        self.epoch = 0

        self.test = False

    def create_net(self):
        torch.manual_seed(42)
        if self.cuda:
            torch.cuda.manual_seed(42)

        self.model = bayes_VGG11(input_dim=self.channels_in * self.side_in * self.side_in,in_channels=self.channels_in, output_dim=self.classes, prior_instance=self.prior_instance, act_drop=self.act_drop)
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
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        # for i in range(samples):
        #     self.optimizer.zero_grad()
        #     out, tklw, tklb = self.model(x)
        #     mlpdw = F.cross_entropy(out,y,reduction='sum')
        #     Edkl = (tklw + tklb) / self.Nbatches
        #     loss = Edkl + mlpdw
        #     loss.backward()
        #     self.optimizer.step()

        self.optimizer.zero_grad()
        loss = 0
        for i in range(samples):
            out, tklw, tklb = self.model(x)
            mlpdw = F.cross_entropy(out,y,reduction='sum')
            Edkl = (tklw + tklb) / self.Nbatches
            loss = loss + Edkl + mlpdw
        loss = loss / samples
        loss.backward()
        self.optimizer.step()

        # ---------    samples and train ----------- #
        # self.optimizer.zero_grad()
        # if samples == 1:
        #     out, tlqw, tlpw = self.model(x)
        #     mlpdw = F.cross_entropy(out, y, reduction='sum')
        #     Edkl = (tlqw - tlpw) / self.Nbatches

        # elif samples > 1:
        #     mlpdw_cum = 0
        #     Edkl_cum = 0

        #     for i in range(samples):
        #         out, tlqw, tlpw = self.model(x, sample=True)
        #         mlpdw_i = F.cross_entropy(out, y, reduction='sum')
        #         Edkl_i = (tlqw - tlpw) / self.Nbatches
        #         mlpdw_cum = mlpdw_cum + mlpdw_i
        #         Edkl_cum = Edkl_cum + Edkl_i

        #     mlpdw = mlpdw_cum / samples
        #     Edkl = Edkl_cum / samples

        # loss = Edkl + mlpdw
        # loss.backward()

        # #print(self.model.bfc1.W_mu.grad)

        # self.optimizer.step()
        # ---------    samples and train ----------- #


        # out: (batch_size, out_channels, out_caps_dims)
        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return Edkl.data, mlpdw.data, err

    def eval(self, x, y, train=False):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, _, _ = self.model(x)

        loss = F.cross_entropy(out, y, reduction='sum')

        probs = F.softmax(out, dim=1).data.cpu()

        pred = out.data.max(dim=1, keepdim=False)[1]  # get the index of the max log-probability
        err = pred.ne(y.data).sum()

        return loss.data, err, probs

    def sample_eval(self, x, y, Nsamples, logits=True, train=False):
        """Prediction, only returining result with weights marginalised"""

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
        # initialize seed if needed
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

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