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

        self.droprate_alpha_mean = 0
        self.droprate_alpha_count = 0
        self.droprate_beta_mean = 0
        self.droprate_beta_count = 0
        self.droprate_relu_mean = 0
        self.droprate_relu_count = 0

    def forward(self, X, sample=False, act_drop=False, first_layer=False, first_sample=False, given_alpha=0, given_beta=0):
        #         print(self.training)

        drop_rate_alpha = 0
        drop_rate_beta  = 0

        if not self.training and not sample:  # When training return MLE of w for quick validation
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            return output, 0, 0

        elif first_sample:
            output = torch.mm(X, self.W_mu) + self.b_mu.unsqueeze(0).expand(X.shape[0], -1)
            self.input_first = X
            # 小さすぎるものはのぞいておく
            self.output_first = torch.where(torch.abs(output) > 1e-6, output, torch.zeros(output.size()).to(device='cuda'))
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

            if (not(act_drop)):
              output = torch.mm(X, W) + b.unsqueeze(0).expand(X.shape[0], -1)  # (batch_size, n_output)
              cond_num = torch.where(torch.abs(X) < 1e-6,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
              drop_rate = torch.sum(cond_num)/(X.shape[0]*X.shape[1])
              self.droprate_relu_mean = (self.droprate_relu_mean * self.droprate_relu_count + drop_rate) / (self.droprate_relu_count + 1)
              self.droprate_relu_count = self.droprate_relu_count + 1
            else:
              if(first_layer):
                # first layerではalphaの値が変わるので
                beta = given_beta

                X_new = torch.where(torch.abs(X)<beta,torch.zeros(X.size()).to(device='cuda'), X)
                output2 = torch.mm(X_new,1 * std_w * eps)
                output = self.output_first + output2
                drop_rate_beta  = torch.sum(torch.abs(X)<beta)/(X.shape[0]*X.shape[1])
                self.droprate_beta_mean  = (self.droprate_beta_mean  * self.droprate_beta_count  + drop_rate_beta ) / (self.droprate_beta_count  + 1)
                self.droprate_beta_count = self.droprate_beta_count + 1
              else:
                alpha = given_alpha
                beta = given_beta
                self.x_for_save = X
                # cond_num = torch.where(torch.abs(X) < 1e-6,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
                # drop_rate = torch.sum(cond_num)/(X.shape[0]*X.shape[1])
                # print(drop_rate)

                X_1 = X - self.input_first
                self.diff = torch.abs(X_1)
                X_2 = X
                X_1 = torch.where(torch.abs(X_1) < alpha,torch.zeros(X_1.size()).to(device='cuda'),X_1)
                X_2 = torch.where(torch.abs(X_2) < beta ,torch.zeros(X_2.size()).to(device='cuda'),X_2)
                self.x1_for_save = X_1
                self.x2_for_save = X_2

                output1 = torch.mm(X_1, self.W_mu)
                output2 = torch.mm(X_2, std_w * eps_W)
                output = self.output_first + output1 + output2

                # # percent that satisfies the condition
                cond_num = torch.where(torch.abs(X_1) < alpha,torch.ones(X_1.size()).to(device='cuda'),torch.zeros(X_1.size()).to(device='cuda'))
                drop_rate_alpha = torch.sum(cond_num)/(X_1.shape[0]*X_1.shape[1])
                cond_num = torch.where(torch.abs(X_2) < beta ,torch.ones(X_2.size()).to(device='cuda'),torch.zeros(X_2.size()).to(device='cuda'))
                drop_rate_beta = torch.sum(cond_num)/(X_2.shape[0]*X_2.shape[1])
                self.droprate_alpha_mean = (self.droprate_alpha_mean * self.droprate_alpha_count + drop_rate_alpha) / (self.droprate_alpha_count + 1)
                self.droprate_alpha_count = self.droprate_alpha_count + 1
                self.droprate_beta_mean  = (self.droprate_beta_mean  * self.droprate_beta_count  + drop_rate_beta ) / (self.droprate_beta_count  + 1)
                self.droprate_beta_count = self.droprate_beta_count + 1


            return output, drop_rate_alpha, drop_rate_beta

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

        self.droprate_alpha_mean = 0
        self.droprate_alpha_count = 0
        self.droprate_beta_mean = 0
        self.droprate_beta_count = 0
        self.droprate_relu_mean = 0
        self.droprate_relu_count = 0

    def forward(self, X, sample=False, act_drop=False, first_layer=False, first_sample=False, given_alpha=0, given_beta=0):
        #         print(self.training)

        drop_rate_alpha = 0
        drop_rate_beta  = 0

        if not self.training and not sample:  # When training return MLE of w for quick validation
            # output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_out)
            output = F.conv2d(X, self.W_mu, bias = self.b_mu, padding = self.padding)
            return output, 0, 0

        elif first_sample:
            output = F.conv2d(X, self.W_mu, bias = self.b_mu, padding = self.padding)
            self.input_first = X
            # 小さすぎるものはのぞいておく
            self.output_first = torch.where(torch.abs(output) > 1e-6, output, torch.zeros(output.size()).to(device='cuda'))
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
              cond_num = torch.where(torch.abs(X) < 1e-6,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
              drop_rate = torch.sum(cond_num)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
              self.droprate_relu_mean = (self.droprate_relu_mean * self.droprate_relu_count + drop_rate) / (self.droprate_relu_count + 1)
              self.droprate_relu_count = self.droprate_relu_count + 1
            else :
              if(first_layer):
                # first layerではalphaの値が変わるので
                beta = given_beta

                X_new = torch.where(torch.abs(X)<beta,torch.zeros(X.size()).to(device='cuda'), X)
                output2 = F.conv2d(X_new, 1 * std_w * eps_W, padding=self.padding)
                output = self.output_first + output2
                drop_rate_beta  = torch.sum(torch.abs(X)<beta)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
                self.droprate_beta_mean  = (self.droprate_beta_mean  * self.droprate_beta_count  + drop_rate_beta ) / (self.droprate_beta_count  + 1)
                self.droprate_beta_count = self.droprate_beta_count + 1


              else:
                #平均を使うsampling手法
                alpha = given_alpha
                beta  = given_beta
                self.x_for_save = X
                cond_num = torch.where(torch.abs(X) < 1e-6,torch.ones(X.size()).to(device='cuda'),torch.zeros(X.size()).to(device='cuda'))
                drop_rate = torch.sum(cond_num)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
                # print(drop_rate)
                X_diff = X - self.input_first
                self.diff = torch.abs(X_diff)
                # X_diff = torch.where(torch.logical_and(torch.abs(X_diff)<beta, X<alpha),torch.zeros(X_diff.size()).to(device='cuda'), X_diff)
                X_diff = torch.where(torch.abs(X_diff)<alpha,torch.zeros(X_diff.size()).to(device='cuda'), X_diff)
                output1 = F.conv2d(X_diff, self.W_mu, padding=self.padding)

                X_new = torch.where(X<beta,torch.zeros(X.size()).to(device='cuda'), X)
                output2 = F.conv2d(X_new, 1 * std_w * eps_W, padding=self.padding)
                output = self.output_first + output1 + output2

                self.x1_for_save = X_diff
                self.x2_for_save = X_new

                # # print how many samples are skipped
                drop_rate_alpha = torch.sum(torch.abs(X_diff)<alpha)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
                drop_rate_beta  = torch.sum(X<beta)/(X.shape[0]*X.shape[1]*X.shape[2]*X.shape[3])
                # print(drop_rate_alpha,drop_rate_beta)
                self.droprate_alpha_mean = (self.droprate_alpha_mean * self.droprate_alpha_count + drop_rate_alpha) / (self.droprate_alpha_count + 1)
                self.droprate_alpha_count = self.droprate_alpha_count + 1
                self.droprate_beta_mean  = (self.droprate_beta_mean  * self.droprate_beta_count  + drop_rate_beta ) / (self.droprate_beta_count  + 1)
                self.droprate_beta_count = self.droprate_beta_count + 1

            return output, drop_rate_alpha, drop_rate_beta





class bayes_VGG11(nn.Module):
    """2 hidden layer Bayes By Backprop (VI) Network"""
    def __init__(self, input_dim, in_channels, output_dim, prior_instance, act_drop):
        super(bayes_VGG11, self).__init__()

        self.prior_instance = prior_instance

        self.in_channels = in_channels
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = BayesConv_Normalq(self.in_channels,64,3,self.prior_instance,padding=1)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv2 = BayesConv_Normalq(64,128,3,self.prior_instance,padding=1)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = BayesConv_Normalq(128,256,3,self.prior_instance,padding=1)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.conv4 = BayesConv_Normalq(256,256,3,self.prior_instance,padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2,2)

        self.conv5 = BayesConv_Normalq(256,512,3,self.prior_instance,padding=1)
        self.batchnorm5 = nn.BatchNorm2d(512)
        self.conv6 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.batchnorm6 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2,2)

        self.conv7 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.batchnorm7 = nn.BatchNorm2d(512)
        self.conv8 = BayesConv_Normalq(512,512,3,self.prior_instance,padding=1)
        self.batchnorm8 = nn.BatchNorm2d(512)
        self.pool5 = nn.MaxPool2d(2,2)

        self.bfc1 = BayesLinear_Normalq(512,4096, self.prior_instance)
        self.batchnorm9 = nn.BatchNorm1d(4096)
        self.bfc2 = BayesLinear_Normalq(4096,4096, self.prior_instance)
        self.batchnorm10 = nn.BatchNorm1d(4096)
        self.bfc3 = BayesLinear_Normalq(4096, self.output_dim, self.prior_instance)

        self.count = 0

        self.act = nn.ReLU(inplace=True)

        self.act_drop = act_drop

    def forward(self, x, sample=False, first_sample=False, alpha=0, beta=0):
        # dropout rateの平均を計算するための変数
        sum_a = 0
        sum_b = 0
        computation_count = 0

        # -----------------
        x1, a1, b1 = self.conv1(x, sample, self.act_drop, first_layer=True, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * self.in_channels * 64) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a1 * comp
        sum_b = sum_b + b1 * comp
        computation_count = computation_count + comp
        x1 = self.batchnorm1(x1)
        x = self.act(x1)
        x = self.pool1(x)
        # -----------------
        x2, a2, b2 = self.conv2(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 64 * 128) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a2 * comp
        sum_b = sum_b + b2 * comp
        computation_count = computation_count + comp
        x2 = self.batchnorm2(x2)
        x = self.act(x2)
        x = self.pool2(x)
        # -----------------
        x3, a3, b3 = self.conv3(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 128 * 256) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a3 * comp
        sum_b = sum_b + b3 * comp
        computation_count = computation_count + comp
        x3 = self.batchnorm3(x3)
        x = self.act(x3)
        x4, a4, b4 = self.conv4(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 256 * 256) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a4 * comp
        sum_b = sum_b + b4 * comp
        computation_count = computation_count + comp
        x4 = self.batchnorm4(x4)
        x = self.act(x4)
        x = self.pool3(x)
        # -----------------
        x5, a5, b5 = self.conv5(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 256 * 512) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a5 * comp
        sum_b = sum_b + b5 * comp
        computation_count = computation_count + comp
        x5 = self.batchnorm5(x5)
        x = self.act(x5)
        x6, a6, b6 = self.conv6(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 512 * 512) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a6 * comp
        sum_b = sum_b + b6 * comp
        computation_count = computation_count + comp
        x6 = self.batchnorm6(x6)
        x = self.act(x6)
        x = self.pool4(x)
        # -----------------
        x7, a7, b7 = self.conv7(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 512 * 512) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a7 * comp
        sum_b = sum_b + b7 * comp
        computation_count = computation_count + comp
        x7 = self.batchnorm7(x7)
        x = self.act(x7)
        x8, a8, b8 = self.conv8(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = (3 * 3 * 512 * 512) * x.shape[2] * x.shape[3]
        sum_a = sum_a + a8 * comp
        sum_b = sum_b + b8 * comp
        computation_count = computation_count + comp
        x8 = self.batchnorm8(x8)
        x = self.act(x8)
        x = self.pool5(x)
        # -----------------
        x = x.view(x.shape[0],-1)
        # -----------------
        x, a9, b9 = self.bfc1(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = 512 * 4096
        sum_a = sum_a + a9 * comp
        sum_b = sum_b + b9 * comp
        computation_count = computation_count + comp
        x = self.batchnorm9(x)
        x = self.act(x)
        x, a10, b10 = self.bfc2(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = 4096 * 4096
        sum_a = sum_a + a10 * comp
        sum_b = sum_b + b10 * comp
        computation_count = computation_count + comp
        x = self.batchnorm10(x)
        x = self.act(x)
        y, a11, b11 = self.bfc3(x, sample, self.act_drop, first_layer=False, first_sample=first_sample, given_alpha=alpha, given_beta=beta)
        comp = 4096 * 10
        sum_a = sum_a + a11 * comp
        sum_b = sum_b + b11 * comp
        computation_count = computation_count + comp


        if(self.count == 22):
          self.conv_out = [x1,x2,x3,x4,x5,x6,x7,x8]
        self.count = self.count + 1

        # dropout rateの平均を計算
        sum_a = sum_a / computation_count
        sum_b = sum_b / computation_count
        return y, sum_a, sum_b

    def sample_predict(self, x, Nsamples, alpha=0, beta=0):
        """Used for estimating the data's likelihood by approximately marginalising the weights with MC"""
        # Just copies type from x, initializes new vector
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        drop_rate_alpha = 0
        drop_rate_beta  = 0

        for i in range(Nsamples+1):
            if(i == 0):
                y, tklw, tklb = self.forward(x, sample=True, first_sample=True)
            else:
                y, tklw, tklb = self.forward(x, sample=True, first_sample=False, alpha=alpha, beta=beta)
                predictions[i-1] = y
                drop_rate_alpha = drop_rate_alpha + tklw
                drop_rate_beta  = drop_rate_beta  + tklb

        # dropout rateの平均をとる
        drop_rate_alpha = drop_rate_alpha / Nsamples
        drop_rate_beta  = drop_rate_beta  / Nsamples

        return predictions, drop_rate_alpha, drop_rate_beta

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

    def fit(self, x, y, samples=1):
        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

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

    def all_sample_eval(self, x, y, Nsamples, alpha=0, beta=0):
        """Returns predictions for each MC sample"""
        # initialize seed if needed
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        x, y = to_variable(var=(x, y.long()), cuda=self.cuda)

        out, drop_rate_alpha, drop_rate_beta = self.model.sample_predict(x, Nsamples, alpha=alpha, beta=beta)

        prob_out = F.softmax(out, dim=2)
        prob_out = prob_out.data

        return prob_out, drop_rate_alpha, drop_rate_beta

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