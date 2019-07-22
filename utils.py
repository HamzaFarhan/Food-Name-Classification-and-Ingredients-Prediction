from imports import*

def display_img_actual_size(im_data,title = ''):
    dpi = 80
    height, width, depth = im_data.shape
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(im_data, cmap='gray')
    plt.title(title,fontdict={'fontsize':25})
    plt.show()

def plt_show(im):
    plt.imshow(im)
    plt.show()

def load_and_show(path):
    img = plt.imread(path)
    plt_show(img)    

def denorm_img_general(inp,mean=None,std=None):
    inp = inp.numpy()
    inp = inp.transpose((1, 2, 0))
    if mean is None:
        mean = np.mean(inp)
    if std is None:    
        std = np.std(inp)
    inp = std * inp + mean
    inp = np.clip(inp, 0., 1.)
    return inp 

def bgr2rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def rgb2bgr(img):
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)    

def plot_in_row(imgs,figsize = (20,20),rows = None,columns = None,titles = [],fig_path = 'fig.png'):
    fig=plt.figure(figsize=figsize)
    if len(titles) == 0:
        titles = ['image_{}'.format(i) for i in range(len(imgs))]
    if not rows:
        rows = 1
        if columns:
            rows = len(imgs)//columns    
    if not columns:    
        columns = len(imgs)
        if rows:
            columns = len(imgs)//rows
    for i in range(1, columns*rows +1):
        img = imgs[i-1]
        fig.add_subplot(rows, columns, i, title = titles[i-1])
        plt.imshow(img)
    fig.savefig(fig_path)    
    plt.show()

def tensor_to_img(t):
    if len(t.shape) > 3:
        return [np.transpose(t_,(1,2,0)) for t_ in t]
    return np.transpose(t,(1,2,0))

def smooth_labels(labels,eps=0.1):
    if len(labels.shape) > 1:
        length = len(labels[0])
    else:
        length = len(labels)
    labels = labels * (1 - eps) + (1-labels) * eps / (length - 1)
    return labels

def get_test_input(paths = [],imgs = [], size = None, size_factor = None, show = False, norm = False, bgr_to_rgb = False):
    if len(paths) > 0:
        bgr_to_rgb = True
        imgs = []
        for p in paths:
            imgs.append(cv2.imread(str(p)))
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        if size_factor:
            img = cv2.resize(img, (img.shape[1]//size_factor,img.shape[0]//size_factor))        
        if bgr_to_rgb:
            img = bgr2rgb(img)
        if show:
            plt_show(img)    
        img_ =  img.transpose((2,0,1)) # H X W C -> C X H X W
        if norm:
            img_ = (img_ - np.mean(img_))/np.std(img_)
        imgs[i] = img_/255.
    return torch.from_numpy(np.asarray(imgs)).float()

def batch_to_imgs(batch,mean=None,std=None):
    imgs = []
    for i in batch:
        imgs.append(denorm_img_general(i,mean,std))
    return imgs    

def mini_batch(dataset,bs,start=0):
    imgs = torch.Tensor(bs,*dataset[0][0].shape)
    s = dataset[0][1].shape
    if len(s) > 0:
        labels = torch.Tensor(bs,*s)
    else:    
        labels = torch.Tensor(bs).int()
    for i in range(start,bs+start):
        b = dataset[i]
        imgs[i-start] = b[0]
        labels[i-start] = tensor(b[1])
    return imgs,labels    

def to_batch(paths = [],imgs = [], size = None):
    if len(paths) > 0:
        imgs = []
        for p in paths:
            imgs.append(cv2.imread(p))
    for i,img in enumerate(imgs):
        if size:
            img = cv2.resize(img, size)
        img =  img.transpose((2,0,1))
        imgs[i] = img
    return torch.from_numpy(np.asarray(imgs)).float()    

def get_optim(optimizer_name,params,lr):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(params=params,lr=lr)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(params=params,lr=lr)
    elif optimizer_name.lower() == 'adadelta':
        return optim.Adadelta(params=params)

def unfreeze_model(model):
    for param in model.parameters():
        param.requires_grad = True

class Printer(nn.Module):
    def forward(self,x):
        print(x.size())
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

DAI_AvgPool = nn.AdaptiveAvgPool2d(1)

class RunningBatchNorm(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1))
        self.register_buffer('batch', tensor(0.))
        self.register_buffer('count', tensor(0.))
        self.register_buffer('step', tensor(0.))
        self.register_buffer('dbias', tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel()/nc)
        mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step<100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c    = c    / self.dbias
        means = sums/c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)

def flatten_tensor(x):
    return x.view(x.shape[0],-1)

def rmse(inputs,targets):
    return torch.sqrt(torch.mean((inputs - targets) ** 2))

def psnr(mse):
    return 10 * math.log10(1 / mse)

def get_psnr(inputs,targets):
    mse_loss = F.mse_loss(inputs,targets)
    return 10 * math.log10(1 / mse_loss)