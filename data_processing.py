from imports import*
import utils

class dai_image_csv_dataset(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None, obj = False,
                    minorities = None, diffs = None, bal_tfms = None):
        super(dai_image_csv_dataset, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')

        # img = torchvision.transforms.functional.to_grayscale(img,num_output_channels=3)

        y = self.data.iloc[index, 1]    
        if self.minorities and self.bal_tfms:
            if y in self.minorities:
                if hasattr(self.bal_tfms,'transforms'):
                    for tr in self.bal_tfms.transforms:
                        tr.p = self.diffs[y]
                    l = [self.bal_tfms]
                    l.extend(self.transforms_)
                    self.tfms = transforms.Compose(l)    
                else:            
                    for t in self.bal_tfms:
                        t.p = self.diffs[y]
                    self.transforms_[1:1] = self.bal_tfms    
                    self.tfms = transforms.Compose(self.transforms_)
                    # print(self.tfms)
            else:
                self.tfms = transforms.Compose(self.transforms_)
        else:    
            self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        # if self.obj:
        #     s = x.size()[1]
        #     if isinstance(s,tuple):
        #         s = s[0]
        #     row_scale = s/img.size[0]
        #     col_scale = s/img.size[1]
        #     y = rescale_bbox(y,row_scale,col_scale)
        #     y.squeeze_()
        #     y2 = self.data.iloc[index, 2]
        #     y = (y,y2)
        return (x,y)


class dai_image_csv_dataset_food(Dataset):
    
    def __init__(self, data_dir, data, transforms_ = None, obj = False,
                    minorities = None, diffs = None, bal_tfms = None):
        super(dai_image_csv_dataset_food, self).__init__()
        self.data_dir = data_dir
        self.data = data
        self.transforms_ = transforms_
        self.tfms = None
        self.obj = obj
        self.minorities = minorities
        self.diffs = diffs
        self.bal_tfms = bal_tfms
        assert transforms_ is not None, print('Please pass some transforms.')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        y1,y2 = self.data.iloc[index, 1],self.data.iloc[index, 2]    
        self.tfms = transforms.Compose(self.transforms_)    
        x = self.tfms(img)
        return (x,y1,y2)

class dai_image_dataset(Dataset):

    def __init__(self, data_dir, data_df, input_transforms = None, target_transforms = None):
        super(dai_image_dataset, self).__init__()
        self.data_dir = data_dir
        self.data_df = data_df
        self.input_transforms = None
        self.target_transforms = None
        if input_transforms:
            self.input_transforms = transforms.Compose(input_transforms)
        if target_transforms:    
            self.target_transforms = transforms.Compose(target_transforms)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir,self.data_df.iloc[index, 0])
        img = Image.open(img_path)
        img = img.convert('RGB')
        target = img.copy()
        if self.input_transforms:
            img = self.input_transforms(img)
        if self.target_transforms:
            target = self.target_transforms(target)
        return img, target

def listdir_fullpath(d):
        return [os.path.join(d, f) for f in os.listdir(d)]     

def get_minorities(df,thresh=0.8):

    c = df.iloc[:,1].value_counts()
    lc = list(c)
    max_count = lc[0]
    diffs = [1-(x/max_count) for x in lc]
    diffs = dict((k,v) for k,v in zip(c.keys(),diffs))
    minorities = [c.keys()[x] for x,y in enumerate(lc) if y < (thresh*max_count)]
    return minorities,diffs

def csv_from_path(path, img_dest):

    path = Path(path)
    img_dest = Path(img_dest)
    labels_paths = list(path.iterdir())
    tr_images = []
    tr_labels = []
    for l in labels_paths:
        if l.is_dir():
            for i in list(l.iterdir()):
                if i.suffix in IMG_EXTENSIONS:
                    name = i.name
                    label = l.name
                    new_name = '{}_{}_{}'.format(path.name,label,name)
                    new_path = img_dest/new_name
#                     print(new_path)
                    os.rename(i,new_path)
                    tr_images.append(new_name)
                    tr_labels.append(label)
            # os.rmdir(l)
    tr_img_label = {'Img':tr_images, 'Label': tr_labels}
    csv = pd.DataFrame(tr_img_label,columns=['Img','Label'])
    csv = csv.sample(frac=1).reset_index(drop=True)
    return csv

def add_extension(a,e):
    a = [x+e for x in a]
    return a

def one_hot(targets, multi = False):
    if multi:
        binerizer = MultiLabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    else:
        binerizer = LabelBinarizer()
        dai_1hot = binerizer.fit_transform(targets)
    return dai_1hot,binerizer.classes_

def get_img_stats(dataset,sz):

    print('Calculating mean and std of the data for standardization. Might take some time, depending on the training data size.')

    size = int(len(dataset)*sz)
    i = 0
    imgs = []
    for d in dataset:
        img = d[0]
        # print(img.size())
        if i > size:
            break
        imgs.append(img)
        i+=1
    imgs_ = torch.stack(imgs,dim=3)
    imgs_ = imgs_.view(3,-1)
    imgs_mean = imgs_.mean(dim=1)
    imgs_std = imgs_.std(dim=1)
    del imgs
    del imgs_
    print('Done')
    return imgs_mean,imgs_std

def split_df(train_df,test_size = 0.15):
    try:    
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2,stratify = train_df.iloc[:,1])
    except:
        train_df,val_df = train_test_split(train_df,test_size = test_size,random_state = 2)
    train_df = train_df.reset_index(drop = True)
    val_df =  val_df.reset_index(drop = True)
    return train_df,val_df    

def save_obj(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class DataProcessor:
    
    def __init__(self, data_path = None, train_csv = None, val_csv = None,test_csv = None,
                 tr_name = 'train', val_name = 'val', test_name = 'test', extension = None, setup_data = True):
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        (self.data_path,self.train_csv,self.val_csv,self.test_csv,
         self.tr_name,self.val_name,self.test_name,self.extension) = (data_path,train_csv,val_csv,test_csv,
                                                                      tr_name,val_name,test_name,extension)
        
        self.multi_label = False
        self.single_label = False
        self.img_mean = self.img_std = None
        self.data_dir,self.num_classes,self.class_names = data_path,0,[]
        if setup_data:
            self.set_up_data()
                
    def set_up_data(self,split_size = 0.15):

        (data_path,train_csv,val_csv,test_csv,tr_name,val_name,test_name) = (self.data_path,self.train_csv,self.val_csv,self.test_csv,
                                                                             self.tr_name,self.val_name,self.test_name)

        # check if paths given and also set paths
        
        if not data_path:
            data_path = os.getcwd() + '/'
        tr_path = os.path.join(data_path,tr_name)
        val_path = os.path.join(data_path,val_name)
        test_path = os.path.join(data_path,test_name)

        if (os.path.exists(os.path.join(data_path,tr_name+'.csv'))) and train_csv is None:
            train_csv = tr_name+'.csv'
        # if os.path.exists(os.path.join(data_path,val_name+'.csv')):
        #     val_csv = val_name+'.csv'
        # if os.path.exists(os.path.join(data_path,test_name+'.csv')):
        #     test_csv = test_name+'.csv'    

        # paths to csv

        if not train_csv:
            # print('no')
            train_csv,val_csv,test_csv = self.data_from_paths_to_csv(data_path,tr_path,val_path,test_path)

        train_csv_path = os.path.join(data_path,train_csv)
        train_df = pd.read_csv(train_csv_path)
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop('Unnamed: 0', 1)
        if len(train_df.columns) > 2:
            self.obj = True    
        img_names = [str(x) for x in list(train_df.iloc[:,0])]
        if self.extension:
            img_names = add_extension(img_names,self.extension)
        if val_csv:
            val_csv_path = os.path.join(data_path,val_csv)
            val_df = pd.read_csv(val_csv_path)
            val_targets = list(val_df.iloc[:,1].apply(lambda x: str(x)))
        if test_csv:
            test_csv_path = os.path.join(data_path,test_csv)
            test_df = pd.read_csv(test_csv_path)
            test_targets = list(test_df.iloc[:,1].apply(lambda x: str(x)))
        targets = list(train_df.iloc[:,1].apply(lambda x: str(x)))
        lengths = [len(t) for t in [s.split() for s in targets]]
        self.target_lengths = lengths
        split_targets = [t.split() for t in targets]

        if lengths[1:] != lengths[:-1]:
            self.multi_label = True
            # print('\nMulti-label Classification\n')
            try:
                split_targets = [list(map(int,x)) for x in split_targets]
            except:
                pass
            dai_onehot,onehot_classes = one_hot(split_targets,self.multi_label)
            train_df.iloc[:,1] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]
            self.data_dir,self.num_classes,self.class_names = data_path,len(onehot_classes),onehot_classes

        else:
            # print('\nSingle-label Classification\n')
            self.single_label = True
            unique_targets = list(np.unique(targets))
            unique_targets_dict = {k:v for v,k in enumerate(unique_targets)}
            train_df.iloc[:,1] = pd.Series(targets).apply(lambda x: unique_targets_dict[x])
            if val_csv:
                val_df.iloc[:,1] = pd.Series(val_targets).apply(lambda x: unique_targets_dict[x])
            if test_csv:
                test_df.iloc[:,1] = pd.Series(test_targets).apply(lambda x: unique_targets_dict[x])   
            self.data_dir,self.num_classes,self.class_names = data_path,len(unique_targets),unique_targets

        if not val_csv:
            train_df,val_df = split_df(train_df,split_size)
        if not test_csv:    
            val_df,test_df = split_df(val_df,split_size)
        tr_images = [str(x) for x in list(train_df.iloc[:,0])]
        val_images = [str(x) for x in list(val_df.iloc[:,0])]
        test_images = [str(x) for x in list(test_df.iloc[:,0])]
        if self.extension:
            tr_images = add_extension(tr_images,self.extension)
            val_images = add_extension(val_images,self.extension)
            test_images = add_extension(test_images,self.extension)
        train_df.iloc[:,0] = tr_images
        val_df.iloc[:,0] = val_images
        test_df.iloc[:,0] = test_images
        if self.single_label:
            dai_df = pd.concat([train_df,val_df,test_df])
            dai_df.iloc[:,1] = [self.class_names[x] for x in dai_df.iloc[:,1]]
            dai_df.to_csv(os.path.join(data_path,'dai_df.csv'),index=False)
        train_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.tr_name)),index=False)
        val_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.val_name)),index=False)
        test_df.to_csv(os.path.join(data_path,'{}.csv'.format(self.test_name)),index=False)
        self.minorities,self.class_diffs = None,None
        if self.single_label:
            self.minorities,self.class_diffs = get_minorities(train_df)
        self.data_dfs = {self.tr_name:train_df, self.val_name:val_df, self.test_name:test_df}
        data_dict = {'data_dfs':self.data_dfs,'data_dir':self.data_dir,'num_classes':self.num_classes,'class_names':self.class_names,
                'minorities':self.minorities,'class_diffs':self.class_diffs,'single_label':self.single_label,'multi_label':self.multi_label}
        self.data_dict = data_dict
        return data_dict

    def data_from_paths_to_csv(self,data_path,tr_path,val_path = None,test_path = None):
            
        train_df = csv_from_path(tr_path,tr_path)
        train_df.to_csv(os.path.join(data_path,self.tr_name+'.csv'),index=False)
        ret = (self.tr_name+'.csv',None,None)
        if val_path is not None:
            val_exists = os.path.exists(val_path)
            if val_exists:
                val_df = csv_from_path(val_path,tr_path)
                val_df.to_csv(os.path.join(data_path,self.val_name+'.csv'),index=False)
                ret = (self.tr_name+'.csv',self.val_name+'.csv',None)
        if test_path is not None:
            test_exists = os.path.exists(test_path)
            if test_exists:
                test_df = csv_from_path(test_path,tr_path)
                test_df.to_csv(os.path.join(data_path,self.test_name+'.csv'),index=False)
                ret = (self.tr_name+'.csv',self.val_name+'.csv',self.test_name+'.csv')        
        return ret
        
    def get_data(self, data_dict = None, s = (224,224), dataset = dai_image_csv_dataset, bs = 32, balance = False, tfms = None,bal_tfms = None,
                 num_workers = 8, stats_percentage = 0.6, img_mean = None, img_std = None):
        
        self.image_size = s
        if not data_dict:
            data_dict = self.data_dict
        data_dfs,data_dir,minorities,class_diffs,single_label,multi_label = (data_dict['data_dfs'],data_dict['data_dir'],
                                                        data_dict['minorities'],data_dict['class_diffs'],
                                                        data_dict['single_label'],data_dict['multi_label'])
        if not single_label:
           balance = False                                                 
        
        if not bal_tfms:
            bal_tfms = { self.tr_name: [transforms.RandomHorizontalFlip()],
                         self.val_name: None,
                         self.test_name: None 
                       }
        else:
            bal_tfms = {self.tr_name: bal_tfms, self.val_name: None, self.test_name: None}
            
        resize_transform = transforms.Resize(s,interpolation=Image.NEAREST)

        if not tfms:
            tfms = [
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        else:
            
            tfms_temp = [
                resize_transform,
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            tfms_temp[1:1] = tfms
            tfms = tfms_temp
            # print(tfms)
        
        data_transforms = {
            self.tr_name: tfms,
            self.val_name: [
                # transforms.Resize(s[0]+50),
                # transforms.CenterCrop(s[0]),
                transforms.Resize(s,interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ],
            self.test_name: [
                # transforms.Resize(s[0]+50),
                # transforms.CenterCrop(s[0]),
                transforms.Resize(s,interpolation=Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        }
        if img_mean is None and self.img_mean is None:
            temp_tfms = [resize_transform, transforms.ToTensor()]
            temp_dataset = dataset(os.path.join(data_dir,self.tr_name),data_dfs[self.tr_name],temp_tfms)
            self.img_mean,self.img_std = get_img_stats(temp_dataset,stats_percentage)
        elif self.img_mean is None:
            self.img_mean,self.img_std = img_mean,img_std
        data_transforms[self.tr_name][-1].mean,data_transforms[self.tr_name][-1].std = self.img_mean,self.img_std
        data_transforms[self.val_name][-1].mean,data_transforms[self.val_name][-1].std = self.img_mean,self.img_std
        data_transforms[self.test_name][-1].mean,data_transforms[self.test_name][-1].std = self.img_mean,self.img_std

        if balance:
            image_datasets = {x: dataset(os.path.join(data_dir,self.tr_name),data_dfs[x],
                                        data_transforms[x],minorities,class_diffs,bal_tfms[x])
                        for x in [self.tr_name, self.val_name, self.test_name]}    
        else:
            image_datasets = {x: dataset(os.path.join(data_dir,self.tr_name),data_dfs[x],
                                                            data_transforms[x])
                        for x in [self.tr_name, self.val_name, self.test_name]}
        
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs,
                                                     shuffle=True, num_workers=num_workers)
                      for x in [self.tr_name, self.val_name, self.test_name]}
        dataset_sizes = {x: len(image_datasets[x]) for x in [self.tr_name, self.val_name, self.test_name]}
        
        self.image_datasets,self.dataloaders,self.dataset_sizes = (image_datasets,dataloaders,
                                                                                    dataset_sizes)
        
        return image_datasets,dataloaders,dataset_sizes