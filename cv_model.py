from imports import*
from utils import *
from model import *
from fc import *
from parallel import DataParallelModel, DataParallelCriterion

class FoodIngredients(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='food',
                 lr=0.02,
                 optimizer_name = 'Adam',
                 criterion1 = nn.CrossEntropyLoss(),
                 criterion2 = nn.BCEWithLogitsLoss(),
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 head1 = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 head2 = {'num_outputs':10,
                    'layers':[],
                    'model_type':'multi_label_classifier'
                 },
                 class_names = [],
                 num_classes = None,
                 ingredient_names = [],
                 num_ingredients = None,
                 add_extra = True,
                 set_params = True,
                 set_head = True
                 ):

        super().__init__(device=device)

        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head1 = head1,
                    head2 = head2,
                    dropout_p = dropout_p,
                    criterion1 = criterion1,
                    criterion2 = criterion2,
                    device = device
                )
        if set_params:
            self.set_model_params(
                              optimizer_name = optimizer_name,
                              lr = lr,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              class_names = class_names,
                              num_classes = num_classes,
                              ingredient_names = ingredient_names, 
                              num_ingredients = num_ingredients,
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,
                         criterion1 = nn.CrossEntropyLoss(),
                         criterion2 = nn.BCEWithLogitsLoss(),
                         optimizer_name = 'Adam',
                         lr  = 0.1,
                         dropout_p = 0.45,
                         model_name = 'DenseNet',
                         model_type = 'cv_transfer',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth',
                         head1 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'classifier'
                                },
                         head2 = {'num_outputs':10,
                                'layers':[],
                                'model_type':'muilti_label_classifier'
                                },       
                         class_names = [],
                         num_classes = None,
                         ingredient_names = [],
                         num_ingredients = None):
        
        print('Food Names: current best accuracy = {:.3f}'.format(best_accuracy))
        if best_validation_loss is not None:
            print('Food Ingredients: current best loss = {:.3f}'.format(best_validation_loss))

        
        super(FoodIngredients, self).set_model_params(
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file
                                              )
        self.class_names = class_names
        self.num_classes = num_classes                                              
        self.ingredeint_names = ingredient_names
        self.num_ingredients = num_ingredients
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def forward(self,x):
        l = list(self.model.children())
        for m in l[:-2]:
            x = m(x)
        food = l[-2](x)
        ingredients = l[-1](x)
        return (food,ingredients)
    
    def compute_loss(self,outputs,labels,w1 = 1.,w2 = 1.): 
        out1,out2 = outputs
        label1,label2 = labels
        loss1 = self.criterion1(out1,label1)
        loss2 = self.criterion2(out2,label2)
        return [(loss1*w1)+(loss2*w2)]

    def freeze(self,train_classifier=True):
        super(FoodIngredients, self).freeze()
        if train_classifier:
            for param in self.model.fc1.parameters():
                 param.requires_grad = True
            for param in self.model.fc2.parameters():
                 param.requires_grad = True     

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head1 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        head2 = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'muilti_label_classifier'
                               },       
                        criterion1 = nn.CrossEntropyLoss(),
                        criterion2 = nn.BCEWithLogitsLoss(), 
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        heads = [head1,head2]
        crits = [criterion1,criterion2]    
        fcs = []
        for head,criterion in zip(heads,crits):
            head['criterion'] = criterion
            if head['model_type'].lower() == 'classifier':
                head['output_non_linearity'] = None
            fc = modules[-1]
            try:
                in_features =  fc.in_features
            except:
                in_features = fc.model.out.in_features    
            fc = FC(
                    num_inputs = in_features,
                    num_outputs = head['num_outputs'],
                    layers = head['layers'],
                    model_type = head['model_type'],
                    output_non_linearity = head['output_non_linearity'],
                    dropout_p = dropout_p,
                    criterion = head['criterion'],
                    optimizer_name = None,
                    device = device
                    )
            fcs.append(fc)          
        if adaptive:
            l += meta['adaptive_head']
        else:
            l += meta['normal_head']
        model = nn.Sequential(*l)
        model.add_module('fc1',fcs[0])
        model.add_module('fc2',fcs[1])
        self.model = model
        self.head1 = head1
        self.head2 = head2
        
        print('Multi-head set up complete.')

    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            batches += 1
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            if self.parallel:
                loss.sum().backward()
                loss = loss.sum()
            else:    
                loss.backward()
                loss = loss.item()
            optimizer.step()
            running_loss += loss
            if batches % print_every == 0:
                elapsed = time.time()-t1
                if elapsed > 60:
                    elapsed /= 60.
                    measure = 'min'
                else:
                    measure = 'sec'
                batch_time = time.time()-t0
                if batch_time > 60:
                    batch_time /= 60.
                    measure2 = 'min'
                else:
                    measure2 = 'sec'    
                print('+----------------------------------------------------------------------+\n'
                        f"{time.asctime().split()[-2]}\n"
                        f"Time elapsed: {elapsed:.3f} {measure}\n"
                        f"Epoch:{epoch+1}/{epochs}\n"
                        f"Batch: {batches+1}/{len(trainloader)}\n"
                        f"Batch training time: {batch_time:.3f} {measure2}\n"
                        f"Batch training loss: {loss:.3f}\n"
                        f"Average training loss: {running_loss/(batches):.3f}\n"
                      '+----------------------------------------------------------------------+\n'     
                        )
                t0 = time.time()
        return running_loss/len(trainloader) 

    def evaluate(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        if self.model_type == 'classifier':# or self.num_classes is not None:
           classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
                inputs = inputs.to(self.device)
                label1 = label1.to(self.device)
                label2 = label2.to(self.device)
                labels = (label1,label2)
                outputs = self.forward(inputs)
                loss = self.compute_loss(outputs,labels)[0]
                if self.parallel:
                    running_loss += loss.sum()
                    outputs = parallel.gather(outputs,self.device)
                else:        
                    running_loss += loss.item()
                if classifier is not None and metric == 'accuracy':
                    classifier.update_accuracies(outputs,labels)
                    y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                    _, preds = torch.max(torch.exp(outputs), 1)
                    y_pred.extend(list(preds.cpu().numpy()))
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret

    def evaluate_food(self,dataloader,metric='accuracy'):
        
        running_loss = 0.
        classifier = None

        classifier = Classifier(self.class_names)

        y_pred = []
        y_true = []

        self.eval()
        rmse_ = 0.
        with torch.no_grad():
            for data_batch in dataloader:
                inputs,labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(inputs)[0]
                if classifier is not None and metric == 'accuracy':
                    try:
                        classifier.update_accuracies(outputs,labels)
                        y_true.extend(list(labels.squeeze(0).cpu().numpy()))
                        _, preds = torch.max(torch.exp(outputs), 1)
                        y_pred.extend(list(preds.cpu().numpy()))
                    except:
                        pass    
                elif metric == 'rmse':
                    rmse_ += rmse(outputs,labels).cpu().numpy()
            
        self.train()

        ret = {}
        # print('Running_loss: {:.3f}'.format(running_loss))
        if metric == 'rmse':
            print('Total rmse: {:.3f}'.format(rmse_))
            ret['final_rmse'] = rmse_/len(dataloader)

        ret['final_loss'] = running_loss/len(dataloader)

        if classifier is not None:
            ret['accuracy'],ret['class_accuracies'] = classifier.get_final_accuracies()
            ret['report'] = classification_report(y_true,y_pred,target_names=self.class_names)
            ret['confusion_matrix'] = confusion_matrix(y_true,y_pred)
            try:
                ret['roc_auc_score'] = roc_auc_score(y_true,y_pred)
            except:
                pass
        return ret    

    def find_lr(self,trn_loader,init_value=1e-8,final_value=10.,beta=0.98,plot=False):
        
        print('\nFinding the ideal learning rate.')

        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())
        optimizer = self.optimizer
        num = len(trn_loader)-1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for data_batch in trn_loader:
            batch_num += 1
            inputs,label1,label2 = data_batch[0],data_batch[1],data_batch[2]
            inputs = inputs.to(self.device)
            label1 = label1.to(self.device)
            label2 = label2.to(self.device)
            labels = (label1,label2)
            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = self.compute_loss(outputs,labels)[0]
            #Compute the smoothed loss
            if self.parallel:   
                avg_loss = beta * avg_loss + (1-beta) * loss.sum()    
            else:    
                avg_loss = beta * avg_loss + (1-beta) * loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                self.log_lrs, self.find_lr_losses = log_lrs,losses
                self.model.load_state_dict(model_state)
                self.optimizer.load_state_dict(optim_state)
                if plot:
                    self.plot_find_lr()
                temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//8)]
                self.lr = (10**temp_lr)
                print('Found it: {}\n'.format(self.lr))
                return self.lr
            #Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            #Do the SGD step
            if self.parallel:
                loss.sum().backward()
            else:    
                loss.backward()
            optimizer.step()
            #Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        self.log_lrs, self.find_lr_losses = log_lrs,losses
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)
        if plot:
            self.plot_find_lr()
        temp_lr = self.log_lrs[np.argmin(self.find_lr_losses)-(len(self.log_lrs)//10)]
        self.lr = (10**temp_lr)
        print('Found it: {}\n'.format(self.lr))
        return self.lr
            
    def plot_find_lr(self):    
        plt.ylabel("Loss")
        plt.xlabel("Learning Rate (log scale)")
        plt.plot(self.log_lrs,self.find_lr_losses)
        plt.show()

    def classify(self,inputs,thresh = 0.4):#,show = False,mean = None,std = None):
        outputs = self.predict(inputs)
        food,ing = outputs
        try:    
            _, preds = torch.max(torch.exp(food), 1)
        except:
            _, preds = torch.max(torch.exp(food.unsqueeze(0)), 1)
        ing_outs = ing.sigmoid()
        ings = (ing_outs >= thresh)
        class_preds = [str(self.class_names[p]) for p in preds]
        ing_preds = [self.ingredeint_names[p.nonzero().squeeze(1).cpu()] for p in ings]
        return class_preds,ing_preds

    def _get_dropout(self):
        return self.dropout_p

    def get_model_params(self):
        params = super(FoodIngredients, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes
        params['ingredient_names'] = self.ingredient_names
        params['num_ingredients'] = self.num_ingredients
        params['head1'] = self.head1
        params['head2'] = self.head2
        return params        

class TransferNetworkImg(Network):
    def __init__(self,
                 model_name='DenseNet',
                 model_type='cv_transfer',
                 lr=0.02,
                 criterion = nn.CrossEntropyLoss(),
                 optimizer_name = 'Adam',
                 dropout_p=0.45,
                 pretrained=True,
                 device=None,
                 best_accuracy=0.,
                 best_validation_loss=None,
                 best_model_file ='best_model.pth',
                 head = {'num_outputs':10,
                    'layers':[],
                    'model_type':'classifier'
                 },
                 class_names = [],
                 num_classes = None,
                 add_extra = True,
                 set_params = True,
                 set_head = True
                 ):

        super().__init__(device=device)

        self.set_transfer_model(model_name,pretrained=pretrained,add_extra=add_extra,dropout_p=dropout_p)

        if set_head:
            self.set_model_head(model_name = model_name,
                    head = head,
                    dropout_p = dropout_p,
                    criterion = criterion,
                    device = device
                )
        if set_params:
            self.set_model_params(criterion = criterion,
                              optimizer_name = optimizer_name,
                              lr = lr,
                              dropout_p = dropout_p,
                              model_name = model_name,
                              model_type = model_type,
                              best_accuracy = best_accuracy,
                              best_validation_loss = best_validation_loss,
                              best_model_file = best_model_file,
                              class_names = class_names,
                              num_classes = num_classes
                              )

        self.model = self.model.to(device)
        
    def set_model_params(self,criterion = nn.CrossEntropyLoss(),
                         optimizer_name = 'Adam',
                         lr  = 0.1,
                         dropout_p = 0.45,
                         model_name = 'DenseNet',
                         model_type = 'cv_transfer',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth',
                         class_names = [],
                         num_classes = None):
        
        print('Transfer Learning: current best accuracy = {:.3f}'.format(best_accuracy))
        
        super(TransferNetworkImg, self).set_model_params(
                                              criterion = criterion,
                                              optimizer_name = optimizer_name,
                                              lr = lr,
                                              dropout_p = dropout_p,
                                              model_name = model_name,
                                              model_type = model_type,
                                              best_accuracy = best_accuracy,
                                              best_validation_loss = best_validation_loss,
                                              best_model_file = best_model_file
                                              )
        self.class_names = class_names
        self.num_classes = num_classes                                              
        if len(class_names) == 0:
            self.class_names = {k:str(v) for k,v in enumerate(list(range(self.head['num_outputs'])))}

    def forward(self,x):
        return self.model(x)
    
    def freeze(self,train_classifier=True):
        super(TransferNetworkImg, self).freeze()
        if train_classifier:
            for param in self.model.fc.parameters():
                 param.requires_grad = True

    def parallelize(self):
        self.parallel = True
        self.model = DataParallelModel(self.model)
        self.criterion  = DataParallelCriterion(self.criterion)

    def set_transfer_model(self,mname,pretrained=True,add_extra=True,dropout_p = 0.45):   
        self.model = None
        models_dict = {

            'densenet': {'model':models.densenet121(pretrained=pretrained),'conv_channels':1024},
            'resnet34': {'model':models.resnet34(pretrained=pretrained),'conv_channels':512},
            'resnet50': {'model':models.resnet50(pretrained=pretrained),'conv_channels':2048}

        }
        meta = models_dict[mname.lower()]
        try:
            model = meta['model']
            for param in model.parameters():
                param.requires_grad = False
            self.model = model    
            print('Setting transfer learning model: self.model set to {}'.format(mname))
        except:
            print('Setting transfer learning model: model name {} not supported'.format(mname))            

        # creating and adding extra layers to the model
        dream_model = None
        if add_extra:
            channels = meta['conv_channels']
            dream_model = nn.Sequential(
                nn.Conv2d(channels,channels,3,1,1),
                # Printer(),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p),
                nn.Conv2d(channels,channels,3,1,1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True),
                nn.Dropout2d(dropout_p)
                )        
        self.dream_model = dream_model          
           
    def set_model_head(self,
                        model_name = 'DenseNet',
                        head = {'num_outputs':10,
                                'layers':[],
                                'class_names': None,
                                'model_type':'classifier'
                               },
                        criterion = nn.NLLLoss(),  
                        adaptive = True,       
                        dropout_p = 0.45,
                        device = None):

        models_meta = {
        'resnet34': {'conv_channels':512,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'resnet50': {'conv_channels':2048,'head_id': -2, 'adaptive_head': [DAI_AvgPool],'normal_head': [nn.AvgPool2d(7,1)]},
        'densenet': {'conv_channels':1024,'head_id': -1,'adaptive_head': [nn.ReLU(inplace=True),DAI_AvgPool]
                    ,'normal_head': [nn.ReLU(inplace=True),nn.AvgPool2d(7,1)]}
        }

        name = model_name.lower()
        meta = models_meta[name]
        modules = list(self.model.children())
        l = modules[:meta['head_id']]
        if self.dream_model:
            l+=self.dream_model
        if type(head).__name__ != 'dict':
            model = nn.Sequential(*l)
            for layer in head.children():
                if(type(layer).__name__) == 'StdConv':
                    conv_module = layer
                    break
            conv_layer = conv_module.conv
            temp_args = [conv_layer.out_channels,conv_layer.kernel_size,conv_layer.stride,conv_layer.padding]
            temp_args.insert(0,meta['conv_channels'])
            conv_layer = nn.Conv2d(*temp_args)
            conv_module.conv = conv_layer
            model.add_module('custom_head',head)
        else:
            head['criterion'] = criterion
            if head['model_type'].lower() == 'classifier':
                head['output_non_linearity'] = None
            self.num_outputs = head['num_outputs']
            fc = modules[-1]
            try:
                in_features =  fc.in_features
            except:
                in_features = fc.model.out.in_features    
            fc = FC(
                    num_inputs = in_features,
                    num_outputs = head['num_outputs'],
                    layers = head['layers'],
                    model_type = head['model_type'],
                    output_non_linearity = head['output_non_linearity'],
                    dropout_p = dropout_p,
                    criterion = head['criterion'],
                    optimizer_name = None,
                    device = device
                    )
            if adaptive:
                l += meta['adaptive_head']
            else:
                l += meta['normal_head']
            model = nn.Sequential(*l)
            model.add_module('fc',fc)
        self.model = model
        self.head = head
        
        if type(head).__name__ == 'dict':
            print('Model: {}, Setting head: inputs: {} hidden:{} outputs: {}'.format(model_name,
                                                                          in_features,
                                                                          head['layers'],
                                                                          head['num_outputs']))
        else:
            print('Model: {}, Setting head: {}'.format(model_name,type(head).__name__))

    def _get_dropout(self):
        return self.dropout_p
        
    def _set_dropout(self,p=0.45):
        
        if self.model.classifier is not None:
            print('{}: setting head (FC) dropout prob to {:.3f}'.format(self.model_name,p))
            self.model.fc._set_dropout(p=p)

    def get_model_params(self):
        params = super(TransferNetworkImg, self).get_model_params()
        params['class_names'] = self.class_names
        params['num_classes'] = self.num_classes                                              
        params['head'] = self.head
        return params        