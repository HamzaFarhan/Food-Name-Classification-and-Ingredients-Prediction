from imports import *
from utils import *
import parallel

class Classifier():
    def __init__(self,class_names):
        self.class_names = class_names
        self.class_correct = defaultdict(int)
        self.class_totals = defaultdict(int)

    def update_accuracies(self,outputs,labels):
        _, preds = torch.max(torch.exp(outputs), 1)
        correct = np.squeeze(preds.eq(labels.data.view_as(preds)))
        for i in range(labels.shape[0]):
            label = labels.data[i].item()
            self.class_correct[label] += correct[i].item()
            self.class_totals[label] += 1

    def get_final_accuracies(self):
        accuracy = (100*np.sum(list(self.class_correct.values()))/np.sum(list(self.class_totals.values())))
        try:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in self.class_names.keys() if self.class_totals[i] > 0]
        except:
            class_accuracies = [(self.class_names[i],100.0*(self.class_correct[i]/self.class_totals[i])) 
                                 for i in range(len(self.class_names)) if self.class_totals[i] > 0]
        return accuracy,class_accuracies

class MultiLabelClassifier():
    def __init__(self,class_names):
        self.class_names = class_names
        self.class_correct = defaultdict(int)
        self.class_totals = defaultdict(int)
        
    def accuracies(self,outputs,labels,thresh = 0.5):    
        
        accuracies = defaultdict(int)
        # outputs = torch.Tensor([[3,1,5,0,2],[0,9,2,7,3]])
        # labels = torch.Tensor([[0.,0.,1.,1,0.],[0.,1.,0.,1,0.]])
        preds = (outputs >= thresh).float()
        correct = (preds==1)*(labels==1)
        for i in range(labels.shape[0]):
            label = labels.data[i]
            label = label.nonzero().squeeze(1)
            for l in label:
                self.class_correct[l.item()] += correct[i][l].item()
                self.class_totals[l.item()] += 1.0
        for c in self.class_correct.keys():
            accuracies[self.class_names[c]] = 100*(self.class_correct[c]/self.class_totals[c])
        return accuracies    

class Network(nn.Module):
    def __init__(self,device=None):
        super().__init__()
        self.parallel = False
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(self.device)

    def forward(self,x):
        pass
    def compute_loss(self,outputs,labels):
        return [self.criterion(outputs,labels)]

    def fit(self,trainloader,validloader,epochs=2,print_every=10,validate_every=1,save_best_every=1):

        optim_path = Path(self.best_model_file)
        optim_path = optim_path.stem + '_optim' + optim_path.suffix
        with mlflow.start_run() as run:
            for epoch in range(epochs):
                self.model = self.model.to(self.device)
                mlflow.log_param('epochs',epochs)
                mlflow.log_param('lr',self.optimizer.param_groups[0]['lr'])
                mlflow.log_param('bs',trainloader.batch_size)
                print('Epoch:{:3d}/{}\n'.format(epoch+1,epochs))
                epoch_train_loss =  self.train_((epoch,epochs),trainloader,self.optimizer,print_every)  
                        
                if  validate_every and (epoch % validate_every == 0):
                    t2 = time.time()
                    eval_dict = self.evaluate(validloader)
                    epoch_validation_loss = eval_dict['final_loss']
                    if self.parallel:
                        try:
                            epoch_train_loss = epoch_train_loss.item()
                            epoch_validation_loss = epoch_validation_loss.item()
                        except:
                            pass  
                    mlflow.log_metric('Train Loss',epoch_train_loss)
                    mlflow.log_metric('Validation Loss',epoch_validation_loss)
                    
                    time_elapsed = time.time() - t2
                    if time_elapsed > 60:
                        time_elapsed /= 60.
                        measure = 'min'
                    else:
                        measure = 'sec'    
                    print('\n'+'/'*36+'\n'
                            f"{time.asctime().split()[-2]}\n"
                            f"Epoch {epoch+1}/{epochs}\n"    
                            f"Validation time: {time_elapsed:.6f} {measure}\n"    
                            f"Epoch training loss: {epoch_train_loss:.6f}\n"                        
                            f"Epoch validation loss: {epoch_validation_loss:.6f}"
                        )
                    if self.model_type == 'classifier':# or self.num_classes is not None:
                        epoch_accuracy = eval_dict['accuracy']
                        mlflow.log_metric('Validation Accuracy',epoch_accuracy)
                        print("Validation accuracy: {:.3f}".format(epoch_accuracy))
                        # print('\\'*36+'/'*36+'\n')
                        print('\\'*36+'\n')
                        if self.best_accuracy == 0. or (epoch_accuracy >= self.best_accuracy):
                            print('\n**********Updating best accuracy**********\n')
                            print('Previous best: {:.3f}'.format(self.best_accuracy))
                            print('New best: {:.3f}\n'.format(epoch_accuracy))
                            print('******************************************\n')
                            self.best_accuracy = epoch_accuracy
                            mlflow.log_metric('Best Accuracy',self.best_accuracy)
                            optim_path = Path(self.best_model_file)
                            optim_path = optim_path.stem + '_optim' + optim_path.suffix
                            torch.save(self.model.state_dict(),self.best_model_file)
                            torch.save(self.optimizer.state_dict(),optim_path)     
                            mlflow.pytorch.log_model(self,'mlflow_logged_models')
                            curr_time = str(datetime.now())
                            curr_time = '_'+curr_time.split()[1].split('.')[0]
                            mlflow_save_path = Path('mlflow_saved_training_models')/\
                                               (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_accuracy,2)),str(epoch)+curr_time))
                            mlflow.pytorch.save_model(self,mlflow_save_path)
                    else:
                        print('\\'*36+'\n')
                        if self.best_validation_loss == None or (epoch_validation_loss <= self.best_validation_loss):
                            print('\n**********Updating best validation loss**********\n')
                            if self.best_validation_loss is not None:
                                print('Previous best: {:.7f}'.format(self.best_validation_loss))
                            print('New best loss = {:.7f}\n'.format(epoch_validation_loss))
                            print('*'*49+'\n')
                            self.best_validation_loss = epoch_validation_loss
                            mlflow.log_metric('Best Loss',self.best_validation_loss)
                            optim_path = Path(self.best_model_file)
                            optim_path = optim_path.stem + '_optim' + optim_path.suffix
                            torch.save(self.model.state_dict(),self.best_model_file)
                            torch.save(self.optimizer.state_dict(),optim_path)     
                            mlflow.pytorch.log_model(self,'mlflow_logged_models')
                            curr_time = str(datetime.now())
                            curr_time = '_'+curr_time.split()[1].split('.')[0]
                            mlflow_save_path = Path('mlflow_saved_training_models')/\
                                (Path(self.best_model_file).stem+'_{}_{}'.format(str(round(epoch_validation_loss,3)),str(epoch)+curr_time))
                            mlflow.pytorch.save_model(self,mlflow_save_path)
                        
                    self.train()
        torch.cuda.empty_cache()
        try:
            print('\nLoading best model\n')
            self.model.load_state_dict(torch.load(self.best_model_file))
            self.optimizer.load_state_dict(torch.load(optim_path))
            os.remove(self.best_model_file)
            os.remove(optim_path)
        except:
            pass    

    def train_(self,e,trainloader,optimizer,print_every):

        epoch,epochs = e
        self.train()
        t0 = time.time()
        t1 = time.time()
        batches = 0
        running_loss = 0.
        for data_batch in trainloader:
            inputs,labels = data_batch[0],data_batch[1]
            batches += 1
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
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
                inputs, labels = data_batch[0],data_batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
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
   
    def classify(self,inputs,thresh = 0.4,class_names = None):#,show = False,mean = None,std = None):
        if class_names is None:
            class_names = self.class_names
        outputs = self.predict(inputs)
        if self.model_type == 'classifier':
            try:    
                _, preds = torch.max(torch.exp(outputs), 1)
            except:
                _, preds = torch.max(torch.exp(outputs.unsqueeze(0)), 1)
        else:
            outputs = outputs.sigmoid()
            preds = (outputs >= thresh).nonzero().squeeze(1)
        class_preds = [str(class_names[p]) for p in preds]
        # imgs = batch_to_imgs(inputs.cpu(),mean,std)
        # if show:
            # plot_in_row(imgs,titles=class_preds)
        return class_preds

    def predict(self,inputs):
        self.eval()
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.forward(inputs)
        return outputs

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
            inputs,labels = data_batch[0],data_batch[1]
            inputs = inputs.to(self.device)           
            labels = labels.to(self.device)
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
                
    def set_criterion(self, criterion):
        if criterion:
            self.criterion = criterion
        
    def set_optimizer(self,params,optimizer_name='adam',lr=0.003):
        if optimizer_name:
            if optimizer_name.lower() == 'adam':
                print('Setting optimizer: Adam')
                self.optimizer = optim.Adam(params,lr=lr)
                self.optimizer_name = optimizer_name
            elif optimizer_name.lower() == 'sgd':
                print('Setting optimizer: SGD')
                self.optimizer = optim.SGD(params,lr=lr)
            elif optimizer_name.lower() == 'adadelta':
                print('Setting optimizer: AdaDelta')
                self.optimizer = optim.Adadelta(params)       
            
    def set_model_params(self,
                         criterion = nn.CrossEntropyLoss(),
                         optimizer_name = 'sgd',
                         lr = 0.01,
                         dropout_p = 0.45,
                         model_name = 'resnet50',
                         model_type = 'classifier',
                         best_accuracy = 0.,
                         best_validation_loss = None,
                         best_model_file = 'best_model_file.pth'):        
        self.set_criterion(criterion)
        self.optimizer_name = optimizer_name
        self.set_optimizer(self.parameters(),optimizer_name,lr=lr)
        self.lr = lr
        self.dropout_p = dropout_p
        self.model_name =  model_name
        self.model_type = model_type
        self.best_accuracy = best_accuracy
        self.best_validation_loss = best_validation_loss
        self.best_model_file = best_model_file
    
    def get_model_params(self):
        params = {}
        params['device'] = self.device
        params['model_type'] = self.model_type
        params['model_name'] = self.model_name
        params['optimizer_name'] = self.optimizer_name
        params['criterion'] = self.criterion
        params['lr'] = self.lr
        params['dropout_p'] = self.dropout_p
        params['best_accuracy'] = self.best_accuracy
        params['best_validation_loss'] = self.best_validation_loss
        params['best_model_file'] = self.best_model_file
        return params
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
