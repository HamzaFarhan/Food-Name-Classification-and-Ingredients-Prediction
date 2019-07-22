import utils
import cv_model
import data_processing
from imports import*


# keys = 'data_path','csv_name','image_size','bs','epochs','load_model','load_model_path','device'
keys = 'data_path','image_size','bs','epochs','load_model','load_model_path','device'
args = {k:v for k,v in zip(keys,sys.argv[1:])}

print()
print('+------------------------------------+')
print('|              Dream AI              |')
print('+------------------------------------+')
print()

data_path = Path(args['data_path'])
anno_path = Path('ingredient101/annotations')
train_name = 'train'
val_name = 'val'
test_name = 'test'
dp_name = 'DP_food.pkl'
train_path = data_path/train_name
dp_path = data_path/dp_name
ing_file = anno_path/'ingredients.txt'

image_size = int(args['image_size'])
image_size = (image_size,image_size)
bs = int(args['bs'])
epochs = int(args['epochs'])

if not dp_path.exists():
    DP = data_processing.DataProcessor(data_path,tr_name=train_name,setup_data=True)
    data_processing.save_obj(dp_path,DP)

DP = data_processing.load_obj(dp_path)
data_dict = DP.data_dict

with open(ing_file) as f:
    ing = f.readlines()
ing = [n.split('\n')[0] for n in ing]

df = pd.concat([data_dict['data_dfs'][train_name],data_dict['data_dfs'][val_name],
                data_dict['data_dfs'][test_name]])
labels = list(df.Label)
ingredients = [ing[l] for l in labels]
df['Ingredients'] = ingredients

targets = list(df.Ingredients.apply(lambda x: str(x)))
split_targets = [t.split(',') for t in targets]
dai_onehot,onehot_classes = data_processing.one_hot(split_targets,True)
df['Ingredients'] = [torch.from_numpy(x).type(torch.FloatTensor) for x in dai_onehot]

tr_df = len(data_dict['data_dfs'][train_name])
val_df = len(data_dict['data_dfs'][val_name])
test_df = len(data_dict['data_dfs'][test_name])
data_dict['data_dfs'][train_name] = df[:tr_df]
data_dict['data_dfs'][val_name] = df[tr_df:tr_df+val_df]
data_dict['data_dfs'][test_name] = df[val_df+tr_df:]
ingredient_names = onehot_classes
num_ingredients = len(ingredient_names)

# This is just a precaution for when normalizing the data
# This is the percentage of the data to use when calculating the mean and standard deviation
# so that it doesn't exceed our memory capacity
# 2500000 is a safe number that usually works for me with a 64GB RAM. It will differ for your system

stats_percentage = min(1.0,2500000/(image_size[0] * len(data_dict['data_dfs'][train_name])))

# Create dataset and dataloaders
# This may take a while because it will normalize according to the whole dataset

sets,loaders,sizes = DP.get_data(data_dict, image_size, bs = bs,
                                 dataset=data_processing.dai_image_csv_dataset_food,
                                 balance = False, stats_percentage = stats_percentage, num_workers = 16,
                                 tfms = [transforms.RandomHorizontalFlip()],
                                 img_mean=DP.img_mean,img_std=DP.img_std)
data_processing.save_obj(dp_path,DP)

print_every = sizes[train_name]//3//bs
device = torch.device(args['device'])

best_model_file = 'ingredients.pth'
print("MLflow will save the models in 'mlflow_saved_training_models'")

if args['load_model'] == True:
    load_model_path = 'mlflow_pretrained_models/ingredients101_model'
    if len(args['load_model_path']) > 0:
        load_model_path = args['load_model_path']
    net = mlflow.pytorch.load_model(load_model_path,map_location=device)
    net.best_model_file = best_model_file
    net.freeze()
    for n,p in net.model.named_parameters():
        if n[0] != '0':
            p.requires_grad = True
else:
    print()
    optim = 'adadelta'
    net = cv_model.FoodIngredients(model_name = 'densenet',model_type = 'food',
                         optimizer_name = optim,
                         criterion1 = nn.CrossEntropyLoss(),criterion2=nn.BCEWithLogitsLoss(),
                         device = device,best_model_file = best_model_file,
                         class_names = data_dict['class_names'],num_classes = data_dict['num_classes'],
                         ingredient_names = ingredient_names,num_ingredients = num_ingredients,      
                         dropout_p = 0.5,add_extra = True,
                         head1 = {'num_outputs':data_dict['num_classes'],'model_type':'classifier',
                                 'layers':[],'output_non_linearity':None},
                         head2 = {'num_outputs':num_ingredients,'model_type':'multi_label_classifier',
                                 'layers':[],'output_non_linearity':None}
                              )
    net.unfreeze()

net.device = device    
net = net.to(device)
lr = net.find_lr(loaders[train_name],plot=False)
net.fit(loaders[train_name],loaders[val_name],epochs=epochs,print_every=print_every)
del net
