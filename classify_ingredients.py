import cv_model
import utils
import argparse
from imports import*

keys = 'input_path','load_model_path','device'
args = {k:v for k,v in zip(keys,sys.argv[1:])}

print()
print('+------------------------------------+')
print('|              Dream AI              |')
print('+------------------------------------+')
print()

device = torch.device(args['device'])
input_path = args['input_path']
img = plt.imread(input_path)

if len(args['load_model_path']) > 0:
    load_model_path = args['load_model_path']
else:
    load_model_path = 'mlflow_pretrained_models/ingredients101_model'

net = mlflow.pytorch.load_model(load_model_path,map_location=device)
net.device = device
net = net.to(device)
batch = utils.get_test_input(paths=[input_path])
batch = batch.to(device)
food,ing = net.classify(batch, thresh=0.4)
ing = ing[0]
if type(ing) == str:
    ing = [ing]
food = ','.join(food).title()
if len(ing) == 0:
    ing = 'Unknown'
else:    
    ing = ', '.join(ing).title()
pred = 'Food Name: {}\nFood Ingredients: {}'.format(food,ing)
plt.imshow(img)
plt.title(pred)
plt.show()

print(pred)
print()

del net
del batch