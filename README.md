# Food Name and Ingredients Classification using convolutional neural networks and transfer learning in DreamAI.

The pre-trained model has "96% Top-1 Accuracy" on the test and validation sets for the food101 labels which is the best ever recorded on the datatset.  
Up until now the the highest accuracy was 90.52% (https://platform.ai/blog/page/3/new-food-101-sota-with-fastai-and-platform-ais-fast-augmentation-search/)  
And not only does it classify the name, it also predicts the ingredients of the food. This can be used as a building block in several other advanced food and diet industry applications such as automated recipe generation, calorie calculations etc.

### Important Note:

Make sure to first download the pre-trained model from here:  
https://drive.google.com/drive/folders/1Mxw8-AuN2pPT2_6ItGFi_jwCdMOq2OYn?usp=sharing  
Download the whole folder 'ingredients101_model' and paste it in 'mlflow_pretrained_models' without changing its name.  

Download the Food101 dataset here: https://www.vision.ee.ethz.ch/datasets_extra/food-101/  
Extract it and rename the 'images' folder to 'train'. That's all.

## Usage

### Train:

List of parameters: 

  data_path: Full path to the directory of training images. Make sure to keep all images in a folder named 'train' before running script. e.g. if images are in '/training_data/train/', data_path = '/training_data/'.  
  
  size: Size of training images. (Default = 256)  
  
  bs: Batch size. (Default = 32)  
  
  epochs: Number of epochs to train. (Default = 20)  
  
  load: Boolean whether or not to load a pretrained model. (Default = True)  
  
  load_path: Directory where MLflow has saved a previous model for loading later. MLflow will save the models you train in 'mlflow_saved_traing_models'. (Default = '', so it will load from 'mlflow_pretrained_models/ingredients101_model' by default.)   
  
  device: Device to run the code on ('cpu', 'cuda', 'cuda:0' etc.) (Default = 'cpu')
  
#### Command:    

mlflow run . -e train_ingredients -P data_path='/training_data/' -P size=256 -P bs=32 -P epochs=32 -P load=True -P device='cpu'

### Classify

List of parameters: 

  input_path: Path of input image.  
  load_path: Directory where MLflow has saved a previous model for loading later. MLflow will save the models you train in 'mlflow_saved_traing_models'. (Default = '', so it will load from 'mlflow_pretrained_models/ingredients101_model' by default.)  
  
  device: Device to run the code on ('cpu', 'cuda', 'cuda:0' etc.) (Default = 'cpu')  
  
#### Command:    

mlflow run . -e classify_ingredients -P input_path='example_images/bg.jpeg' -P device='cpu'  
