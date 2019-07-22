Ingredients101 is a dataset for ingredients recognition. It consists of the list of most common ingredients for each of the 101 types of food contained in the Food101 dataset [1], making a total of 446 unique ingredients (9 per recipe on average). The dataset was divided in training, validation and test splits making sure that the 101 food types were balanced. We make public the lists of ingredients together with the train/val/test split applied to the images from the Food101 dataset.

In order to download our data, we selected the web platform Yummly (http://www.yummly.com/), we retrieved the closest recipe by name, since there are many ways to cook the same dish. Following we downloaded its associated list of ingredients.

## Contents

This dataset is complementary to the images present in Food101 [1]. Following we describe the files available in each of the folders:

* annotations/<set_split>_images.txt - list of images belonging to each of the data splits, where <set_split> can be either 'train', 'val' or 'test'.

* annotations/<set_split>_labels.txt - list of indices for each of the images in <set_split>_images.txt. Each index points to the corresponing line in ingredients_Recipes5k.txt

* annotations/ingredients.txt - comma separated file that contains, in each line, the list of ingredients present in a certain class of the dataset.

* annotations/classes.txt - list of classes of the dataset Food101.

[1] Bossard L, Guillaumin M, Van Gool L. Food-101–mining discriminative components with random forests. InEuropean Conference on Computer Vision 2014 Sep 6 (pp. 446-461). Springer International Publishing.

## Citation

If you use this dataset for any purpose, please, do not forget to cite the following paper:

```
Marc Bolaños, Aina Ferrà and Petia Radeva. "Food Ingredients Recognition through Multi-label Learning" In Proceedings of the 3rd International Workshop on Multimedia Assisted Dietary Management (ICIAP Workshops), 2017. Pre-print: https://arxiv.org/abs/1707.08816
```

## Contact

If you have any doubt or proposal, please, do not hesitate to contact the first author.

Marc Bolaños
marc.bolanos@ub.edu
http://www.ub.edu/cvub/marcbolanos/
