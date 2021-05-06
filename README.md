# PlantNet-300K

This repository contains the code used to produce the benchmark in the paper : .
In order to train a model on the PlantNet-300K dataset, you first have to download the dataset [here](https://doi.org/10.5281/zenodo.4726653).

If you use this work for this research, please cite the paper :

### Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command :

```conda env create -f plantnet_300k_env.yml```

### Training a model

In order to train a model on the PlantNet-300K dataset, run the following command :

```python main.py --lr=0.05 --n_epochs=80 --model=resnet50 --root=path_to_data --save_name_xp=xp1```

 You must provide in the "root" option the path to the train val and test folders. 
 The "save_name_xp" option is the name of the directory where the weights of the model and the results (metrics) will be stored.
 You can check out the different options in the file cli.py.