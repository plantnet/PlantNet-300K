# PlantNet-300K

This repository contains the code used to produce the benchmark in the paper *"Pl@ntNet-300K: a plant image dataset with high label
ambiguity and a long-tailed distribution"*. You can find a link to the paper [here](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf).
In order to train a model on the PlantNet-300K dataset, you first have to download the dataset [here](https://doi.org/10.5281/zenodo.4726653).

If you use this work for this research, please cite the paper :

    @inproceedings{garcin2021pl,
      title={Pl@ ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed distribution},
      author={Garcin, Camille and Joly, Alexis and Bonnet, Pierre and Lombardo, Jean-Christophe and Affouard, Antoine and Chouet, Mathias and Servajean, Maximilien and Salmon, Joseph and Lorieul, Titouan},
      booktitle={NeurIPS 2021-35th Conference on Neural Information Processing Systems},
      year={2021}
    }

### Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command :

```conda env create -f plantnet_300k_env.yml```

### Training a model

In order to train a model on the PlantNet-300K dataset, run the following command :

```python main.py --lr=0.05 --n_epochs=80 --k 1 3 5 10 --model=resnet50 --root=path_to_data --save_name_xp=xp1```

 You must provide in the "root" option the path to the train val and test folders. 
 The "save_name_xp" option is the name of the directory where the weights of the model and the results (metrics) will be stored.
 You can check out the different options in the file cli.py.