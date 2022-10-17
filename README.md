# PlantNet-300K

<p align="middle">
  <img src="/images/1.jpg" width="180" hspace="2"/>
  <img src="/images/2.jpg" width="180" hspace="2"/>
  <img src="/images/3.jpg" width="180" hspace="2"/>
  <img src="/images/4.jpg" width="180" hspace="2"/>
</p>

This repository contains the code used to produce the benchmark in the paper *"Pl@ntNet-300K: a plant image dataset with high label
ambiguity and a long-tailed distribution"*. You can find a link to the paper [here](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/7e7757b1e12abcb736ab9a754ffb617a-Paper-round2.pdf).
In order to train a model on the PlantNet-300K dataset, you first have to download the dataset [here](https://zenodo.org/record/5645731#.Yuehg3ZBxPY). If you are looking for the hyperparameters used in the paper, you can find them in the supplementary material [here](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/7e7757b1e12abcb736ab9a754ffb617a-Abstract-round2.html).

If you use this work for your research, please cite the paper:

    @inproceedings{plantnet-300k,
    author    = {C. Garcin and A. Joly and P. Bonnet and A. Affouard and \JC Lombardo and M. Chouet and M. Servajean and T. Lorieul and J. Salmon},
    booktitle = {NeurIPS Datasets and Benchmarks 2021},
    title     = {{Pl@ntNet-300K}: a plant image dataset with high label ambiguity and a long-tailed distribution},
    year      = {2021},
    }
    
### Dataset Version // Meta-data files

Make sure you download the latest version of the dataset in Zenodo (version 1.1 as in the link above, not 1.0).
The difference lies in the metadata files, the images are the same.
If you wish to download **ONLY** the metadata files (not possible in Zenodo), you will find them [here](https://lab.plantnet.org/seafile/d/bed81bc15e8944969cf6/).

### Pre-trained models

You can find the pre-trained models [here](https://lab.plantnet.org/seafile/d/01ab6658dad6447c95ae/).
To load the pre-trained models, you can simply use the *load_model* function in *utils.py*. For instance, if you want to load the resnet18 weights:

```python
from utils import load_model
from torchvision.models import resnet18

filename = 'resnet18_weights_best_acc.tar' # pre-trained model path
use_gpu = True  # load weights on the gpu
model = resnet18(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename=filename, use_gpu=use_gpu)
```

Note that if you want to fine-tune the model on another dataset, you have to change the last layer. You can find examples in the *get_model* function in *utils.py*. 
### Requirements

Only pytorch, torchvision are necessary for the code to run. 
If you have installed anaconda, you can run the following command:

```conda env create -f plantnet_300k_env.yml```

### Training a model

In order to train a model on the PlantNet-300K dataset, run the following command:

```python main.py --lr=0.01 --batch_size=32 --mu=0.0001 --n_epochs=30 --epoch_decay 20 25 --k 1 3 5 10 --model=resnet18 --pretrained --seed=4 --image_size=256 --crop_size=224 --root=path_to_data --save_name_xp=xp1```

 You must provide in the "root" option the path to the train val and test folders. 
 The "save_name_xp" option is the name of the directory where the weights of the model and the results (metrics) will be stored.
 You can check out the different options in the file cli.py.