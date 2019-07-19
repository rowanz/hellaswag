# hellaswag
HellaSwag: Can a Machine _Really_ Finish Your Sentence?

This repo contains code and data for HellaSwag. If you like this paper, please cite us:

```
@inproceedings{zellers2019hellaswag,
    title={HellaSwag: Can a Machine Really Finish Your Sentence?},
    author={Zellers, Rowan and Holtzman, Ari and Bisk, Yonatan and Farhadi, Ali and Choi, Yejin},
    booktitle ={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
    year={2019}
}
```

## What this repo contains

* The HellaSwag dataset, in `data/`
* Code for Adversarial Filtering, in `adversarial_filtering/`
* Models for HellaSwag, in `hellaswag_models/`

## Getting the environment set up

I used tensorflow and TPUs for this project. My recommendation is to use `ctpu` to start up a VM with access to a `v3-8` TPU. Then, use the following command to install dependencies:
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p ~/conda && \
     rm ~/miniconda.sh && \
     ~/conda/bin/conda install -y python=3.6 tqdm numpy pyyaml scipy ipython mkl mkl-include cython typing h5py pandas && ~/conda/bin/conda clean -ya
     
echo 'export PATH=~/conda/bin:$PATH' >>~/.bashrc
source ~/.bashrc
pip install "tensorflow==1.12.0"
pip install --upgrade google-api-python-client oauth2client
```