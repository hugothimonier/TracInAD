# TracInAD

Repository containing the code for the experiments of [TracInAD : Measuring Influence for Anomaly Detection](https://arxiv.org/abs/2205.01362) accepted in the proceedings of IJCNN 2022.

## Data

To run experiments, datasets must be downloaded in folder `data`:
```
cd tracinad_wcci2022
bash ./data/data.sh
```
`data.sh` requires `wget` to be installed. For Mac user, replace with `curl` in `data.sh`.

## Requirements 

To run the experiments, use the `requirements.txt` file to install the dependencies. 
Using `virtualenv`:

```
cd TracInAD
virtualenv tracinad_env
source ./tracinad_env/bin/activate
pip install -r requirements.txt
```
Using `conda`:
```
cd TracInAD
conda create -n tracinad_env
conda activate tracinad_env
conda install --file requirements.txt
```

## Experiments

To run experiments:
```
cd TracInAD
source ./tracinad_env/bin/activate
bash run.sh -d arrhythmia
```
For other datasets, replace arrhythmia by a dataset contained in `[thyroid, arrhythmia, kdd, kddrev]`.

If you use this code, please cite us:
```
@inproceedings{Thimonier_2022,
   title={Trac{I}n{AD}: Measuring Influence for Anomaly Detection},
   url={http://dx.doi.org/10.1109/IJCNN55064.2022.9892058},
   DOI={10.1109/ijcnn55064.2022.9892058},
   booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
   publisher={IEEE},
   author={Thimonier, Hugo and Popineau, Fabrice and Rimmel, Arpad and Doan, Bich-Lien and Daniel, Fabrice},
   year={2022},
   month=jul }
```
