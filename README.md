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
cd tracinad_wcci2022
virtualenv tracinad_env
source ./tracinad_env/bin/activate
pip install -r requirements.txt
```
Using `conda`:
```
cd tracinad_wcci2022
conda create -n tracinad_env
conda activate tracinad_env
conda install --file requirements.txt
```

## Experiments

To run experiments:
```
cd tracinad_wcci2022
source ./tracinad_env/bin/activate
bash run.sh -d arrhythmia
```
For other datasets, replace arrhythmia by a dataset contained in `[thyroid, arrhythmia, kdd, kddrev]`.

If you use this code, please cite us:
```
@misc{thimonier2022tracinad,
      title={TracInAD: Measuring Influence for Anomaly Detection}, 
      author={Hugo Thimonier and Fabrice Popineau and Arpad Rimmel and Bich-Liên Doan and Fabrice Daniel},
      year={2022},
      eprint={2205.01362},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
