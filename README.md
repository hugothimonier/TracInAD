Repository containing the code for the experiments presented in `TracInAD : Measuring Influence for Anomaly Detection.`

To run experiments, datasets must be downloaded in folder `data`:
```
cd tracinad_wcci2022
bash ./data/data.sh
```
`data.sh` requires `wget` to be installed. For Mac user, replace with `curl` in `data.sh`.


To run the experiments, use the `requirements.txt` file to install the dependencies. Using `virtualenv`:

```
cd tracinad_wcci2022
virtualenv tracinad_env
source ./tracinad_env/bin/activate
pip install -r requirements.txt
```

To run experiments:
```
cd tracinad_wcci2022
source ./tracinad_env/bin/activate
cd VAE
python -u train.py -config ./config/arrhythmia.yaml > ./logs/arrhythmia/train.log
python -u validation.py -config ./config/arrhythmia.yaml > ./logs/arrhythmia/train.log
```
For other datasets, replace arrhythmia by a dataset contained in `[thyroid, arrhythmia, kdd, kddrev]`.
