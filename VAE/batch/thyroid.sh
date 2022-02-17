cd VAE
python -u train.py -config ../config/thyroid.yaml > ./logs/thyroid/train.log
python -u validation.py -config ../config/thyroid.yaml > ./logs/thyroid/validation.log
