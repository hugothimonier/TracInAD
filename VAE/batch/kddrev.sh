cd VAE
python -u train.py -config ../config/kddrev.yaml > ./logs/kddrev/train.log
python -u validation.py -config ../config/kddrev.yaml > ./logs/kddrev/validation.log
