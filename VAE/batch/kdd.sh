cd VAE
python -u train.py -config ../config/kdd.yaml > ./logs/kdd/train.log
python -u validation.py -config ../config/kdd.yaml > ./logs/kdd/validation.log
