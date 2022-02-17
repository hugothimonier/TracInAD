cd VAE
python -u train.py -config ../config/arrhythmia.yaml > ./logs/arrhythmia/train.log
python -u validation.py -config ../config/arrhythmia.yaml > ./logs/arrhythmia/validation.log
