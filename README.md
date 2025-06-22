## STGas

### Step
1. Setup STGas
```shell script
python setup.py develop
```

### Train

```shell script
python task/train.py ./config/STGas_train.yml
```

### Test

```shell script
python task/test.py --task val --config ./config/STGas_test.yml --model ./experiment/STGas_train/model_best/model_best.ckpt
```

### Inference

```shell script
python task/inference.py --config ./config/STGas_inference.yml --model ./experiment/STGas_train/model_best/model_best.ckpt
```