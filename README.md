# Grapg Transformer model for Real Estate Prediction
Implementation of [GT paper](https://arxiv.org/abs/2012.09699) for real estate dataset



### Environment
``` 
conda create -n $ENV_NAME$ python=3.5.2
conda activate $ENV_NAME$

# CUDA 11.3
pip install torch==1.5.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 
# Or, CUDA 10.2 
pip install torch==1.5.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102 
pip install -r requirements.txt
```

### Preprocess
Create adjacency matrix and one hot encoding
```
python preprocess.py --data_path $path_to_csv_dataset --create_adj $wheither_to_produce_adj
```


### Training the model
Specify the parameters in required class in ./config/config_file.json
```
python main.py --config $config_file.json --visible_gpus gpus_to_use
```

### Results
| Model | MSE | MAE |
|-------|--|--|--|


### TODO
- [x] Make positional encodings work
- [ ] Make the yearly and montly training
- [ ] Combine GT network with LSTM
- [ ] Configure hyperparameters


### Reference
* Official Implementation: https://github.com/graphdeeplearning/graphtransformer
