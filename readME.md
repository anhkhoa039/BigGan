# BigGAN 

# How to use
### 1. Prepare dataset
```
--- Dataset:  
        |____ class_1_dir  
        |____ class_2_dir  
        |____ class_3_dir 
        ...
        |____ class_n_dir
```

### 1. Install requirements:
```
pip install -r requirements.txt
```
### 2. Config   
Modify the `config.py`  
- Change the `data_dir` variable 

### 3. Run
```
python main.py
```
### 4. Generate Images
- Run the `Inference.ipynb` notebook

# Reference:
- Most of the code is inspired from this notebook: [kaggle notebook](https://www.kaggle.com/code/tikutiku/gan-dogs-starter-biggan)