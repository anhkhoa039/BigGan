# BigGAN 

# How to use
### 1. Prepare dataset
- Dataset link: [Data, annotations, and evaluation code](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)
- Prepare dataset structure as below 
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
python src/main.py
```
### 4. Generate Images
- Run the `Inference.ipynb` notebook

### Visualize
![Images](assert/airplan.png)

### Training Logs
<video width="320" height="240" controls>
  <source src="assert/training_log.mov" type="video/mp4">
  Your browser does not support the video tag.
</video>
# Reference:
- Most of the code is inspired from this notebook: [kaggle notebook](https://www.kaggle.com/code/tikutiku/gan-dogs-starter-biggan)