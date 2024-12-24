
import torch
import time
class opt:
    num_classes = 30
    batch_size = 32
    NUM_WORKERS = 12
    EPOCHS = 300
    img_height, img_width = (64,64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMA = False
    LABEL_NOISE = False
    LABEL_NOISE_PROB = 0.1

    TIME_LIMIT = 32400 - 60*10
    # start_time = time.time()
    data_dir = 'data/dataset'
