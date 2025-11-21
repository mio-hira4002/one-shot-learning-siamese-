import os
class config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, "..", "data2")
    batch_size = 10
    epochs = 10
    lr = 0.0005
    logs_dir = "logs"


