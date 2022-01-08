"""
    Config the path dataset, shape image, batch size, learning rate, ...
"""

import os


# full data
image_path = 'E:/20211/ThiGiacMayTinh/Project/Dataset/Flickr8k/Flicker8k_Dataset/'
caption_file = 'E:/20211/ThiGiacMayTinh/Project/Dataset/Flickr8k/Flickr8k.lemma.token.txt'

file_data_train = 'E:/20211/ThiGiacMayTinh/Project/Dataset/Flickr8k/Flickr_8k.trainImages.txt'
file_data_val = 'E:/20211/ThiGiacMayTinh/Project/Dataset/Flickr8k/Flickr_8k.devImages.txt'
file_data_test = 'E:/20211/ThiGiacMayTinh/Project/Dataset/Flickr8k/Flickr_8k.testImages.txt'
data_npy_folder = 'data_npy_folder/'

checkpoint_path = 'checkpoint/'
tokenizer_path = 'tokenizer.pickle'

embedding_dim = 256  # embedding dim for encoder
units = 512  # length of hidden state in decoder
top_k = 5000   # caption file has 6604 word
batch_size = 64
epochs = 10
learning_rate = 0.01

# first run, config extraction=True to extract feature and save it with format npy in folder data_npy_folder.
extraction = False


# make init folder
list_folder = os.listdir('.')
if 'data_npy_folder' not in list_folder:
    os .makedirs('data_npy_folder')




