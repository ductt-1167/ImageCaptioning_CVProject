import tensorflow as tf
from tqdm import tqdm
import collections
import numpy as np
import os
import pickle

import config_params

folder_image_path = config_params.image_path
caption_file = config_params.caption_file

bs_inception_v3 = 64


def get_image_path_to_caption(image_path, caption_file):
    """
    get all imgs with their captions
    :param image_path: path of image folder
    :param caption_file: file caption
    :return: a collection
    """
    file = open(caption_file, 'r')
    text = file.read()
    file.close()

    lines = text.split('\n')
    image_path_to_caption = collections.defaultdict(list)
    list_caption = []

    for line in lines[:-1]:
        img_name, caption = line.split('\t')
        img_name = image_path + img_name.split('#')[0]
        caption = f"<start> {caption} <end>"
        image_path_to_caption[img_name].append(caption)
        list_caption.append(caption)

    return image_path_to_caption, list_caption


image_path_to_caption, list_caption = get_image_path_to_caption(folder_image_path, caption_file)


def get_list_data_each_set(file_data):
    """
    Get list caption and list image path for each set (training, validation, test)
    :param file_data: file data train, file data val or file data test
    :return: list caption and list image path
    """

    def get_data(list_image_path):
        captions = []  # save captions list
        img_name_vector = []  # save image path list

        for image_path in list_image_path:
            caption_list = image_path_to_caption[image_path]
            captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))

        return captions, img_name_vector

    def get_list_image_path(file_data):
        f = open(file_data, 'r')
        data = f.read()
        data = data.split('\n')
        list_image_path = []
        for i in data:
            list_image_path.append(folder_image_path + i)

        return list_image_path

    # Training set
    list_image_path = get_list_image_path(file_data)
    captions, img_name_vector = get_data(list_image_path)

    return captions, img_name_vector


def extract_feature_from_images(list_image_path, extract=False):
    """
    Using pretrained model Inception to extract feature from image
    :param extract:
    :param list_image_path:
    :return:
    """
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    input_layer = image_model.input
    output_layer = image_model.layers[-1].output  # 8x8x2048

    image_features_extract_model = tf.keras.Model(input_layer, output_layer)

    # Load each image with tf.io
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (299, 299))  # input shape image for inceptionV3: [299, 299, 3]
        img = tf.keras.applications.inception_v3.preprocess_input(img)

        return img, image_path

    encode_set = sorted(set(list_image_path))

    list_image_name = []
    if not extract:
        for img in encode_set:
            list_image_name.append(os.path.basename(img))
    else:
        image_dataset = tf.data.Dataset.from_tensor_slices(encode_set)
        image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(bs_inception_v3)

        for img, path in tqdm(image_dataset):
            batch_features = image_features_extract_model(img)  # shape (bs, 8, 8, 2048)
            batch_features = tf.reshape(batch_features,
                                        (batch_features.shape[0], -1, batch_features.shape[3]))  # shape (bs, 64, 2048)
            for bf, p in zip(batch_features, path):
                path_of_feature = config_params.data_npy_folder + os.path.basename(tf.compat.as_str_any(p.numpy()))
                np.save(path_of_feature, bf.numpy())
                list_image_name.append(os.path.basename(tf.compat.as_str_any(p.numpy())))

    return encode_set, list_image_name


# ============================= Preprocessing for caption text ========================================
# note that build vocabulary for all caption (train+val+test)
# Find max length of captions in dataset
def get_max_length(tensor):
    return max(len(t) for t in tensor)


def get_tokenizer():
    file = open(config_params.caption_file, 'r')
    text = file.read()
    file.close()

    lines = text.split('\n')

    list_caption = []
    for line in lines[:-1]:
        img_name, caption = line.split('\t')
        caption = f"<start> {caption} <end>"
        list_caption.append(caption)

    top_k = config_params.top_k
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    # The word that appears the most will have the smallest id
    tokenizer.fit_on_texts(list_caption)

    #  Add padding
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer


# compute max_length
tokenizer = get_tokenizer()
# saving tokenizer
with open(config_params.tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

seqs = tokenizer.texts_to_sequences(list_caption)
max_length = get_max_length(seqs)


def preprocessing_caption(tokenizer, text_captions):
    # Transform word to vector
    train_seqs = tokenizer.texts_to_sequences(text_captions)

    # Make the sentences have the same length
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs,
                                                               padding="post")  # post means adding in the back

    return cap_vector, max_length
