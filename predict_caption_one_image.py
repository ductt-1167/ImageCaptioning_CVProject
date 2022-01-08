from PIL import Image, ImageDraw
import tensorflow as tf
import pickle


import config_params
import get_data
import model
import inference


def show_image(path_image):
    # read the image
    im = Image.open(path_image)

    # # add caption predict
    # d = ImageDraw.Draw(im)
    # d.text((10, 10), "Predict caption:" + caption, fill=(255, 255, 0))
    # show image
    im.show()


folder_image_path = config_params.image_path
file_data_test = config_params.file_data_test
caption_file = config_params.caption_file
checkpoint_path = "E:/20211\ThiGiacMayTinh\Project\checkpoints\epoch_21"#config_params.checkpoint_path

# init and load model
embedding_dim = config_params.embedding_dim
units = config_params.units
vocab_size = config_params.top_k + 1

encoder = model.CNN_Encoder(embedding_dim)
decoder = model.RNN_Decoder(units=units, embedding_dim=embedding_dim, vocab_size=vocab_size)
optimizer = tf.keras.optimizers.Adam()

# checkpoints
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)  # keep max 5 last checkpoint

# load checkpoint
ckpt.restore(manager.latest_checkpoint)

# load tokenizer
with open(config_params.tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)


# get testing dataset
def get_list_image_path(file_data):
    f = open(file_data, 'r')
    data = f.read()
    data = data.split('\n')
    list_image_path = []
    for i in data:
        list_image_path.append(folder_image_path + i)

    return list_image_path


list_test_image = get_list_image_path(file_data_test)
print("Testing dataset: {} samples".format(len(list_test_image)))
image_path_to_caption, _ = get_data.get_image_path_to_caption(folder_image_path, caption_file)


for image in list_test_image[30:35]:
    prediction = inference.predict_caption_from_image(image, encoder, decoder, tokenizer)
    prediction_sentence = " ".join(prediction)
    show_image(image)
    print(image_path_to_caption[image])
    print(prediction_sentence)
    print('========================')
