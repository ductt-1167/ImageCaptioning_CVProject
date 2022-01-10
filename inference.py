import tensorflow as tf
import numpy as np

import get_data


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))  # input for inceptionV3
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output  # 8x8x2048

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


# predict caption from a image
def predict_caption_from_image(image, encoder, decoder, tokenizer):
    max_length = get_data.max_length
    # Init hidden state
    hidden = decoder.reset_state(batch_size=1)

    # Using  InceptionV3 to extract feature
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))
    # Encode
    features = encoder(img_tensor_val)

    # Start word: <start>
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):

        # decoder
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)

        # Predict token of next word
        predicted_id = list(predictions[0].numpy()).index(max(predictions[0].numpy()))
        # predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        # If predict token is <end> --> end sentence --> break
        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        result.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    return result


