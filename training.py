import collections
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import time

import get_data
import config_params
import model


tf.config.run_functions_eagerly(True)

# hyper-parameters
BATCH_SIZE = config_params.batch_size
BUFFER_SIZE = 1000
embedding_dim = config_params.embedding_dim
units = config_params.units
vocab_size = config_params.top_k + 1
features_shape = 2048
attention_features_shape = 64

start_epoch = 0
EPOCHS = config_params.epochs
continue_train = False  # for continue training

tokenizer = get_data.get_tokenizer()


def create_dataset(file_dataset):
    # get the captions and images
    captions, img_name_vector = get_data.get_list_data_each_set(file_dataset)
    cap_vector, max_length = get_data.preprocessing_caption(tokenizer, captions)

    img_to_cap_vector = collections.defaultdict(list)

    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # get feature vector
    list_image_paths, list_image_name = get_data.extract_feature_from_images(img_name_vector, extract=config_params.extraction)

    images_name = []
    captions_vector = []
    for index in range(len(list_image_paths)):
        cap_len = len(img_to_cap_vector[list_image_paths[index]])
        for _ in range(cap_len):
            images_name.append(list_image_name[index])
        captions_vector.extend(img_to_cap_vector[list_image_paths[index]])

    # Load the numpy files
    def map_func(img_name, cap):
        img_tensor = np.load(config_params.data_npy_folder + img_name.decode('utf-8') + '.npy')
        return img_tensor, cap

    dataset = tf.data.Dataset.from_tensor_slices((images_name, captions_vector))

    # Load ảnh từ file

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset, len(img_name_vector)


# get training dataset
training_dataset, len_data_train = create_dataset(config_params.file_data_train)
validation_dataset, len_data_val = create_dataset(config_params.file_data_val)

print("Training dataset: {} samples".format(len_data_train))
print("Validation dataset: {} samples".format(len_data_val))

num_steps_train = len_data_train // BATCH_SIZE
num_steps_val = len_data_val // BATCH_SIZE

# Init encoder and decoder
encoder = model.CNN_Encoder(embedding_dim)
decoder = model.RNN_Decoder(units=units, embedding_dim=embedding_dim, vocab_size=vocab_size)

# save loss to plot
loss_train_plot = []
loss_val_plot = []

# Init optimizer and loss function
optimizer = tf.keras.optimizers.Adam()
loss_fc = tf.keras.losses.SparseCategoricalCrossentropy()


def loss_function(y_true, y_pred):
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_fc(y_true, y_pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


# Training
@tf.function
def train_step(img_tensor, target):
    loss = 0

    # Init hidden state
    hidden = decoder.reset_state(batch_size=target.shape[0])

    # Start predict with beginning token: <start>. size: (batch_size, 1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    # init predict to compute acc
    middle = np.zeros((target.shape[0], vocab_size))
    middle[:, tokenizer.word_index['<start>']] = 1

    # Using Gradient Tape
    with tf.GradientTape() as tape:
        # Encoder
        features = encoder(img_tensor)

        # Loop to end the caption sentence, skip <start>
        for i in range(1, target.shape[1]):
            # Decoder: Take feature và previous hidden with present word  to decoder --> output: next word and present hidden
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            # Computer loss
            loss += loss_function(target[:, i], predictions)

            # update present word
            dec_input = tf.expand_dims(target[:, i], 1)

    # compute average loss
    total_loss = (loss / int(target.shape[1]))

    # Update weights and optimizer
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


@tf.function
def val_step(img_tensor, target):
    loss = 0

    # Init hidden state
    hidden = decoder.reset_state(batch_size=target.shape[0])

    # Start predict with beginning token: <start>. size: (batch_size, 1)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * target.shape[0], 1)

    # init predict to compute acc
    middle = np.zeros((target.shape[0], vocab_size))
    middle[:, tokenizer.word_index['<start>']] = 1

    # Encoder
    features = encoder(img_tensor)

    # Loop to end the caption sentence, skip <start>
    for i in range(1, target.shape[1]):
        # Decoder: Take feature và previous hidden with present word  to decoder --> output: next word and present hidden
        predictions, hidden, _ = decoder(dec_input, features, hidden)

        # Computer loss
        loss += loss_function(target[:, i], predictions)

        # update present word
        dec_input = tf.expand_dims(target[:, i], 1)

    # compute average loss
    total_loss = (loss / int(target.shape[1]))

    return loss, total_loss


# checkpoint
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
manager = tf.train.CheckpointManager(ckpt, config_params.checkpoint_path, max_to_keep=5)

# load model if continue training (has checkpoint)
if continue_train:
    ckpt.restore(manager.latest_checkpoint)

for epoch in range(start_epoch, start_epoch+EPOCHS):
    start = time.time()
    total_loss_train = total_loss_val = 0

    # training for each batch
    for (batch, (img_tensor, target)) in enumerate(training_dataset):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss_train += t_loss

        # Print loss
        if batch % 100 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(
                epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

    # Validation for each batch
    for (batch, (img_tensor, target)) in enumerate(validation_dataset):
        batch_loss, t_loss = val_step(img_tensor, target)
        total_loss_val += t_loss

    print('Epoch {} --- Loss_train {:.6f}  --- Loss_val {:.6f}'.format(
        epoch + 1,
        total_loss_train / num_steps_train,
        total_loss_val / num_steps_val))

    loss_train_plot.append(total_loss_train / num_steps_train)
    loss_val_plot.append(total_loss_val / num_steps_val)

    manager.save()

    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

# plot loss
x = np.arange(1, EPOCHS+1)
plt.plot(x, loss_train_plot, label='train_loss')
plt.plot(x, loss_val_plot, label='val_loss')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()