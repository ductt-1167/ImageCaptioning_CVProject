import tensorflow as tf

""" ENCODER
    Get feature maps in this network for encoder process 
"""


class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        # Input shape: (batch_size, 64, features_shape)
        x = self.fc(x)
        x = self.dropout(x)
        return x


""" DECODER: predict word with Attention 
    Use hidden front to make connections with important image areas (features from CNN output)
    This hidden vector includes important information from the beginning of the sentence to the position ahead
"""


# https://machinelearningmastery.com/the-bahdanau-attention-mechanism/
# https://blog.floydhub.com/attention-mechanism/
class Bahdanau_Attention(tf.keras.Model):
    def __init__(self, units):
        super(Bahdanau_Attention, self).__init__()
        self.dense1 = tf.keras.layers.Dense(units)
        self.dense2 = tf.keras.layers.Dense(units)
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.softmax = tf.keras.layers.Softmax(axis=1)
        self.score = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        attention_hidden_layer = (tf.nn.tanh(self.dropout(self.dense1(features)) +
                                             self.dropout(self.dense2(hidden_with_time_axis))))

        score = self.score(attention_hidden_layer)
        attention_weights = self.softmax(score)

        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(self.units,
                                         return_sequences=True,
                                         return_state=True)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True)
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size, activation='softmax')

        self.attention = Bahdanau_Attention(self.units)

    def call(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        output, state, _ = self.lstm(x)
        x = self.fc1(output)
        x = self.dropout(x)
        x = tf.reshape(x, (-1, output.shape[2]))
        x = self.fc2(x)
        x = self.dropout(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
