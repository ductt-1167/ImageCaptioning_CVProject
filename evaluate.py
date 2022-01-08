import nltk.translate.bleu_score as bleu
import tensorflow as tf
import pickle
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor

import config_params
import get_data
import model
import inference

folder_image_path = config_params.image_path
file_data_test = config_params.file_data_test
caption_file = config_params.caption_file
checkpoint_path = config_params.checkpoint_path

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


# preprocessing caption for evaluate
def filter_caption(list_caption):
    filters = '!"#$%&()*+.,-/:;=?@[\]^_`{|}~'
    filter_caption = []
    for caption in list_caption:
        caption_split = caption.split(' ')[1:-1]
        middle = []
        for i in caption_split:
            if i not in filters:
                middle.append(i.lower())
        string = " ".join(middle)
        filter_caption.append(string)

    return filter_caption


# evaluate model with BLEU, CIDER, ROUGE, METEOR
# https://www.programcreek.com/python/example/112322/pycocoevalcap.cider.cider.Cider
def compute_score(ref, sample):
    """
        ref, dictionary of reference sentences (id, sentence)
        sample, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        Example:
            reference = {
                1: [
                    u'A train traveling down-tracks next to lights.',
                    u"A blue and silver train next to train's station and trees.",
                    u'A blue train is next to a sidewalk on the rails.',
                    u'A passenger train pulls into a train station.',
                    u'A train coming down the tracks arriving at a station.'],
                2: [
                    u'A large jetliner flying over a traffic filled street.',
                    u'An airplane flies low in the sky over a city street. ',
                    u'An airplane flies over a street with many cars.',
                    u'An airplane comes in to land over a road full of cars',
                    u'The plane is flying over top of the cars']
            }

            candidate = {
                1: [u'train traveling down a track in front of a road'],
                2: [u'plane is flying through the sky'],
            }
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Rouge(), "ROUGE_L"),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(ref, sample)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


reference = {}
candidate = {}
count = 0

for image in list_test_image:
    prediction = inference.predict_caption_from_image(image, encoder, decoder, tokenizer)

    prediction_sentence = " ".join(prediction)
    count += 1

    reference[count] = filter_caption(image_path_to_caption[image])
    candidate[count] = [prediction_sentence]

scores = compute_score(reference, candidate)

print('Bleu_1:', scores['Bleu_1'])
print('Bleu_2:', scores['Bleu_2'])
print('Bleu_3:', scores['Bleu_3'])
print('Bleu_4:', scores['Bleu_4'])
print('Meteor:', scores['METEOR'])
print('Cider:', scores['CIDEr'])
print('Rouge_L:', scores['ROUGE_L'])

