#!/usr/bin/python

import tensorflow as tf
import numpy as np
import argparse
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import tensorflow.keras.backend

print("If you are generating a lot of sequences, running with GPU available is advised")

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')
#model = load_model('model.h5', custom_objects={'loss': loss})
vocab = ['\n', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'Z', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']

parser = argparse.ArgumentParser(description='tensorflow GRU language model')
parser.add_argument('--lengths', type=int, default=1000, help='length of spike sequence to generate')
parser.add_argument('--seqs', type=int, default=10, help='number of spike sequences to generate')
parser.add_argument('--outfile', type=str, default='outfile.fasta', help='name for output fasta file')
parser.add_argument('--random', choices=('True', 'False'), help='seed text is random if true otherwise leave blank for SARS-CoV-2 start string')
parser.add_argument('--temperature', type=float, default=0.5, help='temperature between 0 and 1, higher temperatures give more surpising results')

args = parser.parse_args()
print("args random", args.random)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

tf.config.list_physical_devices('GPU')
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
batch_size = 1
new_model = build_model(vocab_size=len(vocab),embedding_dim=256,rnn_units=1024,batch_size=batch_size)
weights = loaded_model.get_weights()

new_model.set_weights(weights)
new_model.summary()

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = args.lengths
  print("number to generate is", num_generate)

  # Using character encoding
  input_eval = [char2idx[s] for s in start_string]
  print(input_eval)

  input_eval = tf.expand_dims(input_eval, 0)
  
  # Create list to store the results
  text_generated = []
  
  # Temperature parameter
  # Low temperature values result in more predictable text.
  # Higher temperature values result in more surprising text..
  temperature = args.temperature

  # Batch size == 1 for the model predictions
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      
      predictions = tf.squeeze(predictions, 0)
      
      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))

sequences = []
test_seqs = []
outfile_name = args.outfile + '.fasta'
outfile = open(outfile_name, 'a+')

if args.random == True:
  infile = open('seeds.txt', 'r')
  inseqs = [lin.rstrip() for lin in infile]
  random.shuffle(inseqs)
  for i in range(0,args.seqs):
        # generate a single seed text from list
        seed_text = inseqs.pop()
        print("seed_text")
        
        result = generate_text(new_model, seed_text)
        sequences.append(">{}\n{}".format(str(i), str(result)))
        outfile.write(">{}\n{}\n".format(str(i), str(result)))
else:
    for i in range(0,args.seqs):
       result = generate_text(new_model, 'MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWF')
       print(result)
       sequences.append(">{}\n{}".format(str(i), str(result)))
       outfile.write(">{}\n{}\n".format(str(i), str(result)))


print(len(sequences), sequences)
outfile_name = args.outfile + '.fasta'
outfile = open(outfile_name, 'a+')
for seq in sequences:
  outfile.write(">{}\n{}\n".format(str(i), str(result)))
