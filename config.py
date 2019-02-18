import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = .5
learning_rate = 0.0001
n_iteration = 4000
print_every = 100
save_every = 1
workers = 1
min_word_freq = 10  # Minimum word count threshold for trimming
save_dir = 'models'

# Configure models


train_folder = 'data/ai_challenger_oqmrc_trainingset_20180816'
train_filename = 'ai_challenger_oqmrc_trainingset.json'
train_path = os.path.join(train_folder, train_filename)
valid_folder = 'data/ai_challenger_oqmrc_validationset_20180816'
valid_filename = 'ai_challenger_oqmrc_validationset.json'
valid_path = os.path.join(valid_folder, valid_filename)
test_a_folder = 'data/ai_challenger_oqmrc_testa_20180816'
test_a_filename = 'ai_challenger_oqmrc_testa.json'
test_a_path = os.path.join(test_a_folder, test_a_filename)

# num_train_samples = 8206380
# num_valid_samples = 7034


start_word = '<START>'
stop_word = '<END>'
unknown_word = '<UNK>'
pad_word = '<PAD>'



