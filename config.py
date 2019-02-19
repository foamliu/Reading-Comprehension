import os

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configure training/optimization
batch_size = 100
print_freq = 100
pickle_file = 'data/data.pkl'

# Configure models
hidden_size = 80

train_folder = 'data/ai_challenger_oqmrc_trainingset_20180816'
train_filename = 'ai_challenger_oqmrc_trainingset.json'
train_path = os.path.join(train_folder, train_filename)
valid_folder = 'data/ai_challenger_oqmrc_validationset_20180816'
valid_filename = 'ai_challenger_oqmrc_validationset.json'
valid_path = os.path.join(valid_folder, valid_filename)
test_a_folder = 'data/ai_challenger_oqmrc_testa_20180816'
test_a_filename = 'ai_challenger_oqmrc_testa.json'
test_a_path = os.path.join(test_a_folder, test_a_filename)
