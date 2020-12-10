import flair
import torch
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from flair.trainers import ModelTrainer

# define columns
columns = {0 : 'text', 1 : 'ner'}
# directory where the data resides
data_folder = 'glucose_train/'
# initializing the corpus
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file = 'train.txt',
                              test_file = 'test.txt',
                              dev_file = 'dev.txt')

tag_type = 'ner'
# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types = [

    WordEmbeddings('glove'),

    # comment in this line to use character embeddings
    # CharacterEmbeddings(),

    # comment in these lines to use flair embeddings
    # FlairEmbeddings('news-forward'),
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)
tagger : SequenceTagger = SequenceTagger(hidden_size=256,
                                       embeddings=embeddings,
                                       tag_dictionary=tag_dictionary,
                                       tag_type=tag_type,
                                       use_crf=True)

tagger = SequenceTagger.load('ner')
state = tagger._get_state_dict()
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
state['tag_dictionary'] = tag_dictionary
START_TAG: str = "<START>"
STOP_TAG: str = "<STOP>"
state['state_dict']['transitions'] = torch.nn.Parameter(torch.randn(len(tag_dictionary), len(tag_dictionary)))
state['state_dict']['transitions'].detach()[tag_dictionary.get_idx_for_item(START_TAG), :] = -10000
state['state_dict']['transitions'].detach()[:, tag_dictionary.get_idx_for_item(STOP_TAG)] = -10000
num_directions = 2 if tagger.bidirectional else 1
linear_layer =  torch.nn.Linear(tagger.hidden_size * num_directions, len(tag_dictionary))
state['state_dict']['linear.weight'] = linear_layer.weight
state['state_dict']['linear.bias'] = linear_layer.bias
model = SequenceTagger._init_model_with_state_dict(state)
trainer: ModelTrainer = ModelTrainer(model, corpus)

trainer.train('new-resources/taggers/glucose-finetune-ner',
              learning_rate=0.1, 
              mini_batch_size=64,
              max_epochs=300)                           