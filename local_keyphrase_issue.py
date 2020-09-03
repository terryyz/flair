from flair.data import Corpus
from flair.datasets import SEMEVAL2010, SEMEVAL2017, INSPEC
from flair.embeddings import FlairEmbeddings, WordEmbeddings, StackedEmbeddings, TransformerWordEmbeddings

# 1. get the corpus
for _corpus, corpus_name in [(SEMEVAL2017, "SEMEVAL2017"), (SEMEVAL2010, "SEMEVAL2010"), (INSPEC, "INSPEC")]:
    corpus: Corpus = _corpus()
    print(corpus)

    # 2. what tag do we want to predict?
    tag_type = 'keyword'

    # 3. make the tag dictionary from the corpus
    tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
    print(tag_dictionary)

    # 4. initialize embeddings
    embedding_types = [
        ([WordEmbeddings('glove'),
        FlairEmbeddings('news-forward'),
        FlairEmbeddings('news-backward'),],
         'flair'),
        ([TransformerWordEmbeddings('bert-base-uncased')],
         'bert'),
        ([TransformerWordEmbeddings('distilbert-base-uncased')],
         'distilbert'),
        ([TransformerWordEmbeddings('roberta-large')],
         'roberta'),
    ]

    for embedding_type, embedding_name in embedding_types:
        embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_type)

        # 5. initialize sequence tagger
        from flair.models import SequenceTagger

        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=embeddings,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True)

        # 6. initialize trainer
        from flair.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        # 7. start training
        trainer.train(f'resources/taggers/keyphrase-issue/{embedding_name}_{corpus_name}',
                      learning_rate=0.1,
                      mini_batch_size=32,
                      max_epochs=150)