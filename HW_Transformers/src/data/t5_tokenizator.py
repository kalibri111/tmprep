
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import T5Tokenizer, PreTrainedTokenizer


class T5Tokenizator:
    def __init__(self, sentence_list, pretrained_name="google-t5/t5-base"):
        """
        sentence_list - список предложений для обучения
        """
        self.tokenizer = self.load_pretrained(sentence_list, pretrained_name)
        
        self.set_attr()

    
    def __call__(self, sentence):
        """
        sentence - входное предложение
        """

        return self.tokenizer(sentence, padding='max_length', max_length=self.max_sent_len, truncation=True)


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list, skip_special_tokens=True).split()
    
    
    def load_pretrained(self, sentence_list, pretrained_name):
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
        self.max_sent_len = None
        max_sent_len = 0
        for sentence in sentence_list:
            max_sent_len = max(max_sent_len, len(self.tokenizer.encode(sentence, padding=False, truncation=False)))
        self.max_sent_len = max_sent_len

        return self.tokenizer

    
    def set_attr(self):
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {id: w for w, id in self.word2index.items()}