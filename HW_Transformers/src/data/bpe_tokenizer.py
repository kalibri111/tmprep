
from typing import List

from tokenizers import Tokenizer
from tokenizers import decoders
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from transformers import T5Tokenizer, PreTrainedTokenizer


class BPETokenizer:
    def __init__(self, sentence_list, pretrained_name=None):
        """
        sentence_list - список предложений для обучения
        """

        if pretrained_name is None:
            self.is_pretrained = True
            self.tokenizer = self.train(sentence_list)
        else:
            self.is_pretrained = False
            self.tokenizer = self.load_pretrained(sentence_list, pretrained_name)
        
        self.set_attr()

    
    def __call__(self, sentence):
        """
        sentence - входное предложение
        """
        # if self.is_pretrained:
        #     padding = self.pad_flag if pretrained_force_padding is None else pretrained_force_padding
        #     id_list = self._tokenizer.encode(sentence,
        #                                      padding='max_length' if padding else False,
        #                                      truncation=padding,
        #                                      max_length=self.max_sent_len)
        # else:
        id_list = self.tokenizer.encode(sentence).ids
        return id_list


    def decode(self, token_list):
        """
        token_list - предсказанные ID вашего токенизатора
        """
        return self.tokenizer.decode(token_list, skip_special_tokens=True).split()

    
    def train(self, sentence_list):
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.tokenizer.decoder = decoders.BPEDecoder()

        trainer = BpeTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"],
                            end_of_word_suffix="</w>")
        self.tokenizer.train_from_iterator(sentence_list, trainer)

        self.tokenizer.post_processor = TemplateProcessing(
            single="[SOS] $A [EOS]",
            pair="[SOS] $A [EOS] $B:1 [EOS]:1",
            special_tokens=[
                ("[SOS]", self.tokenizer.token_to_id("[SOS]")),
                ("[EOS]", self.tokenizer.token_to_id("[EOS]")),
            ]
        )

        self.max_sent_len = 0
        for sentence in sentence_list:
            self.max_sent_len = max(self.max_sent_len, len(self(sentence)))
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), length=self.max_sent_len)
        self.tokenizer.enable_truncation(max_length=self.max_sent_len)

        return self.tokenizer
    
    
    def load_pretrained(self, sentence_list, pretrained_name):
        # self.tokenizer = T5Tokenizer.from_pretrained(pretrained_name)
        # if self.pad_flag:
        #     self.max_sent_len = self._get_max_length_in_tokens(sentence_list)
        # # Special tokens
        # self.unknown_token = self._tokenizer.unk_token
        # self.sos_token = self._tokenizer.pad_token
        # self.eos_token = self._tokenizer.eos_token
        # self.pad_token = self._tokenizer.pad_token
        # return self._tokenizer
        pass

    
    def set_attr(self):
        self.word2index = self.tokenizer.get_vocab()
        self.index2word = {id: w for w, id in self.word2index.items()}