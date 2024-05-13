import torch
from torch.nn import CrossEntropyLoss, Embedding
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor

import metrics


class Seq2SeqT5(torch.nn.Module):
    def __init__(self, 
                encoder_vocab_size: int,
                decoder_vocab_size: int,
                lr: float,
                device: str,
                T,
                target_tokenizer,
                pretrained_name: str,
                start_symbol="[SOS]",
    ):
        super(Seq2SeqT5, self).__init__()
        self.device = device
        # TODO: Реализуйте конструктор seq2seq t5 - https://huggingface.co/docs/transformers/model_doc/t5
        
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_name).to(self.device)

        self.model.resize_token_embeddings(encoder_vocab_size)

        self.criterion = CrossEntropyLoss().to(self.device)
        self.optimizer = Adafactor(self.model.parameters(),
                                   lr=lr,
                                   relative_step=False,
                                   warmup_init=False)
        # self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer=self.optimizer,
        #     total_steps=T,
        #     max_lr=lr,
        # )
        
        self.max_sent_len = target_tokenizer.max_sent_len
        self.target_tokenizer = target_tokenizer
        self.start_symbol = start_symbol

    def forward(self, input_tensor: torch.Tensor, labels, attention_mask=None):
        # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения

        return self.model(input_ids=input_tensor, labels=labels, attention_mask=attention_mask)


    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor, attention_mask = batch

        loss = self.forward(input_tensor, target_tensor, attention_mask).loss
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def validation_step(self, batch):
        input_tensor, target_tensor, attention_mask = batch

        with torch.no_grad():
            loss = self.forward(input_tensor, target_tensor, attention_mask).loss

        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = predicted_ids_list.squeeze().detach().cpu().numpy().astype(int)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences

    def generate(self, src, attention_mask):
        outputs = self.model.generate(src, attention_mask=attention_mask)
        return outputs



