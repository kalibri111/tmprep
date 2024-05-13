import torch

import metrics
import math

import torch
import torch.nn as nn
from src.models.positional_encoding import PositionalEncoding


class Seq2SeqTransformer(torch.nn.Module):
    def __init__(
            self,
            encoder_vocab_size: int,
            decoder_vocab_size: int,
            dim_feedforward: int,
            lr: float,
            device:str,
            target_tokenizer,
            T,
            positional_embedding_size=256,
            n_heads_attention=8,
            n_encoders=6,
            n_decoders=6,
            dropout=0.1,
            start_symbol="[SOS]"
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.device = device
        # TODO: Реализуйте конструктор seq2seq трансформера - матрица эмбеддингов, позиционные эмбеддинги, encoder/decoder трансформер, vocab projection head
        
        self.transformer = nn.Transformer(
            d_model=positional_embedding_size,
            dim_feedforward=dim_feedforward,
            nhead=n_heads_attention,
            num_encoder_layers=n_encoders,
            num_decoder_layers=n_decoders,
            dropout=dropout
        ).to(self.device)

        self.encoder_embedding = nn.Embedding(encoder_vocab_size, positional_embedding_size).to(self.device)
        self.decoder_embedding = nn.Embedding(decoder_vocab_size, positional_embedding_size).to(self.device)

        self.positional_encoder = PositionalEncoding(emb_size=positional_embedding_size, max_len=target_tokenizer.max_sent_len).to(self.device)
        
        self.vocab_projection = nn.Linear(positional_embedding_size, decoder_vocab_size).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            total_steps=T,
            max_lr=lr,
        )
        
        self.target_tokenizer = target_tokenizer
        self.positional_embedding_size = positional_embedding_size
        self.src_mask = None
        self.trg_mask = None
        self.start_symbol = start_symbol

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)

        self.vocab_projection.bias.data.zero_()
        self.vocab_projection.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_tensor: torch.Tensor):
        num_vocab = self.vocab_projection.out_features
        encoded_features = self.encoder_embedding(input_tensor.transpose(0, 1)) * math.sqrt(self.positional_embedding_size)
        features = self.positional_encoder(encoded_features)

        encoder_output = self.transformer.encoder(features)
        
        predicted_ids_list = []
        distributions = []
        predicted_ids_list[0] = torch.full((input_tensor.size(0),), self.target_tokenizer.word2index[self.start_symbol])
        distributions[0] = nn.functional.one_hot(predicted_ids_list[0], num_vocab).to(self.device).float()
        distributions[0] = distributions[0].masked_fill(distributions[0] == 0, float('-inf')).masked_fill(distributions[0] == 1, float(0))

        prediction = torch.full((1, input_tensor.size(0)), self.target_tokenizer.word2index[self.start_symbol], dtype=torch.long, device=self.device)
        for i in range(self.target_tokenizer.max_sent_len - 1):
            trg_mask = nn.Transformer.generate_square_subsequent_mask(prediction.size(0)).to(self.device)
            decoder_input = self.decoder_embedding(prediction) * math.sqrt(self.positional_embedding_size)
            decoder_output = self.transformer.decoder(decoder_input, encoder_output, trg_mask)
            logits = self.vocab_projection(decoder_output[-1])
            distributions.append(logits)
            _, next_word = torch.max(logits, dim=1)
            predicted_ids_list.append(next_word.clone().detach().cpu())

            prediction = torch.cat([prediction, next_word.unsqueeze(0)], dim=0)

        return predicted_ids_list, distributions

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch
        _, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        target_length = target_tensor.shape[1]
        loss = 0.0
        for di in range(target_length):
            loss += self.criterion(
                decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
            )
        loss = loss / target_length
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch):
        input_tensor, target_tensor = batch
        _, decoder_outputs = self.forward(input_tensor)
        target_tensor = target_tensor[:, :, None]
        with torch.no_grad():
            target_length = target_tensor.shape[1]
            loss = 0
            for di in range(target_length):
                loss += self.criterion(
                    decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
                )
            loss = loss / target_length

        return loss.item()

    def eval_bleu(self, predicted_ids_list, target_tensor):
        predicted = torch.stack(predicted_ids_list)
        predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)[:, 1:]
        actuals = target_tensor.squeeze().detach().cpu().numpy()[:, 1:]
        bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
            predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
        )
        return bleu_score, actual_sentences, predicted_sentences


