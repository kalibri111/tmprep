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
            final_div_factor=10e+4,
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
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            total_steps=T,
            max_lr=lr,
            pct_start=0.1,
            anneal_strategy='linear',
            final_div_factor=final_div_factor
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

    # def generate_square_subsequent_mask(self, size):
    #     mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1).float()
    #     mask = (mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))).to(self.device)
    #     return mask
    

    # def forward(self, src, trg):
    #     # TODO: Реализуйте forward pass для модели, при необходимости реализуйте другие функции для обучения
    #     if self.src_mask is None or self.src_mask.size(0) != src.size(1):
    #         self.src_mask = nn.Transformer.generate_square_subsequent_mask(src.size(1)).to(self.device)
    #     if self.trg_mask is None or self.trg_mask.size(0) != trg.size(1):
    #         self.trg_mask =  nn.Transformer.generate_square_subsequent_mask(trg.size(1)).to(self.device)

    #     src_embed = self.encoder_embedding(src) * math.sqrt(self.positional_embedding_size)
    #     src_embed = self.positional_encoder(src_embed)
    #     trg_embed = self.decoder_embedding(trg) * math.sqrt(self.positional_embedding_size)
    #     trg_embed = self.positional_encoder(trg_embed)

    #     src_embed = src_embed.permute(1, 0, 2)
    #     trg_embed = trg_embed.permute(1, 0, 2)

    #     output = self.transformer(
    #         src_embed,
    #         trg_embed,
    #         src_mask=self.src_mask,
    #         tgt_mask=self.trg_mask,
    #     )
    #     output = self.vocab_projection(output).permute(1, 0, 2)
    #     top = torch.argmax(output, dim=-1)

    #     return top.clone(), output

    # def training_step(self, batch):
    #     # self.optimizer.zero_grad()
    #     # input_tensor, target_tensor = batch
    #     # _, decoder_outputs = self.forward(input_tensor, target_tensor[:, :-1])
    #     # target_tensor = target_tensor[:, :, None]
    #     # target_length = target_tensor.shape[1]
    #     # loss = 0.0
    #     # for di in range(target_length):
    #     #     loss += self.criterion(
    #     #         decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
    #     #     )
    #     # loss = loss / target_length
    #     # loss.backward()
    #     # self.optimizer.step()
    #     # self.scheduler.step()

    #     # return loss.item()
    #     # TODO: Реализуйте обучение на 1 батче данных по примеру seq2seq_rnn.py
    #     self.optimizer.zero_grad()
    #     input_tensor, target_tensor = batch
    #     (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
    #     target = target_tensor[:, 1:].reshape(-1)
    #     output = output.reshape(-1, output.shape[-1])
    #     loss = self.criterion(output, target)
    #     loss.backward()
    #     self.optimizer.step()
    #     self.scheduler.step()
        
    #     return loss.item()

    # def validation_step(self, batch):
    #     # input_tensor, target_tensor = batch
    #     # _, decoder_outputs = self.forward(input_tensor, target_tensor)
    #     # target_tensor = target_tensor[:, :, None]
    #     # with torch.no_grad():
    #     #     target_length = target_tensor.shape[1]
    #     #     loss = 0
    #     #     for di in range(target_length):
    #     #         loss += self.criterion(
    #     #             decoder_outputs[di].squeeze(), target_tensor[:, di, :].squeeze()
    #     #         )
    #     #     loss = loss / target_length

    #     # return loss.item()
    #     # TODO: Реализуйте оценку на 1 батче данных по примеру seq2seq_rnn.py
    #     input_tensor, target_tensor = batch
    #     (_, output) = self.forward(input_tensor, target_tensor[:, :-1])
    #     target = target_tensor[:, 1:].reshape(-1)
    #     output = output.reshape(-1, output.shape[-1])
    #     loss = self.criterion(output, target)
        
    #     return loss.item()


    # def eval_bleu(self, predicted_ids_list, target_tensor):
    #     predicted = predicted_ids_list.clone()
    #     predicted = predicted.squeeze().detach().cpu().numpy().swapaxes(0, 1)
    #     actuals = target_tensor.squeeze().detach().cpu().numpy()
    #     bleu_score, actual_sentences, predicted_sentences = metrics.bleu_scorer(
    #         predicted=predicted, actual=actuals, target_tokenizer=self.target_tokenizer
    #     )
    #     return bleu_score, actual_sentences, predicted_sentences

    def generate_square_subsequent_mask(self, length: int):
        mask = torch.tril(torch.ones((length, length), device=self.device)).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0))
        return mask

    def forward(self, input_tensor: torch.Tensor):
        # input_tensor: (B, S), S - sequence length, B - batch_size
        # Embedding + positional encoding: (S, B, E), E - encoder_embedding size
        src = self.positional_encoder(self.encoder_embedding(input_tensor.transpose(0, 1)))
        # A memory is an encoder output: (S, B, E):
        memory = self.transformer.encoder(src)
        # Output
        pred_tokens = [torch.full((input_tensor.size(0),), self.target_tokenizer.word2index[self.start_symbol])]
        each_step_distributions = [nn.functional.one_hot(pred_tokens[0],
                                                         self.vocab_projection.out_features).to(self.device).float()]
        each_step_distributions[0] = each_step_distributions[0].masked_fill(
            each_step_distributions[0] == 0, float('-inf')
        ).masked_fill(
            each_step_distributions[0] == 1, float(0)
        )
        # (S, B), where S is the length of the predicted sequence
        prediction = torch.full((1, input_tensor.size(0)), self.target_tokenizer.word2index[self.start_symbol], dtype=torch.long, device=self.device)
        for i in range(self.target_tokenizer.max_sent_len - 1):
            tgt_mask = self.generate_square_subsequent_mask(prediction.size(0))
            out = self.transformer.decoder(self.decoder_embedding(prediction), memory, tgt_mask)
            logits = self.vocab_projection(out[-1])
            _, next_word = torch.max(logits, dim=1)
            prediction = torch.cat([prediction, next_word.unsqueeze(0)], dim=0)
            # Output update
            pred_tokens.append(next_word.clone().detach().cpu())
            each_step_distributions.append(logits)

        return pred_tokens, each_step_distributions

    def training_step(self, batch):
        self.optimizer.zero_grad()
        input_tensor, target_tensor = batch  # (B, S)
        predicted, decoder_outputs = self.forward(input_tensor)
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
        predicted, decoder_outputs = self.forward(input_tensor)
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


