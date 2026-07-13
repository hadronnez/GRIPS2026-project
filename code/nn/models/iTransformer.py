import torch
import torch.nn as nn


from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    iTransformer adaptado para:
        N variables exógenas  --->  1 serie objetivo

    Entrada:
        x_enc : (B, seq_len, N)

    Salida:
        (B, pred_len)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )

        #################################################################
        # NUEVO
        #################################################################

        # Aprende cómo combinar las N variables
        self.token_pool = nn.Linear(configs.enc_in, 1)

        # Convierte la representación latente en la serie de precios
        self.projector = nn.Linear(configs.d_model, configs.pred_len)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        # (B,L,N) -> (B,N,d_model)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Pooling sobre las variables
        enc_out = enc_out.transpose(1, 2)
        enc_out = self.token_pool(enc_out).squeeze(-1)

        # (B,d_model) -> (B,pred_len)
        dec_out = self.projector(enc_out)

        return dec_out, attns


    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
        mask=None,
    ):

        dec_out, attns = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec,
        )

        if self.output_attention:
            return dec_out, attns
        else:
            return dec_out