import torch
import torch.nn as nn


from tests.stage2.model.layers.Transformer_EncDec import Encoder, EncoderLayer
from tests.stage2.model.layers.SelfAttention_Family import FullAttention, AttentionLayer
from tests.stage2.model.layers.Embed import DataEmbedding_inverted


class Model(nn.Module):
    """
    Adapted iTransformer:
        N time series  --->  1 time series

    Input:
        x_enc : (B, seq_len, N)

    Output:
        (B, pred_len)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

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
        # MLP head
        #################################################################

        hidden = 4 * configs.d_model

        self.head = nn.Sequential(
            nn.Linear(configs.enc_in * configs.d_model, hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),

            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(configs.dropout),

            nn.Linear(hidden, configs.pred_len),
)

    def forecast(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):

        # (B, L, N) -> (B, N, d_model)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # Encoder iTransformer
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # (B, N, d_model) -> (B, N*d_model)
        enc_out = enc_out.reshape(enc_out.size(0), -1)

        # MLP Head: (B, N*d_model) -> (B, pred_len)
        dec_out = self.head(enc_out)

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