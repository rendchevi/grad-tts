import torch
import torch.nn as nn

from model.utils import sequence_mask
from model.text_encoder import ConvReluNorm, Encoder


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    ''' Gradient Reversal Layer
            Y. Ganin, V. Lempitsky,
            "Unsupervised Domain Adaptation by Backpropagation",
            in ICML, 2015.
        Forward pass is the identity function
        In the backward pass, upstream gradients are multiplied by -lambda (i.e. gradient are reversed)
    '''
    def __init__(self, lambda_reversal=1.0):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_reversal

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ReferenceEncoder(torch.nn.Module):
    def __init__(
        self,
        n_feats,
        n_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        with_film,
        with_aux_clf,
        n_spks=1,
        spk_emb_dim=None,
        window_size=None,
        n_aux_labels=5,
        lambda_reversal=1.0,
    ):
        super().__init__()

        self.prenet = ConvReluNorm(
            n_feats,
            n_channels,
            n_channels, 
            kernel_size=5,
            n_layers=3,
            p_dropout=0.5
        )

        self.encoder = Encoder(
            n_channels + (spk_emb_dim if n_spks > 1 else 0),
            filter_channels,
            n_heads,
            n_layers, 
            kernel_size,
            p_dropout,
            window_size=window_size,
        )

        self.with_film = with_film
        if self.with_film:
            self.gammas_generator = torch.nn.Conv1d(n_channels, n_channels*n_layers, 1)
            self.betas_generator = torch.nn.Conv1d(n_channels, n_channels*n_layers, 1)

        self.with_aux_clf = with_aux_clf
        if self.with_aux_clf:
            self.aux_clf = nn.Sequential(
                GradientReversal(lambda_reversal=lambda_reversal),
                nn.Linear(n_channels, n_channels),
                nn.ReLU(),
                nn.Linear(n_channels, n_channels),
                nn.ReLU(),
                nn.Linear(n_channels, n_aux_labels)
            )


    def extract_embedding(self, x, x_lengths, spk=None):

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.prenet(x, x_mask, add_residual=False) # [B, D, T]

        x = self.encoder(x, x_mask) # [B, D, T]
        u = torch.sum(x, dim=2) / x_lengths.unsqueeze(1) # [B, D]

        return u
    

    def generate_film(self, u):

        gammas = self.gammas_generator(u.unsqueeze(-1)).squeeze(-1).reshape([-1, 6, 192]).unsqueeze(-1) # [B, L, D, 1]
        betas = self.betas_generator(u.unsqueeze(-1)).squeeze(-1).reshape([-1, 6, 192]).unsqueeze(-1) # [B, L, D, 1]

        return gammas, betas


    def forward(self, x, x_lengths, spk=None):

        u = self.extract_embedding(x, x_lengths)
        if self.with_film:
            gammas, betas = self.generate_film(u)
        else:
            gammas, betas = None, None

        return u, gammas, betas