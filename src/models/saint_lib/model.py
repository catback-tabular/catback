import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def exists(val):
    """
    Checks if a value (val) is not None.
    Useful for concise conditional statements.
    """
    return val is not None

def default(val, d):
    """
    Returns val if it exists, otherwise returns d.
    This is a handy shortcut for setting default parameters.
    """
    return val if exists(val) else d

def ff_encodings(x, B):
    """
    Computes random Fourier feature encodings. 
    This is not heavily used in SAINT by default, but can be useful 
    for adding explicit feature encodings to continuous data.

    Args:
        x (Tensor): input of shape [batch_size, ...]
        B (Tensor): random projection matrix

    Returns:
        Tensor of shape [..., 2 * B.shape[0]] containing sin & cos projections
    """
    # Expand x to [..., 1] for matrix multiplication, multiply by B transpose
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    # Concatenate sine and cosine of the projections
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ----------------------------------------------------------------------------
# Residual & Normalization Layers
# ----------------------------------------------------------------------------

class Residual(nn.Module):
    """
    A residual connection wrapper that adds the output of fn(x) to x.
    """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # e.g. an Attention or FeedForward layer

    def forward(self, x, **kwargs):
        # Residual connection: out = x + fn(x)
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    """
    A wrapper that applies LayerNorm to x, then passes it through fn.
    This is a common pattern in Transformers, referred to as "Pre-LN."
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # Apply LayerNorm first, then call fn
        return self.fn(self.norm(x), **kwargs)

# ----------------------------------------------------------------------------
# Feed-Forward and Attention Modules
# ----------------------------------------------------------------------------

class GEGLU(nn.Module):
    """
    Gated Linear Unit with GELU activation: 
    Splits features in half, applies GELU to one half, then multiplies.
    This is similar to GLU, except the gating uses GELU instead of sigmoid.
    """
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)  # split tensor along last dimension
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    """
    The standard Transformer Feed-Forward block:
    - expands dimension by 'mult'
    - uses a gating mechanism (GEGLU)
    - projects back to original dimension
    - includes optional dropout
    """
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),           # Gated activation
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)

class Attention(nn.Module):
    """
    A Multi-Head Self-Attention block.
    - Splits input embeddings into multiple heads.
    - Scales dot product by sqrt(dim_head).
    - Applies softmax over the attention scores.
    """
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=16,
        dropout=0.
    ):
        super().__init__()
        # inner_dim is total hidden dimension across all heads
        inner_dim = dim_head * heads
        self.heads = heads
        # scaling factor for the dot-product attention
        self.scale = dim_head ** -0.5

        # Linear layers to transform input x into queries, keys, and values
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        # Final linear layer to combine heads output
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        # 1) Project input to q, k, v (each shape [b, n, inner_dim])
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # 2) Reshape them to [b, heads, n, dim_head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # 3) Compute scaled dot product similarity
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        
        # 4) Softmax to get attention probabilities
        attn = sim.softmax(dim=-1)

        # 5) Weighted sum of values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 6) Reshape back to [b, n, inner_dim]
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)

        # 7) Final linear projection
        return self.to_out(out)

# ----------------------------------------------------------------------------
# RowColTransformer - special Transformer with row/column attention
# ----------------------------------------------------------------------------

class RowColTransformer(nn.Module):
    """
    A specialized Transformer that can apply 'row' or 'colrow' attention:
    - 'colrow': applies attention along columns first, 
                then reshapes and applies attention along rows.
    - 'row': or a simpler row-based approach if style=='row'.
    """
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, style='col'):
        """
        Args:
            nfeats (int): total number of features (categorical + continuous)
            style (str): 'colrow' or 'row'. 
        """
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style

        # Build the multi-layer structure
        for _ in range(depth):
            if self.style == 'colrow':
                # For 'colrow', we do: (1) column-wise attn, (2) feed-forward,
                # then (3) reshape, apply row-wise attn, (4) feed-forward again
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim * nfeats, Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))
            else:
                # For 'row' style, we do a single dimension transformation by flattening [n, d] -> [n*d],
                # then apply attention, feed-forward, reshape back
                self.layers.append(nn.ModuleList([
                    PreNorm(dim * nfeats, Residual(Attention(dim * nfeats, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim * nfeats, Residual(FeedForward(dim * nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask=None):
        """
        Args:
            x (Tensor):   shape [batch_size, n_categ, dim] for categorical
            x_cont (Tensor): shape [batch_size, n_cont, dim] for continuous
            mask (Tensor): optional masking (not heavily used here)

        Returns:
            x (Tensor): shape [batch_size, n_categ + n_cont, dim]
        """
        if x_cont is not None:
            # Concatenate categorical & continuous embeddings along feature dimension
            x = torch.cat((x, x_cont), dim=1)

        b, n, d = x.shape

        # Depending on the style, apply the appropriate sequence of transformations
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                # Column-wise attention + feed-forward
                x = attn1(x)
                x = ff1(x)

                # Reshape for row-wise attention:
                # from [b, n, d] -> [1, b, n*d], so we treat rows as tokens.
                x = rearrange(x, 'b n d -> 1 b (n d)')
                # row-wise attention + feed-forward
                x = attn2(x)
                x = ff2(x)
                # Reshape back to [b, n, d]
                x = rearrange(x, '1 b (n d) -> b n d', n=n)

        else:
            # 'row' style
            for attn1, ff1 in self.layers:
                # Flatten [b, n, d] into [1, b, n*d]
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                # Reshape back
                x = rearrange(x, '1 b (n d) -> b n d', n=n)

        return x

# ----------------------------------------------------------------------------
# Standard Transformer (column-based)
# ----------------------------------------------------------------------------

class Transformer(nn.Module):
    """
    A simpler Transformer that processes [batch_size, n, dim]. 
    It applies a series of (Attention + FeedForward) blocks, each with PreNorm + Residual.
    """
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        # We do not necessarily use num_tokens here directly for embedding 
        # because embeddings might be handled externally.

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        """
        Args:
            x (Tensor):   shape [batch_size, n_categ, dim]
            x_cont (Tensor): shape [batch_size, n_cont, dim] or None

        Returns:
            x (Tensor):   shape [batch_size, n_categ + n_cont, dim]
        """
        if x_cont is not None:
            # Concatenate categorical and continuous embeddings along the sequence dimension
            x = torch.cat((x, x_cont), dim=1)

        # Pass through each Transformer layer
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x

# ----------------------------------------------------------------------------
# MLP Components
# ----------------------------------------------------------------------------

class MLP(nn.Module):
    """
    A generic multi-layer perceptron:
    - dims: list of layer sizes, e.g. [input_dim, hidden_dim, ..., output_dim]
    - act: optional activation function to insert between layers (besides the last)
    """
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))

        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 2)  # True if this is the last layer
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            # If not the last layer, optionally append a custom activation
            if not is_last and act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for MLP
        """
        return self.mlp(x)

class simple_MLP(nn.Module):
    """
    A simpler MLP with a fixed structure of:
    Linear -> ReLU -> Linear

    dims = [input_dim, hidden_dim, output_dim]
    """
    def __init__(self, dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        # If x is a 1D tensor, reshape it to 2D [batch_size, feature_dim]
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# ----------------------------------------------------------------------------
# TabAttention: Another class demonstrating how attention can be applied to
# tabular data, combining categorical & continuous features.
# ----------------------------------------------------------------------------

class TabAttention(nn.Module):
    """
    A high-level module showing how to apply the above Transformers and 
    MLPs to tabular data (categorical + continuous features).
    """
    def __init__(
        self,
        *,
        categories,           
        num_continuous,       
        dim,
        depth,
        heads,
        dim_head=16,
        dim_out=1,
        mlp_hidden_mults=(4, 2),
        mlp_act=None,
        num_special_tokens=1,
        continuous_mean_std=None,
        attn_dropout=0.,
        ff_dropout=0.,
        lastmlp_dropout=0.,
        cont_embeddings='MLP',
        scalingfactor=10,
        attentiontype='col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # --------------------------------------------------------------------
        # Category / Token Embeddings
        # --------------------------------------------------------------------
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # Offset for each categorical feature in the embedding space
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens)
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        # --------------------------------------------------------------------
        # Continuous Feature Embeddings
        # --------------------------------------------------------------------
        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([
                simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)
            ])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continuous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # --------------------------------------------------------------------
        # Transformer (RowCol or Standard)
        # --------------------------------------------------------------------
        if attentiontype == 'col':
            # Standard column-based Transformer
            self.transformer = Transformer(
                num_tokens=self.total_tokens,
                dim=dim,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout
            )
        elif attentiontype in ['row', 'colrow']:
            # RowColTransformer
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=dim,
                nfeats=nfeats,
                depth=depth,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                style=attentiontype
            )

        # --------------------------------------------------------------------
        # Final MLP for classification/regression
        # --------------------------------------------------------------------
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

        # Main embedding layer for categorical tokens
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        # Additional mask embeddings for missing values
        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('cat_mask_offset', cat_mask_offset)

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)

    def forward(self, x_categ, x_cont, x_categ_enc, x_cont_enc):
        """
        Forward pass for TabAttention.

        Args:
            x_categ (Tensor): raw categorical input, shape [batch_size, ..., ?]
            x_cont (Tensor): raw continuous input, shape [batch_size, ..., ?]
            x_categ_enc (Tensor): pre-embedded categorical data, shape [batch_size, n_categ, dim]
            x_cont_enc (Tensor): pre-embedded continuous data, shape [batch_size, n_cont, dim]

        Returns:
            The MLP output (e.g., classification or regression prediction).
        """
        device = x_categ.device

        # 'justmlp' is a style in which we skip the transformer and pass everything into MLP
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                # Flatten the categorical data and continuous data, then concatenate
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim=-1)
            else:
                # If no categorical data, just use continuous
                x = x_cont.clone()
        else:
            # Otherwise, use the transformer
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc, x_cont_enc.to(device))
            else:
                # If no continuous embeddings, or cont_embeddings != 'MLP'
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else:
                    # Flatten only after passing x_categ_enc to the transformer
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim=-1)

        # Flatten final embeddings before the MLP
        flat_x = x.flatten(1)
        return self.mlp(flat_x)
