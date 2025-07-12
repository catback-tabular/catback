# The following import brings in all contents from model.py, which likely includes:
# - Transformer, RowColTransformer classes
# - MLP, simple_MLP definitions
# - Possibly other utility functions or classes
from .model import *


###############################################################################
# 1. sep_MLP CLASS
###############################################################################
# The 'sep_MLP' class defines a separate MLP for each feature dimension. 
# This can be useful if, for instance, we want to output a separate prediction 
# (or reconstruction) for each feature. It's commonly used in pretraining tasks 
# like denoising or reconstructing features.
###############################################################################
class sep_MLP(nn.Module):
    def __init__(self, dim, len_feats, categories):
        """
        Args:
            dim (int):         The size of the input embedding dimension.
            len_feats (int):   The number of features we need separate predictions for.
            categories (List): The cardinalities (number of classes) of each categorical feature
                               or just 1 if it is continuous.
        """
        super(sep_MLP, self).__init__()
        self.len_feats = len_feats

        # We create a ModuleList of 'simple_MLP' networks, 
        # each responsible for handling one feature's output dimension.
        self.layers = nn.ModuleList([])
        for i in range(len_feats):
            # For the i-th feature, we define a small MLP: [dim -> 5*dim -> categories[i]]
            # so that each feature has its own "head" of output.
            self.layers.append(simple_MLP([dim, 5 * dim, categories[i]]))

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape [batch_size, len_feats, dim], 
                        i.e. for each batch and for each feature, we have an embedding.

        Returns:
            A list of length 'len_feats', where each element is the prediction
            corresponding to one feature.
        """
        y_pred = []
        # For each feature dimension i, we slice out x[:, i, :] 
        # and pass it through that feature's MLP.
        for i in range(self.len_feats):
            x_i = x[:, i, :]        # shape [batch_size, dim]
            pred = self.layers[i](x_i)  # shape [batch_size, categories[i]]
            y_pred.append(pred)
        return y_pred


###############################################################################
# 2. SAINT CLASS
###############################################################################
# The main SAINT model class. This class shows how the architecture is organized:
#  - Category embeddings for each categorical feature
#  - Optional continuous feature embeddings (MLP or other methods)
#  - A Transformer or RowColTransformer-based encoder
#  - Final MLP heads for various tasks (e.g., classification, reconstruction, etc.)
###############################################################################
class SAINT(nn.Module):
    def __init__(
        self,
        *,
        categories,           # A list/tuple with the number of classes for each categorical feature
        num_continuous,       # Number of continuous features
        dim,                  # Embedding dimension
        depth,                # Number of Transformer blocks
        heads,                # Number of attention heads
        dim_head = 16,        # Dimension per attention head
        dim_out = 1,          # Output dimension (e.g., for final MLP)
        mlp_hidden_mults = (4, 2),  # Hidden dimension multipliers for the MLP
        mlp_act = None,       # Activation function for the MLP (if any)
        num_special_tokens = 0,# Number of special tokens (like CLS) in the embedding table
        attn_dropout = 0.,    # Dropout ratio for attention
        ff_dropout = 0.,      # Dropout ratio for feed-forward layers
        cont_embeddings = 'MLP',  # Method to embed continuous features: 'MLP', 'pos_singleMLP', etc.
        scalingfactor = 10,   # Not explicitly used in this snippet, might be for rescaling something
        attentiontype = 'col',# Type of attention: 'col', 'row', 'colrow', etc.
        final_mlp_style = 'common', # Whether final MLP is common or separated for each feature
        y_dim = 2             # Dimension of the final classification/regression output
    ):
        super().__init__()
        # Make sure each category in 'categories' is > 0
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # ---------------------------------------------------------------------
        # 2.1 BASIC MODEL INFORMATION
        # ---------------------------------------------------------------------
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)  # total number of (unique) categories across features

        # 'num_special_tokens' are tokens like a [CLS] token or other special tokens 
        # that might be appended to the embedding table.
        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens
        
        # 'categories_offset' is used to map each feature's category IDs into 
        # a global ID space. For example, if the first feature has 10 categories,
        # the second feature's category IDs get offset by 10, and so on.
        categories_offset = F.pad(
            torch.tensor(list(categories)), (1, 0), value=num_special_tokens
        )
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # LayerNorm for continuous features
        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype
        self.final_mlp_style = final_mlp_style

        # ---------------------------------------------------------------------
        # 2.2 CONTINUOUS FEATURE EMBEDDINGS
        # ---------------------------------------------------------------------
        # If we choose 'MLP', we create a separate simple_MLP for each continuous feature.
        # Then each continuous feature is mapped into an embedding dimension 'dim'.
        if self.cont_embeddings == 'MLP':
            # self.simple_MLP is a ModuleList, one MLP for each continuous feature
            self.simple_MLP = nn.ModuleList([
                simple_MLP([1, 100, self.dim]) for _ in range(self.num_continuous)
            ])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous

        # If we choose 'pos_singleMLP', we only create a single MLP that might 
        # be re-used or combined for position encoding, etc.
        elif self.cont_embeddings == 'pos_singleMLP':
            self.simple_MLP = nn.ModuleList([
                simple_MLP([1, 100, self.dim]) for _ in range(1)
            ])
            input_size = (dim * self.num_categories) + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            # Otherwise, we do not pass continuous features through attention, 
            # so the input to the model has dimension = (categorical_emb + raw_cont).
            print('Continuous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories

        # ---------------------------------------------------------------------
        # 2.3 TRANSFORMER ENCODER
        # ---------------------------------------------------------------------
        # We now define the Transformer or RowColTransformer depending on the 'attentiontype'.
        if attentiontype == 'col':
            # A standard Transformer that focuses on "column" attention
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
            # A specialized version that can do row attention or both row & column attention
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

        # ---------------------------------------------------------------------
        # 2.4 MLP FOR (OPTIONAL) FINAL PREDICTION
        # ---------------------------------------------------------------------
        # We define an MLP that might process the concatenated embeddings for 
        # certain tasks. The dimension of each subsequent layer is determined
        # by mlp_hidden_mults. Finally, it outputs 'dim_out' dimension.
        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        self.mlp = MLP(all_dimensions, act=mlp_act)

        # ---------------------------------------------------------------------
        # 2.5 CATEGORY EMBEDDING TABLE
        # ---------------------------------------------------------------------
        # 'embeds' is the main embedding table used for all categorical tokens. 
        # Size = [self.total_tokens, self.dim], where total_tokens includes 
        # the sum of categories plus any special tokens.
        self.embeds = nn.Embedding(self.total_tokens, self.dim)

        # ---------------------------------------------------------------------
        # 2.6 MASK EMBEDDINGS AND OFFSETS
        # ---------------------------------------------------------------------
        # For each categorical feature, we can have two possible states: 
        # (0) missing, (1) not missing. So we have an embedding table for 
        # "mask_embeds_cat" of size [self.num_categories*2, self.dim].
        cat_mask_offset = F.pad(
            torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0
        )
        cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('cat_mask_offset', cat_mask_offset)

        con_mask_offset = F.pad(
            torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0
        )
        con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.dim)
        self.single_mask = nn.Embedding(2, self.dim)  # Maybe for a global mask token

        # A positional encoding embedding that can embed up to (num_categories+num_continuous) positions
        self.pos_encodings = nn.Embedding(self.num_categories + self.num_continuous, self.dim)

        # ---------------------------------------------------------------------
        # 2.7 FINAL MLP HEADS FOR PRETRAINING/TASKS
        # ---------------------------------------------------------------------
        # final_mlp_style = 'common' or 'sep'
        # 'common': We have a single MLP for categoricals and one for continuous.
        # 'sep':    We have separate heads (sep_MLP) for each feature.
        if self.final_mlp_style == 'common':
            # A single MLP for all categorical features
            self.mlp1 = simple_MLP([dim, (self.total_tokens) * 2, self.total_tokens])
            # A single MLP for all continuous features
            self.mlp2 = simple_MLP([dim, self.num_continuous, 1])
        else:
            # A separate MLP for each categorical feature, dimension = categories[i]
            self.mlp1 = sep_MLP(dim, self.num_categories, categories)
            # A separate MLP for each continuous feature (each has 1 output dimension)
            self.mlp2 = sep_MLP(dim, self.num_continuous, np.ones(self.num_continuous).astype(int))

        # MLP for final classification (or regression) of label y
        self.mlpfory = simple_MLP([dim, 1000, y_dim])

        # Additional MLPs for pretraining tasks (like contrastive or denoising).
        # 'pt_mlp' might be used for projection heads or reconstruction tasks.
        self.pt_mlp  = simple_MLP([
            dim * (self.num_continuous + self.num_categories),
            6 * dim * (self.num_continuous + self.num_categories) // 5,
            dim * (self.num_continuous + self.num_categories) // 2
        ])
        self.pt_mlp2 = simple_MLP([
            dim * (self.num_continuous + self.num_categories),
            6 * dim * (self.num_continuous + self.num_categories) // 5,
            dim * (self.num_continuous + self.num_categories) // 2
        ])

    def forward(self, x_categ, x_cont):
        """
        The forward method for reconstruction or pretraining.

        Args:
            x_categ (Tensor): [batch_size, num_categories, dim] or appropriate shape
            x_cont  (Tensor): [batch_size, num_continuous, dim] or appropriate shape

        Returns:
            cat_outs, con_outs: predicted reconstructions/outputs from 
            mlp1 for categorical features and mlp2 for continuous features.
        """

        # Pass the categorical and continuous embeddings into the Transformer 
        # (or RowColTransformer).
        x = self.transformer(x_categ, x_cont)
        # The output 'x' typically has shape [batch_size, (num_categories + num_continuous), dim]

        # We then slice out the first portion (categorical part) and pass it to mlp1
        cat_outs = self.mlp1(x[:, :self.num_categories, :])  
        # Then slice out the continuous part for mlp2
        con_outs = self.mlp2(x[:, self.num_categories:, :])

        return cat_outs, con_outs
