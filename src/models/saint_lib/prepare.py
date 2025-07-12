
import argparse

def argparse_prepare():


    # -------------------
    # ARGUMENT PARSER SETUP
    # -------------------
    
    parser = argparse.ArgumentParser()

    # # --dset_id is the OpenML dataset ID to load; required argument
    # parser.add_argument('--dset_id', required=True, type=int)

    # --vision_dset is a flag (boolean) that indicates if this is a vision dataset
    parser.add_argument('--vision_dset', action='store_true')

    # --task indicates the type of problem: 'binary', 'multiclass', or 'regression'; required
    parser.add_argument('--task', required=False, type=str, choices=['binary','multiclass','regression'])

    # --cont_embeddings chooses how continuous features are embedded. Options:
    # 'MLP', 'Noemb', 'pos_singleMLP'
    parser.add_argument('--cont_embeddings', default='MLP', type=str, choices=['MLP','Noemb','pos_singleMLP'])

    # --embedding_size is the dimensionality of embeddings for features
    parser.add_argument('--embedding_size', default=32, type=int)

    # --transformer_depth is the number of Transformer encoder layers
    parser.add_argument('--transformer_depth', default=6, type=int)

    # --attention_heads is the number of attention heads in the Multi-Head Attention
    parser.add_argument('--attention_heads', default=8, type=int)

    # --attention_dropout is the dropout applied to attention probabilities
    parser.add_argument('--attention_dropout', default=0.1, type=float)

    # --ff_dropout is the dropout applied in the feed-forward blocks
    parser.add_argument('--ff_dropout', default=0.1, type=float)

    # --attentiontype controls the type of attention used, e.g., 'colrow', 'col', etc.
    parser.add_argument('--attentiontype', default='colrow', type=str,
        choices=['col','colrow','row','justmlp','attn','attnmlp'])

    # --optimizer chooses the optimizer type, e.g., AdamW, Adam, or SGD
    parser.add_argument('--optimizer', default='AdamW', type=str, choices=['AdamW','Adam','SGD'])

    # --scheduler chooses how the learning rate is scheduled, either 'cosine' or 'linear'
    parser.add_argument('--scheduler', default='cosine', type=str, choices=['cosine','linear'])

    # --lr is the learning rate
    parser.add_argument('--lr', default=0.0001, type=float)

    # --epochs is the total number of epochs for training
    parser.add_argument('--epochs', default=100, type=int)

    # --batchsize is the batch size used in the DataLoader
    parser.add_argument('--batchsize', default=256, type=int)

    # --savemodelroot is the folder in which to save the trained model checkpoints
    parser.add_argument('--savemodelroot', default='./bestmodels', type=str)

    # --run_name is a string label for the run (helpful when logging with wandb)
    parser.add_argument('--run_name', default='testrun', type=str)

    # --set_seed sets a random seed for reproducibility
    parser.add_argument('--set_seed', default=1, type=int)

    # --dset_seed is another seed for dataset splitting
    parser.add_argument('--dset_seed', default=5, type=int)

    # --active_log is a flag indicating whether to log metrics to wandb
    parser.add_argument('--active_log', action='store_true')

    # --pretrain is a flag to indicate if the model should be pretrained
    parser.add_argument('--pretrain', action='store_true')

    # --pretrain_epochs is the number of epochs to pretrain
    parser.add_argument('--pretrain_epochs', default=50, type=int)

    # --pt_tasks lists the self-supervised learning tasks used during pretraining (contrastive, denoising, etc.)
    parser.add_argument('--pt_tasks', default=['contrastive','denoising'], type=str, nargs='*',
        choices=['contrastive','contrastive_sim','denoising'])

    # --pt_aug indicates which augmentations (e.g., mixup, cutmix) to apply during pretraining
    parser.add_argument('--pt_aug', default=[], type=str, nargs='*', choices=['mixup','cutmix'])

    # --pt_aug_lam is a hyperparameter for the augmentations' lamda value in pretraining
    parser.add_argument('--pt_aug_lam', default=0.1, type=float)

    # --mixup_lam is the lamda hyperparameter for mixup during training
    parser.add_argument('--mixup_lam', default=0.3, type=float)

    # --train_mask_prob is the probability of masking certain features during training
    parser.add_argument('--train_mask_prob', default=0, type=float)

    # --mask_prob is the probability of masking certain features generally
    parser.add_argument('--mask_prob', default=0, type=float)

    # --ssl_avail_y is used if certain self-supervised learning labels are available
    parser.add_argument('--ssl_avail_y', default=0, type=int)

    # --pt_projhead_style sets the style of projection head used for SSL tasks
    parser.add_argument('--pt_projhead_style', default='diff', type=str,
        choices=['diff','same','nohead'])

    # --nce_temp is a temperature hyperparameter for contrastive loss
    parser.add_argument('--nce_temp', default=0.7, type=float)

    # --lam0, lam1, lam2, lam3 are weighting hyperparameters used in the pretraining tasks
    parser.add_argument('--lam0', default=0.5, type=float)
    parser.add_argument('--lam1', default=10, type=float)
    parser.add_argument('--lam2', default=1, type=float)
    parser.add_argument('--lam3', default=10, type=float)

    # --final_mlp_style sets the style of the final MLP used in the model, 'common' or 'sep'
    parser.add_argument('--final_mlp_style', default='sep', type=str, choices=['common','sep'])

    # **IMPORTANT**: We parse with an empty list so no external arguments can be passed.
    opt = parser.parse_args(args=[])

    return opt
