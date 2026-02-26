# Beatrix_image.py
"""
Beatrix Defense (Original Image Domain)
===============================================================

This file contains the original image-domain implementation of the
Beatrix (BEAT) backdoor detection defense. It is included for reference
alongside the tabular adaptation (Beatrix_tabular.py).

This implementation operates on image classifiers (e.g., PreActResNet,
NetC_MNIST) and uses Gram-matrix-based feature correlations and KMMD
distance to detect backdoor attacks.

Note:
 - This file is the original image-domain code and is NOT used for
   tabular data experiments. See Beatrix_tabular.py for the tabular version.
"""

import sys
import os
import torch
import torchvision
import numpy as np
from sklearn.utils import shuffle
import torch.nn.functional as F
import skimage.io
import skimage.transform

import config

# Insert the parent directory into the system path so that modules can be imported from higher-level directories.
sys.path.insert(0, "../..")

# Import specific models and functions from the repository.
from classifier_models import PreActResNet18, ResNet18, PreActResNet34  # Different classifier models.
from dataloader import get_dataloader  # Function to load datasets.
from networks.models import Generator, NetC_MNIST  # Network modules: a Generator and a classifier for MNIST.
from utils import progress_bar  # Utility function for displaying progress bars during training/testing.


# =============================================================================
# Helper functions for creating backdoor targets and patterns
# =============================================================================
def create_targets_bd(targets, opt):
    """
    Create backdoor targets based on the attack mode.
    
    For "all2one" attack: All target labels are set to a specific target label (opt.target_label).
    For "all2all_mask" attack: Each label is cyclically shifted by one (i.e., (label+1)%num_classes).
    
    Args:
        targets (torch.Tensor): Original labels.
        opt (Namespace): Options including the attack mode, target label, device, etc.
        
    Returns:
        torch.Tensor: The new backdoor target labels.
    """
    if opt.attack_mode == "all2one":
        bd_targets = torch.ones_like(targets) * opt.target_label
    elif opt.attack_mode == "all2all_mask":
        bd_targets = torch.tensor([(label + 1) % opt.num_classes for label in targets])
    else:
        raise Exception("{} attack mode is not implemented".format(opt.attack_mode))
    return bd_targets.to(opt.device)


def create_bd(inputs, targets, netG, netM, opt):
    """
    Create backdoor examples from clean inputs using the generator (netG) and mask network (netM).
    
    The generator produces a pattern which is then normalized.
    The mask network outputs a mask (after thresholding) which determines where the pattern is applied.
    The backdoor input is then created by blending the original input and the generated pattern using the mask.
    
    Args:
        inputs (torch.Tensor): Clean input images.
        targets (torch.Tensor): Original target labels.
        netG (Generator): Generator network that creates trigger patterns.
        netM (Generator): Mask network that creates a mask for applying the trigger.
        opt (Namespace): Options including device, attack mode, etc.
    
    Returns:
        tuple: (backdoor inputs, new target labels, generated patterns, masks)
    """
    # Create target labels for the backdoor attack.
    bd_targets = create_targets_bd(targets, opt)
    # Generate trigger patterns using netG.
    patterns = netG(inputs)
    # Normalize the generated patterns.
    patterns = netG.normalize_pattern(patterns)

    # Get mask outputs from netM, thresholded to obtain a binary (or sparse) mask.
    masks_output = netM.threshold(netM(inputs))
    # Blend original inputs with trigger patterns according to the mask.
    bd_inputs = inputs + (patterns - inputs) * masks_output
    return bd_inputs, bd_targets, patterns, masks_output


def create_cross(inputs1, inputs2, netG, netM, opt):
    """
    Create a cross-domain backdoor example where the pattern from one input batch is applied to another.
    
    The process is similar to create_bd, but uses patterns generated from inputs2 and applies them to inputs1.
    
    Args:
        inputs1 (torch.Tensor): Base images to be modified.
        inputs2 (torch.Tensor): Images from which the trigger pattern is generated.
        netG (Generator): Generator network.
        netM (Generator): Mask network.
        opt (Namespace): Options.
    
    Returns:
        tuple: (cross inputs, generated patterns from inputs2, masks)
    """
    # Generate pattern from the second set of inputs.
    patterns2 = netG(inputs2)
    # Normalize the generated pattern.
    patterns2 = netG.normalize_pattern(patterns2)
    # Obtain thresholded mask from netM for inputs2.
    masks_output = netM.threshold(netM(inputs2))
    # Create cross examples by applying the generated pattern (and mask) to inputs1.
    inputs_cross = inputs1 + (patterns2 - inputs1) * masks_output
    return inputs_cross, patterns2, masks_output


# =============================================================================
# Class to capture intermediate activations from a specific layer using hooks
# =============================================================================
class LayerActivations:
    def __init__(self, model, opt):
        """
        Initialize the hook for capturing activations from a specific layer.
        
        Args:
            model (torch.nn.Module): The neural network model from which activations are captured.
            opt (Namespace): Options (may include device or other configurations).
        """
        self.opt = opt
        self.model = model
        self.model.eval()  # Set the model to evaluation mode.
        self.build_hook()  # Register the hook on the desired layer.

    def build_hook(self):
        """
        Build the forward hook on the specified layer.
        
        In this case, we attach the hook to the 'layer4' module if it is a torch.nn.Sequential.
        """
        for name, m in self.model.named_children():
            # Attach hook only to 'layer4' which is assumed to be a sequential module.
            if isinstance(m, torch.nn.Sequential) and name == 'layer4':
                self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        """
        Hook function that captures the input of the hooked layer.
        
        Args:
            module (torch.nn.Module): The module where the hook is attached.
            input (tuple): The input to the module.
            output (torch.Tensor): The output from the module.
        """
        # Store the input tensor of the layer as the features.
        self.features = input[0]

    def remove_hook(self):
        """
        Remove the forward hook to avoid memory leaks.
        """
        self.hook.remove()

    def run_hook(self, x):
        """
        Run the model on input 'x' to capture and return the features from the hooked layer.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Captured intermediate features from the specified layer.
        """
        self.model(x)
        # Optionally remove the hook after one run (commented out here).
        # self.remove_hook()
        return self.features


# =============================================================================
# Evaluation function for the backdoor detection
# =============================================================================
def eval(netC, netG, netM, test_dl1, test_dl2, opt):
    """
    Evaluate the classifier (netC) on clean, backdoor, and cross inputs, and record intermediate features.
    
    This function performs evaluation over the test dataset, computes accuracies,
    extracts intermediate activations, and saves a few backdoor example images for visualization.
    
    Args:
        netC (torch.nn.Module): The classifier network.
        netG (Generator): The generator network used for creating patterns.
        netM (Generator): The mask network.
        test_dl1 (DataLoader): DataLoader for the first set of test data.
        test_dl2 (DataLoader): DataLoader for the second set of test data.
        opt (Namespace): Options including device, dataset, attack mode, etc.
    
    Returns:
        dict: A dictionary containing concatenated clean features, backdoor features, and labels.
    """
    print(" Eval:")

    # Number of batches and images to output visualizations for.
    n_output_batches = 3
    n_output_images = 3

    # Initialize counters for sample counts and correct predictions.
    total_sample = 0
    total_correct_clean = 0
    total_correct_bd = 0
    total_correct_cross = 0

    # Lists to store features and labels for later analysis.
    clean_feature = []
    bd_feature = []
    cross_feature = []
    ori_label = []
    bd_label = []

    # Create an instance of LayerActivations to capture features from the classifier.
    intermedia_feature = LayerActivations(netC.to(opt.device), opt)

    # Loop through the test dataloaders (zipping two dataloaders together).
    for batch_idx, (inputs, targets), (inputs2, targets2) in zip(range(len(test_dl1)), test_dl1, test_dl2):
        # Move the data to the specified device.
        inputs1, targets1 = inputs.to(opt.device), targets.to(opt.device)
        inputs2, targets2 = inputs2.to(opt.device), targets2.to(opt.device)
        bs = inputs1.shape[0]
        total_sample += bs

        # --------------------------
        # Evaluate on clean inputs.
        preds_clean = netC(inputs1)
        correct_clean = torch.sum(torch.argmax(preds_clean, 1) == targets1)
        total_correct_clean += correct_clean
        acc_clean = total_correct_clean * 100.0 / total_sample

        # --------------------------
        # Evaluate on backdoor examples.
        if opt.attack_mode == "all2one":
            # Generate backdoor inputs and corresponding target labels.
            inputs_bd, targets_bd, _, _ = create_bd(inputs1, targets1, netG, netM, opt)
            preds_bd = netC(inputs_bd)
            correct_bd = torch.sum(torch.argmax(preds_bd, 1) == targets_bd)
            total_correct_bd += correct_bd
            acc_bd = total_correct_bd * 100.0 / total_sample
            # Create cross examples using inputs from test_dl1 and test_dl2.
            inputs_cross, _, _ = create_cross(inputs1, inputs2, netG, netM, opt)
            preds_cross = netC(inputs_cross)
            correct_cross = torch.sum(torch.argmax(preds_cross, 1) == targets1)
            total_correct_cross += correct_cross
            acc_cross = total_correct_cross * 100.0 / total_sample
        else:
            raise Exception("Invalid attack mode")

        # --------------------------
        # Capture intermediate features using the hook.
        clean_feature.append(intermedia_feature.run_hook(inputs1).to(torch.device('cpu')))
        bd_feature.append(intermedia_feature.run_hook(inputs_bd).to(torch.device('cpu')))
        ori_label.append(torch.argmax(preds_clean, 1).to(torch.device('cpu')))
        bd_label.append(torch.argmax(preds_bd, 1).to(torch.device('cpu')))

        # Display progress and accuracies using a progress bar.
        progress_bar(
            batch_idx,
            len(test_dl1),
            "Acc Clean: {:.3f} | Acc Bd: {:.3f} | Acc Cross: {:.3f}".format(acc_clean, acc_bd, acc_cross),
        )

        # Save some backdoor images for visualization if within the first few batches.
        if batch_idx < n_output_batches:
            # Create directory to save images if it does not exist.
            dir_temps = os.path.join(opt.temps, opt.dataset)
            if not os.path.exists(dir_temps):
                os.makedirs(dir_temps)
            subs = []
            for i in range(n_output_images):
                subs.append(inputs_bd[i : (i + 1), :, :, :])
            # Concatenate images along width for visualization.
            images = netG.denormalize_pattern(torch.cat(subs, dim=3))
            file_name = "%s_%s_sample_%d.png" % (opt.dataset, opt.attack_mode, batch_idx)
            file_path = os.path.join(dir_temps, file_name)
            torchvision.utils.save_image(images, file_path, normalize=True, pad_value=1)

    # Concatenate features and labels from all batches.
    data = {
        "clean_feature": torch.cat(clean_feature, dim=0),
        "bd_feature": torch.cat(bd_feature, 0),
        "ori_label": torch.cat(ori_label, 0),
        "bd_label": torch.cat(bd_label, 0),
    }

    # Optionally save the features data for later use.
    if opt.save_feature_data:
        dir_data = 'feature_data/'
        ckpt_folder = os.path.join(dir_data, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        ckpt_path = os.path.join(ckpt_folder, 'data.pt')
        torch.save(data, ckpt_path)
    return data


# =============================================================================
# Main training function (mostly for setting up models and loading data)
# =============================================================================
def train(opt):
    """
    Set up models, load pre-trained weights, and run evaluation to extract features.
    
    Depending on the dataset, the appropriate classifier model is instantiated.
    Then, the pre-trained weights for the classifier, generator, and mask networks are loaded.
    Finally, data is evaluated and features are extracted.
    
    Args:
        opt (Namespace): Options and hyperparameters.
    
    Returns:
        dict: Extracted features and labels.
    """
    # --------------------------
    # Prepare classifier model based on dataset.
    if opt.dataset == "cifar10":
        netC = PreActResNet18().to(opt.device)
    elif opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=43).to(opt.device)
    elif opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)
    else:
        raise Exception("Invalid dataset")

    # --------------------------
    # Set up TensorBoard logging directory (if using tensorboard, currently commented out).
    log_dir = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_dir = os.path.join(log_dir, "log_dir")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # tf_writer = SummaryWriter(log_dir=log_dir)

    # --------------------------
    # Load pre-trained checkpoint for the classifier and backdoor networks.
    ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
    ckpt_path = os.path.join(ckpt_folder, "{}_{}_ckpt.pth.tar".format(opt.attack_mode, opt.dataset))

    # Load state dictionary from the checkpoint.
    state_dict = torch.load(ckpt_path, map_location=opt.device)
    print("load C")
    netC.load_state_dict(state_dict["netC"])
    netC.to(opt.device)
    netC.eval()  # Set classifier to evaluation mode.
    netC.requires_grad_(False)
    
    print("load G")
    netG = Generator(opt)
    netG.load_state_dict(state_dict["netG"])
    netG.to(opt.device)
    netG.eval()
    netG.requires_grad_(False)
    
    print("load M")
    netM = Generator(opt, out_channels=1)  # The mask network; note out_channels=1.
    netM.load_state_dict(state_dict["netM"])
    netM.to(opt.device)
    netM.eval()
    netM.requires_grad_(False)

    # --------------------------
    # Overwrite some options for evaluation.
    opt.n_iters = 1
    opt.batchsize = 256
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------
    # Prepare test dataloaders.
    test_dl1 = get_dataloader(opt, train=False)
    test_dl2 = get_dataloader(opt, train=False)

    # --------------------------
    # Check if features data has been previously saved; if yes, load it, otherwise run evaluation.
    dir_data = 'feature_data/'
    ckpt_folder = os.path.join(dir_data, opt.dataset, opt.attack_mode, 'target_' + str(opt.target_label))
    ckpt_path = os.path.join(ckpt_folder, 'data.pt')
    if os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=opt.device)
    else:
        data = eval(netC, netG, netM, test_dl1, test_dl2, opt)

    return data


# =============================================================================
# Create a dataset for evaluating the backdoor detection using clean and poisoned samples.
# =============================================================================
def bd_dataset(X_all, y_all, n_poison=100, num_class=10, target_class=[], source_class=[1], balance=True):
    """
    Prepare a dataset for backdoor detection evaluation.
    
    This function extracts a balanced set of clean and poisoned examples based on the given labels.
    It uses a specified number of examples (n_poison) per class.
    
    Args:
        X_all (np.array or torch.Tensor): All feature data.
        y_all (np.array or torch.Tensor): Corresponding labels.
        n_poison (int): Number of poison samples to extract.
        num_class (int): Total number of classes.
        target_class (list): Classes that are considered target classes (to be excluded from clean set).
        source_class (list): Classes that are considered for poisoning.
        balance (bool): Whether to balance the dataset across classes.
    
    Returns:
        tuple: (clean samples, poison labels)
    """
    # If labels are one-hot encoded, convert them to class indices.
    if len(y_all.shape) > 1:
        print('y_all.shape:', y_all.shape)
        labels = np.argmax(y_all, axis=-1)
    else:
        labels = y_all

    # All class labels in the training set.
    train_classes = np.arange(num_class)
    clean_x = []
    clean_y = []
    poison_y = []
    if balance:
        # Loop through each class and select samples for poisoning.
        for tc in train_classes:
            if tc in target_class:
                continue
            if tc not in source_class:
                continue
            # Select n_poison samples from clean data for class tc.
            index = np.where(labels == tc)
            clean_x.append(X_all[index][0:n_poison])
            clean_y.append(y_all[index][0:n_poison])
            # For poisoned labels, select samples from the target class.
            poison_index = np.where(labels == target_class[0])
            poison_y.append(y_all[poison_index][0:n_poison])
        clean_x = torch.cat(clean_x, dim=0)
        clean_y = torch.cat(clean_y, dim=0)
        poison_y = torch.cat(poison_y, dim=0)
    else:
        # If not balancing, simply select non-target and target examples.
        index = np.where(labels != target_class[0])
        clean_x.append(X_all[index][0:n_poison])
        clean_y.append(y_all[index][0:n_poison])
        poison_index = np.where(labels == target_class[0])
        poison_y.append(y_all[poison_index][0:n_poison])
        clean_x = torch.cat(clean_x, dim=0)
        clean_y = torch.cat(clean_y, dim=0)
        poison_y = torch.cat(poison_y, dim=0)

    return clean_x, poison_y


# =============================================================================
# Gaussian Kernel functions and Maximum Mean Discrepancy (MMD)
# =============================================================================
def gaussian_kernel(x1, x2, kernel_mul=2.0, kernel_num=5, fix_sigma=0, mean_sigma=0):
    """
    Compute a multi-scale Gaussian kernel between two sets of samples.
    
    This function computes the pairwise L2 distances between samples in x1 and x2,
    then applies multiple Gaussian kernels with different bandwidths.
    
    Args:
        x1 (torch.Tensor): First set of samples.
        x2 (torch.Tensor): Second set of samples.
        kernel_mul (float): Multiplicative factor for adjusting the bandwidth.
        kernel_num (int): Number of different Gaussian kernels to sum.
        fix_sigma (float): If provided, fixes the sigma value.
        mean_sigma (float): If provided, uses the mean L2 distance as sigma.
    
    Returns:
        torch.Tensor: Sum of the computed Gaussian kernel values.
    """
    # Get the sample sizes.
    x1_sample_size = x1.shape[0]
    x2_sample_size = x2.shape[0]
    x1_tile_shape = []
    x2_tile_shape = []
    norm_shape = []
    # Create tiling shapes for broadcasting differences.
    for i in range(len(x1.shape) + 1):
        if i == 1:
            x1_tile_shape.append(x2_sample_size)
        else:
            x1_tile_shape.append(1)
        if i == 0:
            x2_tile_shape.append(x1_sample_size)
        else:
            x2_tile_shape.append(1)
        if not (i == 0 or i == 1):
            norm_shape.append(i)

    # Tile x1 and x2 to compute pairwise differences.
    tile_x1 = torch.unsqueeze(x1, 1).repeat(x1_tile_shape)
    tile_x2 = torch.unsqueeze(x2, 0).repeat(x2_tile_shape)
    # Compute L2 distance between all pairs.
    L2_distance = torch.square(tile_x1 - tile_x2).sum(dim=norm_shape)
    # Determine the bandwidth for the Gaussian kernel.
    if fix_sigma:
        bandwidth = fix_sigma
    elif mean_sigma:
        bandwidth = torch.mean(L2_distance)
    else:  # Use median distance if no sigma is provided.
        bandwidth = torch.median(L2_distance.reshape(L2_distance.shape[0], -1))
    bandwidth /= kernel_mul ** (kernel_num // 2)
    # Generate a list of bandwidths for multiple kernels.
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    print(bandwidth_list)
    # Compute the Gaussian kernel values for each bandwidth and sum them.
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def kmmd_dist(x1, x2):
    """
    Compute the Kernel Maximum Mean Discrepancy (KMMD) distance between two sets of features.
    
    KMMD is used as a statistical distance measure between distributions.
    
    Args:
        x1 (torch.Tensor): First set of features.
        x2 (torch.Tensor): Second set of features.
    
    Returns:
        float: The computed KMMD distance.
    """
    # Concatenate both feature sets.
    X_total = torch.cat([x1, x2], 0)
    # Compute the Gram matrix using the Gaussian kernel.
    Gram_matrix = gaussian_kernel(X_total, X_total, kernel_mul=2.0, kernel_num=2, fix_sigma=0, mean_sigma=0)
    n = int(x1.shape[0])
    m = int(x2.shape[0])
    # Split the Gram matrix into parts corresponding to each distribution.
    x1x1 = Gram_matrix[:n, :n]
    x2x2 = Gram_matrix[n:, n:]
    x1x2 = Gram_matrix[:n, n:]
    # Compute the MMD distance.
    diff = torch.mean(x1x1) + torch.mean(x2x2) - 2 * torch.mean(x1x2)
    diff = (m * n) / (m + n) * diff
    return diff.to(torch.device('cpu')).numpy()


# =============================================================================
# Feature Correlations for backdoor detection
# =============================================================================
class Feature_Correlations:
    def __init__(self, POWER_list, mode='mad'):
        """
        Initialize the Feature_Correlations detector.
        
        Args:
            POWER_list (list): List of power orders for computing Gram matrices.
            mode (str): Mode for deviation calculation; default is 'mad' (median absolute deviation).
        """
        self.power = POWER_list
        self.mode = mode

    def train(self, in_data):
        """
        Train the detector by computing the medians and MADs of the features.
        
        Args:
            in_data (list): List containing feature tensors used for training.
        """
        self.in_data = in_data
        if 'mad' in self.mode:
            # Compute median and MAD (median absolute deviation) for each order in POWER_list.
            self.medians, self.mads = self.get_median_mad(self.in_data)
            # Also compute min and max thresholds based on the MAD.
            self.mins, self.maxs = self.minmax_mad()

    def minmax_mad(self):
        """
        Compute minimum and maximum thresholds based on medians and MADs.
        
        The thresholds are set as median ± 10 * MAD.
        
        Returns:
            tuple: (mins, maxs) lists for each feature.
        """
        mins = []
        maxs = []
        # Loop over each layer's medians and MADs.
        for L, mm in enumerate(zip(self.medians, self.mads)):
            medians, mads = mm[0], mm[1]
            if L == len(mins):
                mins.append([None] * len(self.power))
                maxs.append([None] * len(self.power))
            for p, P in enumerate(self.power):
                mins[L][p] = medians[p] - mads[p] * 10
                maxs[L][p] = medians[p] + mads[p] * 10
        return mins, maxs

    def G_p(self, ob, p):
        """
        Compute the Gram matrix raised to the power p for a given feature tensor.
        
        This function applies element-wise power, reshapes the tensor,
        computes a matrix multiplication to obtain second-order statistics, and then processes
        the result to obtain a flattened representation.
        
        Args:
            ob (torch.Tensor): Input feature tensor.
            p (int or float): The power to raise the feature elements.
        
        Returns:
            torch.Tensor: The processed Gram matrix features.
        """
        temp = ob.detach()
        temp = temp ** p
        temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
        temp = (torch.matmul(temp, temp.transpose(dim0=2, dim1=1)))
        # Consider only the upper triangular part of the matrix.
        temp = temp.triu()
        # Restore the original scale by taking the p-th root.
        temp = temp.sign() * torch.abs(temp) ** (1 / p)
        temp = temp.reshape(temp.shape[0], -1)
        # Record the number of features (divided by 2 because of symmetry).
        self.num_feature = temp.shape[-1] / 2
        return temp

    def get_median_mad(self, feat_list):
        """
        Compute the median and MAD for each feature in the list.
        
        Args:
            feat_list (list): List of feature tensors.
        
        Returns:
            tuple: Two lists containing medians and MADs respectively for each power order.
        """
        medians = []
        mads = []
        for L, feat_L in enumerate(feat_list):
            if L == len(medians):
                medians.append([None] * len(self.power))
                mads.append([None] * len(self.power))
            for p, P in enumerate(self.power):
                g_p = self.G_p(feat_L, P)
                current_median = g_p.median(dim=0, keepdim=True)[0]
                current_mad = torch.abs(g_p - current_median).median(dim=0, keepdim=True)[0]
                medians[L][p] = current_median
                mads[L][p] = current_mad
        return medians, mads

    def get_deviations_(self, feat_list):
        """
        Compute deviations using a threshold-based approach on the Gram matrices.
        
        Deviations are computed by summing the relative differences when values fall outside
        the [min, max] range.
        
        Args:
            feat_list (list): List of feature tensors.
        
        Returns:
            np.array: Array of deviation values for each sample.
        """
        deviations = []
        batch_deviations = []
        for L, feat_L in enumerate(feat_list):
            dev = 0
            for p, P in enumerate(self.power):
                g_p = self.G_p(feat_L, P)
                dev += (F.relu(self.mins[L][p] - g_p) / torch.abs(self.mins[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
                dev += (F.relu(g_p - self.maxs[L][p]) / torch.abs(self.maxs[L][p] + 10 ** -6)).sum(dim=1, keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations, axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0) / self.num_feature / len(self.power)
        return deviations

    def get_deviations(self, feat_list):
        """
        Compute deviations as the absolute difference from the median, normalized by MAD.
        
        Args:
            feat_list (list): List of feature tensors.
        
        Returns:
            np.array: Normalized deviation values for each sample.
        """
        deviations = []
        batch_deviations = []
        for L, feat_L in enumerate(feat_list):
            dev = 0
            for p, P in enumerate(self.power):
                g_p = self.G_p(feat_L, P)
                dev += torch.sum(torch.abs(g_p - self.medians[L][p]) / (self.mads[L][p] + 1e-6), dim=1, keepdim=True)
            batch_deviations.append(dev.cpu().detach().numpy())
        batch_deviations = np.concatenate(batch_deviations, axis=1)
        deviations.append(batch_deviations)
        deviations = np.concatenate(deviations, axis=0) / self.num_feature / len(self.power)
        return deviations


# =============================================================================
# Function to determine detection thresholds based on clean features.
# =============================================================================
def threshold_determine(clean_feature_target, ood_detection):
    """
    Determine threshold values (95th and 99th percentiles) for out-of-distribution (OOD) detection.
    
    The clean features are partitioned, and the deviations are computed to determine
    appropriate thresholds.
    
    Args:
        clean_feature_target (torch.Tensor): Clean feature representations for a specific target class.
        ood_detection (Feature_Correlations): The detector instance.
    
    Returns:
        tuple: (95th percentile threshold, 99th percentile threshold)
    """
    test_deviations_list = []
    step = 5  # Divide the data into 5 steps.
    for i in range(step):
        # Create a mask to split the data.
        index_mask = np.ones((len(clean_feature_target),))
        index_mask[i * int(len(clean_feature_target) // step):(i + 1) * int(len(clean_feature_target) // step)] = 0
        clean_feature_target_train = clean_feature_target[np.where(index_mask == 1)]
        clean_feature_target_test = clean_feature_target[np.where(index_mask == 0)]
        # Train the OOD detector on the training split.
        ood_detection.train(in_data=[clean_feature_target_train])
        # Get deviation values for the test split.
        test_deviations = ood_detection.get_deviations_([clean_feature_target_test])
        test_deviations_list.append(test_deviations)
    test_deviations = np.concatenate(test_deviations_list, 0)
    test_deviations_sort = np.sort(test_deviations, 0)
    # Determine thresholds at the 95th and 99th percentiles.
    percentile_95 = test_deviations_sort[int(len(test_deviations_sort) * 0.95)][0]
    percentile_99 = test_deviations_sort[int(len(test_deviations_sort) * 0.99)][0]
    print(f'percentile_95:{percentile_95}')
    print(f'percentile_99:{percentile_99}')
    # Optionally, one could visualize the histogram here.
    return percentile_95, percentile_99


# =============================================================================
# BEAT Detector class for evaluating backdoor detection.
# =============================================================================
class BEAT_detector():
    def __init__(self, opt, clean_test=500, bd_test=500, order_list=np.arange(1, 9)):
        """
        Initialize the BEAT detector.
        
        Args:
            opt (Namespace): Options containing dataset, target label, etc.
            clean_test (int): Number of clean samples to test.
            bd_test (int): Number of backdoor (poisoned) samples to test.
            order_list (np.array): Array of orders (powers) used in the Gram matrix computations.
        """
        self.opt = opt
        self.test_target_label = opt.target_label
        self.order_list = order_list
        # Set the number of clean samples per class depending on the dataset.
        if opt.dataset == 'cifar10':
            self.clean_data_perclass = 30
        elif opt.dataset == 'gtsrb':
            self.clean_data_perclass = 30
        else:
            raise Exception("Invalid dataset")

        self.clean_test = clean_test
        self.bd_test = bd_test

    def _detecting(self, data):
        """
        Run the backdoor detection procedure using the computed features.
        
        This involves:
         - Shuffling the clean and backdoor features.
         - Training the OOD detection method (Feature_Correlations).
         - Determining thresholds for each class.
         - Computing deviations for clean and backdoor samples.
         - Reporting false positives/negatives.
         - Computing a distance metric (KMMD) between clean and backdoor feature groups.
         - Saving the results.
        
        Args:
            data (dict): Dictionary containing features and labels extracted from eval().
        """
        opt = self.opt
        # Retrieve features and labels from the input data.
        clean_feature = data['clean_feature'].to(opt.device)
        bd_feature = data['bd_feature'].to(opt.device)
        ori_label = data['ori_label'].cpu()
        bd_label = data['bd_label'].cpu()

        # Shuffle the features and labels to randomize the order.
        (clean_feature, bd_feature, ori_label, bd_label) = shuffle(clean_feature, bd_feature, ori_label, bd_label)

        ##### Use gram-matrix based OOD detection.
        ood_detection = Feature_Correlations(POWER_list=self.order_list, mode='mad')

        J_t = []  # List to store the KMMD distance for each class.
        threshold_list = []  # List to store threshold values for each class.

        # Iterate over each class to perform detection.
        for test_target_label in range(opt.num_classes):
            print(f'*****class:{test_target_label}*****')
            # Extract features for the current class.
            clean_feature_target = clean_feature[np.where(ori_label == test_target_label)]
            clean_feature_defend = clean_feature_target[:self.clean_data_perclass]

            # Determine thresholds based on a subset of clean features.
            threshold_95, threshold_99 = threshold_determine(clean_feature_defend, ood_detection)
            threshold_list.append([test_target_label, threshold_95, threshold_99])

            # Train the OOD detector on the clean features for the current class.
            ood_detection.train(in_data=[clean_feature_defend])
            clean_feature_test = clean_feature[np.where(ori_label == test_target_label)][-self.clean_test:]
            clean_label_test = np.zeros((clean_feature_test.shape[0],))

            # If the current class is the backdoor target class, also prepare poisoned samples.
            if test_target_label == opt.target_label:
                bd_feature_test, _ = bd_dataset(bd_feature, ori_label, n_poison=self.bd_test,
                                                num_class=opt.num_classes, target_class=[test_target_label],
                                                source_class=list(np.arange(opt.num_classes)), balance=False)
                bd_label_test = np.ones((bd_feature_test.shape[0],))
                # Combine clean and poisoned features for testing.
                feature_test = torch.cat([clean_feature_test, bd_feature_test], 0)
                label_test = np.concatenate([clean_label_test, bd_label_test], 0)

                # Compare deviations of backdoor and clean features.
                clean_deviations_sort = np.sort(ood_detection.get_deviations_([clean_feature_test]), 0)
                bd_deviations_sort = np.sort(ood_detection.get_deviations_([bd_feature_test]), 0)
                percentile_95 = np.where(bd_deviations_sort > clean_deviations_sort[int(len(clean_deviations_sort) * 0.95)], 1, 0)
                print(f'percentile_95:{clean_deviations_sort[int(len(clean_deviations_sort) * 0.95)]},TP95:{percentile_95.sum() / len(bd_deviations_sort)}')
                percentile_99 = np.where(bd_deviations_sort > clean_deviations_sort[int(len(clean_deviations_sort) * 0.99)], 1, 0)
                print(f'percentile_99:{clean_deviations_sort[int(len(clean_deviations_sort) * 0.99)]},TP99:{percentile_99.sum() / len(bd_deviations_sort)}')
            else:
                # For non-target classes, only use clean features.
                feature_test = clean_feature_test
                label_test = clean_label_test

            # Get deviation values for the test set.
            test_deviations = ood_detection.get_deviations_([feature_test])
            # Generate binary OOD labels based on the thresholds.
            ood_label_95 = np.where(test_deviations > threshold_95, 1, 0).squeeze()
            ood_label_99 = np.where(test_deviations > threshold_99, 1, 0).squeeze()

            # Calculate false negatives and false positives.
            false_negetive_95 = np.where(label_test - ood_label_95 > 0, 1, 0).squeeze()
            false_negetive_99 = np.where(label_test - ood_label_99 > 0, 1, 0).squeeze()
            false_positive_95 = np.where(label_test - ood_label_95 < 0, 1, 0).squeeze()
            false_positive_99 = np.where(label_test - ood_label_99 < 0, 1, 0).squeeze()

            print(f'false_negetive_95:{false_negetive_95.sum()},false_negetive_99:{false_negetive_99.sum()}')
            print(f'false_positive_95:{false_positive_95.sum()},false_positive_99：{false_positive_99.sum()}')

            # Group features based on the OOD detection results.
            clean_feature_group = feature_test[np.where(ood_label_95 == 0)]
            bd_feature_group = feature_test[np.where(ood_label_95 == 1)]

            # Flatten the grouped features by taking the mean over spatial dimensions.
            clean_feature_flat = torch.mean(clean_feature_group, dim=(2, 3))
            bd_feature_flat = torch.mean(bd_feature_group, dim=(2, 3))
            if bd_feature_flat.shape[0] < 1:
                kmmd = np.array([0.0])
            else:
                # Compute the KMMD distance between clean and backdoor features.
                kmmd = kmmd_dist(clean_feature_flat, bd_feature_flat)
            print(f'KMMD:{kmmd.item()}.')
            J_t.append(kmmd.item())

        print(J_t)
        # Convert list of distances to a NumPy array.
        J_t = np.asarray(J_t)
        # Compute the median and median absolute deviation (MAD) of the distances.
        J_t_median = np.median(J_t)
        J_MAD = np.median(np.abs(J_t - J_t_median))
        # Compute a standardized score (J_star) for each class.
        J_star = np.abs(J_t - J_t_median) / 1.4826 / (J_MAD + 1e-6)
        [print('%.2f' % (J_star_i)) for i, J_star_i in enumerate(J_star)]
        # Save the results.
        self._save_result_to_dir(result=[J_star])

    def _save_result_to_dir(self, result):
        """
        Save the detection results to a file.
        
        Args:
            result (list): List of results to be saved.
        """
        opt = self.opt
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, opt.attack_mode, 'target_' + str(opt.target_label))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        output_path = os.path.join(result_path, "{}_{}_output.txt".format(opt.attack_mode, opt.dataset))
        with open(output_path, "w+") as f:
            J_star_to_save = [str(value) for value in result[0]]
            f.write(", ".join(J_star_to_save) + "\n")


# =============================================================================
# Main function to run the detection pipeline.
# =============================================================================
def main(k):
    """
    Main routine that sets up the configuration, loads data, and runs the BEAT detector.
    
    Args:
        k (int): Target label for the backdoor attack.
    """
    # Parse command-line arguments.
    opt = config.get_argument().parse_args()
    # Set device (GPU if available, otherwise CPU).
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(opt.device)
    # Set the target label for this run.
    opt.target_label = k
    print('-' * 50 + 'opt.target_label:', opt.target_label)
    
    # Set the number of classes based on the dataset.
    if opt.dataset == "mnist" or opt.dataset == "cifar10":
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    else:
        raise Exception("Invalid Dataset")

    # Set image dimensions based on the dataset.
    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    else:
        raise Exception("Invalid Dataset")

    # Train/evaluate the model to extract features.
    data = train(opt)
    # Initialize the BEAT detector with the configuration.
    beat_detector = BEAT_detector(opt)
    # Run the detection process.
    beat_detector._detecting(data)


# =============================================================================
# Entry point for script execution.
# =============================================================================
if __name__ == "__main__":
    # Parse arguments for configuration.
    opt = config.get_argument().parse_args()
    # Set the visible GPU device.
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
    # Run the detection for each target label (here, iterating over 10 labels).
    for k in range(10):
        main(k)

## Example usage:
## python Beatrix.py --dataset cifar10 --gpu 0
