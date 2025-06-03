import torch
import torch.nn as nn
import os # For path joining if using torchvision example

# --- Configuration --- 
# !!! USER ACTION REQUIRED: Update these paths !!!
CHECKPOINT_PATH = "path/to/your/vlg_cbm_cub_checkpoint.pth" # e.g., saved_models/vlg_cbm_cub_nec5/best_model.pth
# CONFIG_PATH = "path/to/your/configs/cub.json" # May be needed for model loading by VLG-CBM's utilities
CUB_DATASET_ROOT = "path/to/your/datasets/CUB_200_2011" # This should point to the root of the CUB dataset folder
# !!! END USER ACTION REQUIRED !!!

BATCH_SIZE = 32 # Or as per original evaluation settings in VLG-CBM
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_TOP_CONCEPTS = 5 # As per Table 4 experiment description

# --- Placeholder Helper Functions --- 
def load_model_from_checkpoint(checkpoint_path, device):
    """
    Loads the VLG-CBM model from a checkpoint.
    This is a CRITICAL placeholder. You MUST adapt this function based on how models 
    are defined, instantiated, and loaded in the VLG-CBM project.
    The VLG-CBM project likely has utility functions or specific classes for this.
    Consult 'train_cbm.py' or similar scripts in the VLG-CBM repository for guidance.
    """
    print(f"Attempting to load model from: {checkpoint_path}")
    
    # --- BEGIN CRITICAL PLACEHOLDER CODE --- 
    # This section needs to be replaced with actual VLG-CBM model loading logic.
    # For demonstration, we define a very simple nn.Module. 
    # The actual VLG-CBM model will be significantly more complex.
    class PlaceholderVLG_CBM(nn.Module):
        def __init__(self, num_input_features=1024, num_concepts=312, num_classes=200):
            super().__init__()
            # These are dummy layers. The VLG-CBM will have a proper backbone, concept layer, etc.
            self.feature_extractor = nn.Linear(num_input_features, 512) # Dummy backbone
            self.concept_predictor = nn.Linear(512, num_concepts) # Dummy concept layer
            # This is assumed to be the layer whose weights are W_F
            self.final_linear_layer = nn.Linear(num_concepts, num_classes) 
            print("Initialized PlaceholderVLG_CBM. YOU MUST REPLACE THIS WITH THE ACTUAL VLG-CBM MODEL.")

        def forward(self, x):
            # Dummy forward pass. Assumes x is image features or similar.
            # The actual forward pass will depend on the VLG-CBM architecture.
            # For this script, the key is that `model.final_linear_layer` exists and is used.
            # If x is raw image, it needs to go through a full backbone.
            # For simplicity, let's assume x is already processed to some extent.
            if x.dim() > 2: # If x looks like batch of images (B, C, H, W)
                x = torch.flatten(x, 1) # Flatten to (B, Features)
            if x.shape[-1] != self.feature_extractor.in_features:
                 # Crude resizing if input features don't match dummy layer
                 x = x[:, :self.feature_extractor.in_features] 
            
            # This is a simplified flow. Real CBMs have specific paths.
            # features = self.feature_extractor(x) 
            # concepts = self.concept_predictor(features)
            # logits = self.final_linear_layer(concepts)
            # For this placeholder, we'll just pass through a dummy concept layer if needed
            # to ensure the final_linear_layer gets an input of the correct num_concepts dimension.
            if x.shape[-1] != self.final_linear_layer.in_features:
                # If x is not already concept-like, pass through a dummy concept predictor
                dummy_concepts = torch.randn(x.shape[0], self.final_linear_layer.in_features).to(x.device)
                logits = self.final_linear_layer(dummy_concepts)
            else:
                logits = self.final_linear_layer(x) # Assume x is already concept activations
            return logits

    # Replace 'PlaceholderVLG_CBM' with the actual model class from the VLG-CBM project.
    model = PlaceholderVLG_CBM().to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}. Using randomly initialized placeholder model.")
        print("Pruning results will not be meaningful.")
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # The structure of the checkpoint dictionary (e.g., 'model_state_dict', 'state_dict')
            # depends on how VLG-CBM saves models. Adjust accordingly.
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Successfully loaded model weights from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}. Using randomly initialized placeholder model.")
            print("Pruning results will not be meaningful.")
    # --- END CRITICAL PLACEHOLDER CODE --- 
    
    model.eval()
    return model

def get_cub_test_dataloader(cub_dataset_root, batch_size):
    """
    Creates a DataLoader for the CUB-200-2011 test set.
    This is a CRITICAL placeholder. You MUST adapt this function using VLG-CBM's 
    data loading utilities (e.g., from 'data.py' or 'datasets.py' in VLG-CBM).
    The VLG-CBM project should have specific transforms and dataset class for CUB.
    """
    print(f"Attempting to set up CUB test dataloader from: {cub_dataset_root}")
    
    # --- BEGIN CRITICAL PLACEHOLDER CODE ---
    # This section needs to be replaced with actual CUB dataloader setup from VLG-CBM.
    # For demonstration, this creates a loader with dummy data.
    from torchvision import transforms # Keep for potential use with actual dataset
    
    # Example transforms (VLG-CBM will have its own specific ones):
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224), # Or input size used by VLG-CBM
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    
    # The CUB test set typically resides in a subdirectory like 'test' or is identified by a list file.
    # test_dataset_path = os.path.join(cub_dataset_root, 'images') # This is just an example path structure
    # if not os.path.isdir(test_dataset_path):
    #     print(f"Error: CUB dataset not found or 'images' subdirectory missing at {test_dataset_path}.")
    #     print("Using dummy dataloader. Pruning results will not be meaningful.")
    #     # Fallback to dummy data if path is invalid
    
    # Using dummy data for placeholder functionality:
    num_test_samples = 5794 # Actual CUB test set size. Use a smaller number for quick tests if needed.
    # num_test_samples = 100 # For faster placeholder execution
    dummy_images = torch.randn(num_test_samples, 3, 224, 224) # Assuming (B, C, H, W) and 224x224 input
    dummy_labels = torch.randint(0, 200, (num_test_samples,)) # CUB has 200 classes
    dummy_dataset = torch.utils.data.TensorDataset(dummy_images, dummy_labels)
    test_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Initialized placeholder CUB test dataloader with {len(dummy_dataset)} dummy samples.")
    print("YOU MUST REPLACE THIS WITH THE ACTUAL CUB DATALOADER FROM VLG-CBM.")
    # --- END CRITICAL PLACEHOLDER CODE ---
    return test_loader

# !!! END USER ACTION REQUIRED for placeholder functions !!!

def get_all_predictions(model, dataloader, device):
    """Evaluates the model and returns all predictions and true labels."""
    model.eval()
    all_predictions = []
    all_true_labels = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted_classes = torch.max(outputs, 1)
            all_predictions.append(predicted_classes.cpu())
            all_true_labels.append(labels.cpu())
            if (i + 1) % 50 == 0:
                print(f"  Processed batch {i+1}/{len(dataloader)}")
    return torch.cat(all_predictions), torch.cat(all_true_labels)

# --- Main Pruning Experiment Logic ---
def run_pruning_experiment():
    print(f"Starting pruning experiment on device: {DEVICE}")

    # 1. Load Model from checkpoint
    model = load_model_from_checkpoint(CHECKPOINT_PATH, DEVICE)

    # 2. Identify the final weight matrix W_F
    # !!! USER ACTION REQUIRED: Update this to correctly access W_F in your VLG-CBM model !!!
    # This path is HIGHLY DEPENDENT on the specific VLG-CBM model architecture.
    # Example: if W_F is the weight of a layer named 'model.classifier.linear':
    #   final_layer_for_pruning = model.classifier.linear
    # For the PlaceholderVLG_CBM defined above, it's 'model.final_linear_layer'.
    try:
        final_layer_for_pruning = model.final_linear_layer 
        print(f"Identified final layer for pruning: {type(final_layer_for_pruning)}")
    except AttributeError:
        print("Error: Could not access 'model.final_linear_layer'.")
        print("Please modify the script to correctly point to the layer representing W_F in the VLG-CBM model.")
        print("This is typically the linear layer mapping concept activations to class logits.")
        return
    # --- END USER ACTION REQUIRED for W_F path ---

    W_F_original = final_layer_for_pruning.weight.data.clone()
    num_classes, num_concepts = W_F_original.shape
    print(f"Original W_F shape: (num_classes={num_classes}, num_concepts={num_concepts})")

    # 3. Get CUB Test Dataloader
    test_loader = get_cub_test_dataloader(CUB_DATASET_ROOT, BATCH_SIZE)

    # 4. Get Predictions from the Original Model
    print("\nGetting predictions from the original model...")
    original_model_predictions, true_labels = get_all_predictions(model, test_loader, DEVICE)
    original_accuracy = torch.sum(original_model_predictions == true_labels).item() / len(true_labels)
    print(f"Original model accuracy: {original_accuracy*100:.2f}%")

    # 5. Prune W_F: Keep only top-K concepts per class by weight magnitude
    print(f"\nPruning W_F to top-{NUM_TOP_CONCEPTS} concepts per class...")
    W_F_pruned = torch.zeros_like(W_F_original)
    for class_idx in range(num_classes):
        class_concept_weights = W_F_original[class_idx, :] # Weights for this class from all concepts
        abs_concept_weights = torch.abs(class_concept_weights)
        
        # Determine actual k for topk (cannot be more than num_concepts)
        k_to_select = min(NUM_TOP_CONCEPTS, num_concepts)
        
        if k_to_select > 0:
            _, top_k_indices = torch.topk(abs_concept_weights, k_to_select)
            # Copy the original weights for these top-k concepts to the pruned matrix
            W_F_pruned[class_idx, top_k_indices] = W_F_original[class_idx, top_k_indices]
        elif num_concepts > 0: # k_to_select is 0 but concepts exist
            print(f"Warning: For class {class_idx}, NUM_TOP_CONCEPTS is 0 or less. No concepts selected.")
        # If num_concepts is 0, W_F_pruned remains zero for this class, which is correct.

    # 6. Apply Pruned W_F to the model and Get Predictions
    # Temporarily replace the model's weights with the pruned version
    final_layer_for_pruning.weight.data = W_F_pruned
    print("\nGetting predictions from the pruned model...")
    pruned_model_predictions, _ = get_all_predictions(model, test_loader, DEVICE) # True labels are the same
    pruned_accuracy = torch.sum(pruned_model_predictions == true_labels).item() / len(true_labels)
    print(f"Pruned model accuracy: {pruned_accuracy*100:.2f}%")

    # IMPORTANT: Restore original weights to the model instance if it's used elsewhere
    final_layer_for_pruning.weight.data = W_F_original
    print("Original weights restored to the model.")

    # 7. Calculate and Report '% Changed Decisions'
    num_changed_decisions = torch.sum(original_model_predictions != pruned_model_predictions).item()
    total_samples = len(original_model_predictions)
    percent_changed = (num_changed_decisions / total_samples) * 100 if total_samples > 0 else 0

    print("\n--- Pruning Experiment Results ---")
    print(f"Total test samples evaluated: {total_samples}")
    print(f"Number of predictions changed after pruning: {num_changed_decisions}")
    print(f"Percentage of changed decisions: {percent_changed:.2f}% (Target from paper for VLG-CBM: 0.12%)")

if __name__ == "__main__":
    run_pruning_experiment()
