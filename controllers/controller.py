import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataset import MotionDataset
from eead_model import VolumetricModel
from loss_function import make_total_loss


# ---------------------------------------------------------
# Count parameters per module
# ---------------------------------------------------------
def count_parameters(model):
    module_params = {}
    total = 0

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        module_params[name] = params
        total += params

    return module_params, total


# ---------------------------------------------------------
# Find animation paths
# ---------------------------------------------------------
def find_animation_paths(root_dir, animation_names):
    anim_paths = []

    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category, "animations_data")
        if not os.path.isdir(category_path):
            continue

        for anim in animation_names:
            candidate = os.path.join(category_path, anim)
            if os.path.isdir(candidate):
                anim_paths.append(candidate)

    return anim_paths


# ---------------------------------------------------------
# Main training controller
# ---------------------------------------------------------
def main():
    # -----------------------------
    # Paths & config
    # -----------------------------
    root_dir = r"/home/gdrongoulas/Documents/paddingsolution/newsol"
    semantic_json = os.path.join(root_dir, "semantic_labels.json")

    batch_size = 1
    learning_rate = 1e-3 # 1e-3 to eixa
    epochs = 4000
    hidden_size = 40 #40 eixa kala apotelesmata kai 64 douleuei 
    num_layers = 2
    # weight_decay = 1e-5 # 1e-4 to eixa
    betas = (0.9, 0.999)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Select animations manually
    # -----------------------------
    train_anim_names = [
        "RaiseLeftHand", "RaiseRightHand",
        "turnleft", "jump",
        "walk", "walk_inPlace",
    ]


    val_anim_names = ["walk"]

    train_paths = find_animation_paths(root_dir, train_anim_names)
    val_paths   = find_animation_paths(root_dir, val_anim_names)

    print("Train paths:", train_paths)
    print("Val paths:", val_paths)

    # -----------------------------
    # Dataset & DataLoader
    # -----------------------------
    train_dataset = MotionDataset(train_paths, semantic_json, root_dir=root_dir)
    val_dataset   = MotionDataset(val_paths, semantic_json, root_dir=root_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=default_collate
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=default_collate
    )

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # -----------------------------
    # Loss function
    # -----------------------------
    loss_fn = make_total_loss(
        w_vertex=1.0,
        w_smooth=0.1
    )
    print("Loss function created.")

    # -----------------------------
    # Model
    # -----------------------------
    sample = train_dataset[0]
    max_vertices = sample["rest_pose"].shape[0]
    max_bones = sample["bone_matrices"].shape[1]

    model = VolumetricModel(
        max_vertices=max_vertices,
        max_bones=max_bones,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    print("\n==============================")
    print("  MODEL PARAMETER COUNT")
    print("==============================")

    module_params, total_params = count_parameters(model)

    for name, count in module_params.items():
        print(f"{name:25s}: {count:,} parameters")

    print("--------------------------------")
    print(f"Total trainable parameters: {total_params:,}")
    print("================================\n")

    # -----------------------------
    # Optimizer
    # -----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        # weight_decay=weight_decay,
        betas=betas
    )

    # -----------------------------
    # Train
    # -----------------------------
    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        loss_fn=loss_fn,
        save_path="best_volumetric_model_semantic_compression.pth"
    )


if __name__ == "__main__":
    main()
