import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from dataset_enhance import PaddedMeshSequenceDataset
from eead_modelV2_enhanceVolume import VolumetricModel
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

def padded_collate_fn(batch):
    out = {}
    for key in batch[0]:
        out[key] = default_collate([b[key] for b in batch])
    return out

# ---------------------------------------------------------
# Fine‑tuning controller
# ---------------------------------------------------------
def main():
    # -----------------------------
    # Paths & config
    # -----------------------------
    root_dir = r"/home/gdrongoulas/Documents/paddingsolution/newsol"
    # semantic_json = os.path.join(root_dir, "semantic_labels.json")

    batch_size = 1
    learning_rate = 5e-4      # μικρό LR για fine‑tuning
    epochs = 3000              # 100–300 είναι ιδανικά
    hidden_size = 80
    num_layers = 2
    betas = (0.9, 0.999)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # -----------------------------
    # Select animations
    # -----------------------------
    character_alpha = "alpha_2"
    train_dirs_alpha = [
    os.path.join(root_dir, character_alpha, f"Animation{str(i).zfill(3)}")
    for i in list(range(1,5)) 
    ]


    val_dir_alpha = [
        os.path.join(root_dir, character_alpha, "Animation001")
    ]

    train_dataset = PaddedMeshSequenceDataset(train_dirs_alpha)
    val_dataset = PaddedMeshSequenceDataset(val_dir_alpha)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=padded_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=padded_collate_fn
)

    print("Train dataset size:", len(train_dataset))
    print("Val dataset size:", len(val_dataset))

    # -----------------------------
    # Load volume data
    # -----------------------------
    volume_dir = os.path.join(root_dir, "blender_export")
    clusters = np.load(os.path.join(volume_dir, "clusters.npy"))
    target_volumes = np.load(os.path.join(volume_dir, "cluster_volumes_rest.npy"))


    # for fast loss
    faces = np.load(os.path.join(volume_dir, "faces.npy"))
    cluster_faces = np.load(os.path.join(volume_dir, "cluster_faces.npy"), allow_pickle=True).item()
    target_volumes = np.load(os.path.join(volume_dir, "cluster_volumes_rest.npy"))


    print("Loaded clusters:", clusters.shape)
    print("Loaded target volumes:", target_volumes.shape)


    adjacency_matrix = np.load(os.path.join(volume_dir, "adjacency_matrix.npy"))

    loss_fn = make_total_loss(
        w_vertex=1.0,
        w_smooth=0.1,
        w_curvature=0.0,
        w_volume=0.0,
        faces=faces,
        cluster_faces=cluster_faces,
        target_volumes=target_volumes,
        adjacency=adjacency_matrix
    )

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

    # Load pretrained model
    pretrained_path = os.path.join(root_dir, "best_volumetric_model_displacement_Alpha_enhance.pth")
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    print("Loaded pretrained model for fine‑tuning.")

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
    # Fine‑tuning
    # -----------------------------
    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=epochs,
        device=device,
        loss_fn=loss_fn,
        save_path="best_volumetric_model_displacement_Alpha_enhance.pth"
    )


if __name__ == "__main__":
    main()
