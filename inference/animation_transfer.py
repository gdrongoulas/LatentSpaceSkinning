import os
import numpy as np
import torch

from dataset_enhance import PaddedMeshSequenceDataset
from eead_modelV2_enhanceVolume import VolumetricModel


@torch.no_grad()
def run_animation_transfer(model, bones, rest_pose, device="cuda"):
    """
    bones: (F, B, 3, 4)
    rest_pose: (V, 3)
    returns: (F, V, 3)
    """
    model.eval()

    bones_t = torch.from_numpy(bones).float().to(device).unsqueeze(0)   # (1,F,B,3,4)
    rest_t  = torch.from_numpy(rest_pose).float().to(device).unsqueeze(0)  # (1,V,3)

    # Αν το μοντέλο σου παίρνει και semantic, προσαρμόζεις εδώ:
    # π.χ. sem_t = torch.from_numpy(semantic_vec).float().to(device).unsqueeze(0)
    # και μετά: pred_disp = model(bones_t, rest_t, sem_t)

    pred_disp = model(bones_t, rest_t)   # (1,F,V,3) αν το forward είναι (bones, rest_pose)
    pred_disp = pred_disp.squeeze(0).cpu().numpy()  # (F,V,3)

    rest_pose_expanded = np.broadcast_to(rest_pose[None, :, :], pred_disp.shape)
    pred_mesh = rest_pose_expanded + pred_disp

    return pred_mesh


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # Paths
    # -----------------------------
    root_dir = r"/home/gdrongoulas/Documents/paddingsolution/newsol"

    # ίδιο checkpoint με το controller
    pretrained_path = os.path.join(
        root_dir,
        "best_volumetric_model_displacement_Alpha_enhance.pth"
    )

    # νέος χαρακτήρας (ίδιο topology, sculpted)
    rest_pose_path = os.path.join(
        root_dir,
        "alpha_2/static_data/rest_pose_surface.npy"
    )

    # ένα από τα ίδια animation dirs που χρησιμοποίησες στο training
    character_alpha = "alpha_2"
    anim_dir = os.path.join(root_dir, character_alpha, "Animation001")

    # output
    out_animation_path = os.path.join(
        root_dir,
        "animation_transfer_alpha.npy"
    )

    # -----------------------------
    # Dataset sample για σωστά bones
    # -----------------------------
    dataset = PaddedMeshSequenceDataset([anim_dir])
    sample = dataset[0]

    # Περιμένω ότι το sample έχει:
    #   sample["rest_pose"]      -> (V,3)
    #   sample["bone_matrices"]  -> (F,B,3,4)
    #   (αν έχει και semantic, το προσθέτεις στο run_animation_transfer)

    bone_matrices = sample["bone_matrices"].numpy()   # (F,B,3,4)

    # -----------------------------
    # Load new rest pose (target character)
    # -----------------------------
    rest_pose = np.load(rest_pose_path).astype(np.float32)  # (V,3)

    # -----------------------------
    # Build model with SAME config as controller
    # -----------------------------
    max_vertices = sample["rest_pose"].shape[0]
    max_bones = sample["bone_matrices"].shape[1]
    hidden_size = 80
    num_layers = 2

    model = VolumetricModel(
        max_vertices=max_vertices,
        max_bones=max_bones,
        hidden_size=hidden_size,
        num_layers=num_layers
    ).to(device)

    state = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(state)
    print("Loaded pretrained model:", pretrained_path)

    # -----------------------------
    # Run animation transfer
    # -----------------------------
    pred_mesh_seq = run_animation_transfer(
        model=model,
        bones=bone_matrices,
        rest_pose=rest_pose,
        device=device
    )

    np.save(out_animation_path, pred_mesh_seq)
    print("Saved transferred animation:", pred_mesh_seq.shape)
