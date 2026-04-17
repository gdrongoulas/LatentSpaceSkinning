import os
import numpy as np
import torch

from eead_model import VolumetricModel


# ---------------------------------------------------------
# Blend multiple bone sequences safely
# ---------------------------------------------------------
def blend_bones(bones_list, weights):
    """
    bones_list: list of np arrays, each (F, B, 3, 4)
    weights: list of floats that sum to 1
    """
    assert len(bones_list) == len(weights)

    # Align frame counts
    minF = min(b.shape[0] for b in bones_list)
    bones_list = [b[:minF] for b in bones_list]

    bones_blend = np.zeros_like(bones_list[0], dtype=np.float32)
    for w, b in zip(weights, bones_list):
        bones_blend += w * b

    return bones_blend


# ---------------------------------------------------------
# Temporal smoothing (safe)
# ---------------------------------------------------------
def smooth_mesh_sequence(mesh_seq, alpha=0.7):
    smoothed = mesh_seq.copy()
    for t in range(1, mesh_seq.shape[0]):
        if np.isnan(smoothed[t-1]).any():
            continue
        smoothed[t] = alpha * smoothed[t-1] + (1 - alpha) * mesh_seq[t]
    return smoothed


# ---------------------------------------------------------
# Load model (FP32 for stability)
# ---------------------------------------------------------
def load_model(model_path, max_vertices, max_bones, device="cuda"):
    hidden_size = 40

    model = VolumetricModel(
        max_vertices=max_vertices,
        max_bones=max_bones,
        hidden_size=hidden_size,
        num_layers=2
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("Model loaded:", model_path)
    return model


# ---------------------------------------------------------
# Run inference (stable FP32)
# ---------------------------------------------------------
def run_inference(
    model,
    bones,
    rest_pose,
    semantic_vec,
    device="cuda",
    smooth_alpha=None
):
    bones_t = torch.from_numpy(bones).float().to(device).unsqueeze(0)
    rest_t  = torch.from_numpy(rest_pose).float().to(device).unsqueeze(0)
    sem_t   = torch.from_numpy(semantic_vec).float().to(device).unsqueeze(0)

    with torch.no_grad():
        pred_disp = model(bones_t, rest_t, sem_t)  # (1,F,V,3)

    pred_disp = pred_disp.squeeze(0).cpu().numpy()  # (F,V,3)

    # Safe mesh reconstruction
    rest_pose_expanded = np.broadcast_to(rest_pose[None, :, :], pred_disp.shape)
    pred_mesh = rest_pose_expanded + pred_disp

    # Optional smoothing
    if smooth_alpha is not None:
        pred_mesh = smooth_mesh_sequence(pred_mesh, alpha=smooth_alpha)

    return pred_mesh


# ---------------------------------------------------------
# Save mesh
# ---------------------------------------------------------
def save_mesh_npy(mesh, out_path):
    np.save(out_path, mesh)
    print("Saved mesh:", out_path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    root = r"/home/gdrongoulas/Documents/paddingsolution/newsol"
    model_path = os.path.join(root, "best_volumetric_model_semantic_compression.pth")

    # Load rest pose
    rest_pose = np.load(os.path.join(root, "rest_pose.npy")).astype(np.float32)

    # Load motions
    bones_walk = np.load(
        r"/home/gdrongoulas/Documents/paddingsolution/newsol/walking/animations_data/walk/walk_Bones.npy"
    ).astype(np.float32)

    bones_run = np.load(
        r"/home/gdrongoulas/Documents/paddingsolution/newsol/running/animations_data/running_turnright_180/running_turnright_180_Bones.npy"
    ).astype(np.float32)

    # Reshape if needed
    def reshape_bones(b):
        if b.ndim == 2:
            F, flat = b.shape
            B = flat // 12
            return b.reshape(F, B, 3, 4)
        return b

    bones_walk = reshape_bones(bones_walk)
    bones_run  = reshape_bones(bones_run)

    # Blend motions (70% walk, 30% raise hand)
    bones_mix = blend_bones(
        [bones_walk, bones_run],
        [0.2, 0.8]
    )

    # Semantic vector
    semantic_vec = np.array([1, 1, 0, 0, 0, 0], dtype=np.float32)

    max_vertices = rest_pose.shape[0]
    max_bones = bones_mix.shape[1]

    model = load_model(model_path, max_vertices, max_bones, device=device)

    pred_mesh = run_inference(
        model,
        bones_mix,
        rest_pose,
        semantic_vec,
        device=device,
        smooth_alpha=0.7
    )

    save_mesh_npy(pred_mesh, "predicted_mesh.npy")
