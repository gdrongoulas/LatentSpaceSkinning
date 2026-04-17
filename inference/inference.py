import os
import numpy as np
import torch

from eead_model import VolumetricModel


# ---------------------------------------------------------
# Load model (MUST match training)
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
# Run inference (bones untouched)
# ---------------------------------------------------------
def run_inference(model, bones, rest_pose, semantic_vec, device="cuda"):
    bones = torch.from_numpy(bones).float().to(device)
    rest_pose = torch.from_numpy(rest_pose).float().to(device)
    semantic_vec = torch.from_numpy(semantic_vec).float().to(device)

    bones = bones.unsqueeze(0)
    rest_pose = rest_pose.unsqueeze(0)
    semantic_vec = semantic_vec.unsqueeze(0)

    with torch.no_grad():
        pred_disp = model(bones, rest_pose, semantic_vec)

    pred_disp = pred_disp.squeeze(0).cpu().numpy()
    pred_mesh = rest_pose.squeeze(0).cpu().numpy() + pred_disp

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
    model_path = os.path.join(root, "best_volumetric_model_semantic_compression_volume?.pth")

    # Load rest pose
    rest_pose = np.load(os.path.join(root, "rest_pose.npy")).astype(np.float32)

    # Load bones
    bones = np.load(
        r"/home/gdrongoulas/Documents/paddingsolution/newsol/walking/animations_data/walk_inPlace/walk_inPlace_Bones.npy"
    ).astype(np.float32)

    hands_bones = np.load(
        r"/home/gdrongoulas/Documents/paddingsolution/newsol/hands/animations_data/RaiseLeftHand/RaiseLeftHand_Bones.npy"
    ).astype(np.float32)
    w = 0.5
    bones = (1-w) * bones + w *hands_bones 

    # If bones are (F, B*12), reshape
    if bones.ndim == 2:
        F, flat = bones.shape
        B = flat // 12
        bones = bones.reshape(F, B, 3, 4)
    else:
        B = bones.shape[1]
    # CATEGORIES = ["walking", "running", "turnleft", "turnright", "raisehands", "jump"]
    semantic_vec = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    max_vertices = rest_pose.shape[0]
    max_bones = B

    model = load_model(
        model_path,
        max_vertices=max_vertices,
        max_bones=max_bones,
        device=device
    )

    pred_mesh = run_inference(model, bones, rest_pose, semantic_vec, device=device)
    save_mesh_npy(pred_mesh, "predicted_mesh.npy")
