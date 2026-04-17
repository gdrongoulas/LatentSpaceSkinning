import torch
import numpy as np
from eead_modelV2_enhanceVolume import VolumetricModel


# ---------------------------------------------------------
# Load model for inference (with cuDNN fix)
# ---------------------------------------------------------
def load_model(model_path, max_vertices, max_bones, hidden_size=80, num_layers=2, device="cuda"):

    model = VolumetricModel(
        max_vertices=max_vertices,
        max_bones=max_bones,
        hidden_size=hidden_size,
        num_layers=num_layers
    )

    # Load weights BEFORE moving to device
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    # Disable cuDNN flattening (fixes CUDNN_STATUS_BAD_PARAM)
    for m in model.modules():
        if isinstance(m, torch.nn.LSTM):
            m.flatten_parameters = lambda *args, **kwargs: None

    model = model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------
# Prepare a single animation sample
# ---------------------------------------------------------
def prepare_sample(bone_matrices_np, rest_pose_np, device="cuda"):

    F, D = bone_matrices_np.shape
    num_bones = D // 12

    bone_matrices = torch.tensor(bone_matrices_np, dtype=torch.float32, device=device)
    rest_pose     = torch.tensor(rest_pose_np,     dtype=torch.float32, device=device)

    bone_matrices = bone_matrices.unsqueeze(0)   # (1, F, D)
    rest_pose     = rest_pose.unsqueeze(0)       # (1, V, 3)

    return bone_matrices, rest_pose, num_bones


# ---------------------------------------------------------
# Run inference
# ---------------------------------------------------------
def run_inference(model, bone_matrices, rest_pose):

    with torch.no_grad():
        pred_disp = model(bone_matrices, rest_pose)          # (1, F, V, 3)
        pred_vertices = pred_disp + rest_pose.unsqueeze(1)   # add rest pose

    return pred_vertices.squeeze(0).cpu().numpy()            # (F, V, 3)


# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load correct files
    bone_matrices_np = np.load(
        "/home/gdrongoulas/Documents/paddingsolution/newsol/saildfish/Animation001/bone_matrices_padded.npy"
    )   # (F, D)

    rest_pose_np = np.load(
        "/home/gdrongoulas/Documents/paddingsolution/newsol/saildfish/Animation001/rest_pose_surface_padded.npy"
    )   # (V, 3)

    # Prepare tensors
    bone_matrices, rest_pose, num_bones = prepare_sample(
        bone_matrices_np, rest_pose_np, device=device
    )

    max_vertices = rest_pose_np.shape[0]
    max_bones    = num_bones

    model_path = "best_volumetric_model_displacement_sailfish.pth"

    # Load model
    model = load_model(
        model_path=model_path,
        max_vertices=max_vertices,
        max_bones=max_bones,
        hidden_size=80,
        num_layers=2,
        device=device
    )

    # Run inference
    pred_vertices = run_inference(model, bone_matrices, rest_pose)

    print("Predicted vertices shape:", pred_vertices.shape)
    np.save("predicted_vertices_saildfish.npy", pred_vertices)
    print("Saved predicted vertices.")
