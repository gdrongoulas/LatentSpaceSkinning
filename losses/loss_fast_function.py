import torch
import numpy as np


# ---------------------------------------------------------
# 1. Vertex displacement RMS loss
# ---------------------------------------------------------
def vertex_rms_loss(pred_disp, target_disp):
    """
    pred_disp:   (B, F, V, 3)
    target_disp: (B, F, V, 3)
    """
    mse = (pred_disp - target_disp).pow(2).mean()
    return torch.sqrt(mse + 1e-8)


# ---------------------------------------------------------
# 2. Temporal smoothness loss
# ---------------------------------------------------------
def smoothness_loss(pred_disp):
    """
    Penalizes large differences between consecutive frames.
    pred_disp: (B, F, V, 3)
    """
    if pred_disp.shape[1] < 2:
        return torch.tensor(0.0, device=pred_disp.device)

    diff = pred_disp[:, 1:] - pred_disp[:, :-1]
    return diff.pow(2).mean()


# ---------------------------------------------------------
# 3. disPer loss (monitoring only)
# ---------------------------------------------------------
def disper_loss(pred_disp, target_disp):
    return torch.mean(torch.norm(pred_disp - target_disp, dim=-1))


# ---------------------------------------------------------
# 4. FAST volume loss (GPU‑only, no SciPy)
# ---------------------------------------------------------
# def fast_volume_loss(pred_disp, faces, cluster_faces, target_volumes):
#     """
#     pred_disp: (B, F, V, 3)
#     faces: (M, 3) numpy array
#     cluster_faces: dict {cluster_id: face_indices}
#     target_volumes: (num_clusters,)
#     """

#     device = pred_disp.device

#     # B=1 → παίρνουμε το πρώτο batch
#     pred = pred_disp[0]   # (F, V, 3)

#     # Χρησιμοποιούμε το πρώτο frame για volume
#     verts = pred[0]       # (V, 3)
#     verts = verts.to(device)

#     pred_volumes = []

#     for cid, face_ids in cluster_faces.items():
#         if len(face_ids) == 0:
#             pred_volumes.append(torch.tensor(0.0, device=device))
#             continue

#         # Load faces for this cluster
#         f = torch.tensor(faces[face_ids], dtype=torch.long, device=device)  # (K, 3)

#         v0 = verts[f[:, 0]]
#         v1 = verts[f[:, 1]]
#         v2 = verts[f[:, 2]]

#         # Signed tetra volume: |dot(v0, cross(v1, v2))| / 6
#         vol = torch.sum(torch.abs(torch.einsum("ij,ij->i", v0, torch.cross(v1, v2)))) / 6.0

#         pred_volumes.append(vol)

#     pred_volumes = torch.stack(pred_volumes)
#     target_volumes = torch.tensor(target_volumes, dtype=torch.float32, device=device)

#     return torch.mean((pred_volumes - target_volumes)**2)


def fast_volume_loss(pred_disp, faces, cluster_faces, target_volumes):
    """
    pred_disp: (B, F, V, 3)
    faces: (M, 3) numpy array
    cluster_faces: dict {cluster_id: face_indices}
    target_volumes: (num_clusters,)
    """

    device = pred_disp.device

    # B=1 → παίρνουμε το πρώτο batch
    pred = pred_disp[0]   # (F, V, 3)
    F = pred.shape[0]

    # Frame 0 και frame mid
    frame_ids = [0, F // 2]

    pred_volumes_all = []

    for frame_id in frame_ids:
        verts = pred[frame_id]  # (V, 3)
        verts = verts.to(device)

        pred_volumes = []

        for cid, face_ids in cluster_faces.items():
            if len(face_ids) == 0:
                pred_volumes.append(torch.tensor(0.0, device=device))
                continue

            f = torch.tensor(faces[face_ids], dtype=torch.long, device=device)

            v0 = verts[f[:, 0]]
            v1 = verts[f[:, 1]]
            v2 = verts[f[:, 2]]

            vol = torch.sum(torch.abs(torch.einsum("ij,ij->i", v0, torch.cross(v1, v2)))) / 6.0
            pred_volumes.append(vol)

        pred_volumes = torch.stack(pred_volumes)
        pred_volumes_all.append(pred_volumes)

    # Μέσος όρος volume loss από frame0 + frameMid
    pred_volumes_all = torch.stack(pred_volumes_all).mean(dim=0)

    target_volumes = torch.tensor(target_volumes, dtype=torch.float32, device=device)

    return torch.mean((pred_volumes_all - target_volumes)**2)



# ---------------------------------------------------------
# 5. Total loss (RMS + smooth + fast volume)
# ---------------------------------------------------------
def make_total_loss(
    w_vertex=1.0,
    w_smooth=0.1,
    w_volume=0.0,
    faces=None,
    cluster_faces=None,
    target_volumes=None
):
    def loss_fn(pred_disp, target_disp, batch):
        # 1) Vertex RMS loss
        L_vertex = vertex_rms_loss(pred_disp, target_disp)

        # 2) Temporal smoothness
        L_smooth = smoothness_loss(pred_disp)

        loss = w_vertex * L_vertex + w_smooth * L_smooth

        # 3) FAST volume preservation loss
        if w_volume > 0:
            L_vol = fast_volume_loss(pred_disp, faces, cluster_faces, target_volumes)
            loss = loss + w_volume * L_vol

        return loss

    return loss_fn
