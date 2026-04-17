import torch
import numpy as np


# ---------------------------------------------------------
# 1. Vertex displacement RMS loss
# ---------------------------------------------------------
def vertex_rms_loss(pred_disp, target_disp):
    mse = (pred_disp - target_disp).pow(2).mean()
    return torch.sqrt(mse + 1e-8)


# ---------------------------------------------------------
# 2. Temporal smoothness loss
# ---------------------------------------------------------
def smoothness_loss(pred_disp):
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
# 4. FAST Curvature (Laplacian) loss — GPU vectorized
def fast_curvature_loss(pred_disp, target_disp, adj_mat, num_frames=2):
    device = pred_disp.device

    adj = torch.tensor(adj_mat, dtype=torch.long, device=device)
    mask = (adj != -1).float()

    adj_safe = adj.clone()
    adj_safe[adj_safe < 0] = 0

    pred = pred_disp[0]
    target = target_disp[0]

    F, V, _ = pred.shape

    if F <= num_frames:
        frame_ids = torch.arange(F, device=device)
    else:
        frame_ids = torch.linspace(0, F-1, steps=num_frames).long().to(device)

    total = 0.0

    for f in frame_ids:
        pf = pred[f]
        tf = target[f]

        pf_n = pf[adj_safe]
        tf_n = tf[adj_safe]

        mask3 = mask.unsqueeze(-1)
        pf_n = pf_n * mask3
        tf_n = tf_n * mask3

        denom = mask3.sum(dim=1) + 1e-8
        pf_mean = pf_n.sum(dim=1) / denom
        tf_mean = tf_n.sum(dim=1) / denom

        pf_lap = pf - pf_mean
        tf_lap = tf - tf_mean

        total += ((pf_lap - tf_lap)**2).mean()

    return total / len(frame_ids)


# ---------------------------------------------------------
# 5. FAST volume loss (ONLY frame 0)
# ---------------------------------------------------------
def fast_volume_loss(pred_disp, faces, cluster_faces, target_volumes):
    device = pred_disp.device

    pred = pred_disp[0]
    verts = pred[0].to(device)

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
    target_volumes = torch.tensor(target_volumes, dtype=torch.float32, device=device)

    return torch.mean((pred_volumes - target_volumes)**2)


# ---------------------------------------------------------
# 6. Total loss (RMS + smooth + curvature + volume)
# ---------------------------------------------------------
def make_total_loss(
    w_vertex=1.0,
    w_smooth=0.1,
    w_curvature=0.0,
    w_volume=0.0,
    faces=None,
    cluster_faces=None,
    target_volumes=None,
    adjacency=None
):

    # Hand clusters:
    # - 1,7,58,63 exist in clusters.npy
    # - ONLY 1,7 exist in target_volumes (size 52)
    hand_clusters_all = [1, 7, 58, 63]     # for curvature/smoothness/vertex
    hand_clusters_volume = [1, 7]          # for volume loss only

    # Load clusters once
    clusters_np = np.load("/home/gdrongoulas/Documents/paddingsolution/newsol/blender_export/clusters.npy")

    def loss_fn(pred_disp, target_disp, batch):

        device = pred_disp.device

        # Build hand mask (per vertex)
        clusters_t = torch.tensor(clusters_np, device=device)
        hand_mask = torch.isin(clusters_t, torch.tensor(hand_clusters_all, device=device)).float()

        # 1. Vertex RMS
        L_vertex = vertex_rms_loss(pred_disp, target_disp)

        # 2. Smoothness
        L_smooth = smoothness_loss(pred_disp)

        loss = w_vertex * L_vertex + w_smooth * L_smooth

        # 3. Curvature (hand-weighted)
        if w_curvature > 0:
            L_curv_raw = fast_curvature_loss(pred_disp, target_disp, adjacency)
            L_curv = (1 + 2 * hand_mask.mean()) * L_curv_raw
            loss += w_curvature * L_curv

        # 4. Volume (hand-weighted)
        if w_volume > 0:

            # Hand clusters (only 1 and 7 exist in target_volumes)
            hand_cf = {cid: cluster_faces[cid] for cid in hand_clusters_volume}
            hand_tv = target_volumes[hand_clusters_volume]

            # Body clusters
            body_cf = {cid: cluster_faces[cid] for cid in cluster_faces if cid not in hand_clusters_volume}
            body_tv = np.delete(target_volumes, hand_clusters_volume)

            L_vol_hand = fast_volume_loss(pred_disp, faces, hand_cf, hand_tv)
            L_vol_body = fast_volume_loss(pred_disp, faces, body_cf, body_tv)

            L_vol = 3.0 * L_vol_hand + 1.0 * L_vol_body
            loss += w_volume * L_vol

        return loss

    return loss_fn


def metric_maxavg(pred_disp, target_disp):
    diff = torch.norm(pred_disp - target_disp, dim=-1)  # (B,F,V)
    max_per_frame = diff.max(dim=-1).values             # (B,F)
    return max_per_frame.mean().item()



def metric_erms(pred_disp, target_disp):
    diff = pred_disp - target_disp
    B, F, V, _ = diff.shape

    frob = torch.norm(diff.reshape(B, -1), dim=1)
    denom = torch.sqrt(torch.tensor(3 * F * V, device=diff.device, dtype=diff.dtype))

    erms = 100.0 * (frob / denom)
    return erms.mean().item()