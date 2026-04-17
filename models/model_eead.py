import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_function import disper_loss   # monitoring only


# =====================================================================
# 1. Motion Encoder
# =====================================================================
class MotionEncoder(nn.Module):
    def __init__(self, bone_feat_dim, hidden_size, num_layers, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=bone_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.post = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, bone_seq):
        out, _ = self.lstm(bone_seq)
        return self.post(out)   # (B, F, H)


# =====================================================================
# 2. Rest Pose Encoder (per‑vertex)
# =====================================================================
class RestPoseEncoder(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        mid = hidden_size * 2   # 256

        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),

            nn.Linear(hidden_size, mid),
            nn.GELU(),
            nn.LayerNorm(mid),

            nn.Linear(mid, mid),
            nn.GELU(),
            nn.LayerNorm(mid),

            nn.Linear(mid, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, rest_pose):
        # rest_pose: (B, V, 3)
        return self.mlp(rest_pose)   # (B, V, H)


# =====================================================================
# 3. Semantic Encoder (6D → H)
# =====================================================================
class SemanticEncoder(nn.Module):
    def __init__(self, hidden_size=128, semantic_dim=6):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(semantic_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),

            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, semantic_vec):
        # semantic_vec: (B, 6)
        return self.net(semantic_vec)   # (B, H)


# =====================================================================
# 4. Fusion (motion + rest_global + semantic)
# =====================================================================
class Fusion(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 4),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 4),

            nn.Linear(hidden_size * 4, hidden_size * 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size * 2),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, motion_seq, rest_global, semantic_lat):
        """
        motion_seq:   (B, F, H)
        rest_global:  (B, H)
        semantic_lat: (B, H)
        """
        B, F, H = motion_seq.shape

        cond = torch.cat([rest_global, semantic_lat], dim=-1)   # (B, 2H)
        cond = cond.unsqueeze(1).expand(B, F, 2 * H)            # (B, F, 2H)

        x = torch.cat([motion_seq, cond], dim=-1)               # (B, F, 3H)
        fused = self.mlp(x)                                     # (B, F, H)

        return fused + motion_seq   # residual


# =====================================================================
# 5. Vertex Decoder (per‑vertex, dynamic num_vertices)
# =====================================================================
class VertexDecoder(nn.Module):
    def __init__(self, hidden_size=128, num_layers=2, num_vertices=None, embed_dim=16):
        super().__init__()

        assert num_vertices is not None, "num_vertices must be provided dynamically"

        self.hidden_size = hidden_size
        self.num_vertices = num_vertices

        # Temporal LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(hidden_size)

        # Dynamic vertex embedding
        self.vertex_embed = nn.Embedding(self.num_vertices, embed_dim)
        self.register_buffer("vertex_ids", torch.arange(self.num_vertices))

        # Balanced MLP (<2M total params with whole model)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size + hidden_size + embed_dim, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),

            nn.Linear(hidden_size // 2, 3),
        )

    def forward(self, fused_seq, rest_feat):
        """
        fused_seq: (B, F, H)
        rest_feat: (B, V, H)
        """
        B, F, H = fused_seq.shape
        _, V, Hr = rest_feat.shape
        assert V == self.num_vertices, "rest_feat vertex count mismatch"
        assert H == Hr

        # Temporal modeling
        lstm_out, _ = self.lstm(fused_seq)      # (B, F, H)
        lstm_out = self.norm(lstm_out)

        # Flatten frames
        lstm_flat = lstm_out.reshape(B * F, H)  # (B*F, H)
        lstm_expanded = lstm_flat.unsqueeze(1).expand(B * F, V, H)

        # Expand rest features
        rest_expanded = rest_feat.unsqueeze(1).expand(B, F, V, H)
        rest_expanded = rest_expanded.reshape(B * F, V, H)

        # Vertex embeddings
        vert_emb = self.vertex_embed(self.vertex_ids)          # (V, E)
        vert_emb = vert_emb.unsqueeze(0).expand(B * F, V, -1)  # (B*F, V, E)

        # Concatenate all features
        feat = torch.cat([lstm_expanded, rest_expanded, vert_emb], dim=-1)

        # Predict displacement
        disp = self.mlp(feat)                                 # (B*F, V, 3)
        return disp.view(B, F, V, 3)


# =====================================================================
# 6. Full Model + Training Utilities
# =====================================================================
class VolumetricModelV2(nn.Module):
    def __init__(self, max_vertices, max_bones, hidden_size=128, num_layers=2):
        super().__init__()

        bone_feat_dim = max_bones * 12

        self.motion_encoder = MotionEncoder(bone_feat_dim, hidden_size, num_layers)
        self.rest_encoder   = RestPoseEncoder(hidden_size)
        self.semantic_enc   = SemanticEncoder(hidden_size)
        self.fusion         = Fusion(hidden_size)
        self.decoder        = VertexDecoder(hidden_size, num_layers, max_vertices)

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    def forward(self, bone_matrices, rest_pose, semantic_vec):
        """
        bone_matrices: (B, F, Bmax, 4, 4) or (B, F, Bmax, 3, 4) → we use first 12 elems
        rest_pose:     (B, V, 3)
        semantic_vec:  (B, 6)
        """
        B, F, Bmax, _, _ = bone_matrices.shape

        bones_flat = bone_matrices.view(B, F, Bmax, 12)
        bone_seq = bones_flat.view(B, F, Bmax * 12)           # (B, F, Bmax*12)

        motion_lat = self.motion_encoder(bone_seq)            # (B, F, H)
        rest_feat  = self.rest_encoder(rest_pose)             # (B, V, H)
        rest_global = rest_feat.mean(dim=1)                   # (B, H)

        semantic_lat = self.semantic_enc(semantic_vec)        # (B, H)

        fused = self.fusion(motion_lat, rest_global, semantic_lat)  # (B, F, H)

        return self.decoder(fused, rest_feat)                 # (B, F, V, 3)

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------
    def train_one_epoch(self, loader, optimizer, device, loss_fn):
        self.train()
        total_loss = 0.0
        total_disper = 0.0

        for batch in loader:
            bone_matrices = batch["bone_matrices"].to(device)
            rest_pose     = batch["rest_pose"].to(device)
            mesh_vertices = batch["mesh_vertices"].to(device)
            semantic_vec  = batch["semantic_vector"].to(device)

            rest_expanded = rest_pose.unsqueeze(1)                 # (B, 1, V, 3)
            target_disp = mesh_vertices - rest_expanded            # (B, F, V, 3)

            pred_disp = self(bone_matrices, rest_pose, semantic_vec)   # (B, F, V, 3)
            pred_vertices = rest_expanded + pred_disp              # (B, F, V, 3)

            loss = loss_fn(pred_disp, target_disp, batch)
            disper = disper_loss(pred_vertices, mesh_vertices)     # monitoring only

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_disper += disper.item()

        return total_loss / len(loader), total_disper / len(loader)

    # ---------------------------------------------------------
    # Validation step
    # ---------------------------------------------------------
    def validate_one_epoch(self, loader, device, loss_fn):
        self.eval()
        total_loss = 0.0
        total_disper = 0.0

        with torch.no_grad():
            for batch in loader:
                bone_matrices = batch["bone_matrices"].to(device)
                rest_pose     = batch["rest_pose"].to(device)
                mesh_vertices = batch["mesh_vertices"].to(device)
                semantic_vec  = batch["semantic_vector"].to(device)

                rest_expanded = rest_pose.unsqueeze(1)
                target_disp = mesh_vertices - rest_expanded

                pred_disp = self(bone_matrices, rest_pose, semantic_vec)
                pred_vertices = rest_expanded + pred_disp

                loss = loss_fn(pred_disp, target_disp, batch)
                disper = disper_loss(pred_vertices, mesh_vertices)

                total_loss += loss.item()
                total_disper += disper.item()

        return total_loss / len(loader), total_disper / len(loader)

    # ---------------------------------------------------------
    # Full training loop
    # ---------------------------------------------------------
    def train_model(self, train_loader, val_loader, optimizer, num_epochs, device, loss_fn, save_path):
        best_val = float("inf")

        for epoch in range(num_epochs):
            train_loss, train_dis = self.train_one_epoch(train_loader, optimizer, device, loss_fn)

            if val_loader is None:
                print(f"Epoch {epoch+1}/{num_epochs} | Train {train_loss:.4f} | DisPer {train_dis:.4f} | Val: SKIPPED")
                continue

            val_loss, val_dis = self.validate_one_epoch(val_loader, device, loss_fn)

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train {train_loss:.4f} | DisPer {train_dis:.4f} | "
                f"Val {val_loss:.4f} | DisPer {val_dis:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.state_dict(), save_path)
                print(f"  -> Saved best model to {save_path}")
