import torch
import torch.nn as nn

from loss_function import disper_loss


# ---------------------------------------------------------
# Motion encoder (bones sequence)
# ---------------------------------------------------------

class MotionEncoder(nn.Module):
    """
    Input:
        bone_seq: (B, F, D) όπου D = max_bones * 12
    Output:
        motion_seq: (B, F, H)
    """

    def __init__(self, bone_feat_dim: int, hidden_size: int, num_layers: int):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=bone_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, bone_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(bone_seq)          # (B, F, H)
        out = self.norm(out)                  # (B, F, H)
        out = torch.tanh(self.out_proj(out))  # (B, F, H)
        return out


# ---------------------------------------------------------
# Rest pose encoder (surface vertices)
# ---------------------------------------------------------

class RestPoseVertexEncoder(nn.Module):
    """
    Encodes rest pose vertices into a shape embedding.

    Input:
        rest_pose: (B, V, 3)
    Output:
        rest_embed: (B, H)
    """

    def __init__(self, num_vertices: int, hidden_size: int):
        super().__init__()
        in_dim = num_vertices * 3

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, rest_pose: torch.Tensor) -> torch.Tensor:
        B, V, _ = rest_pose.shape
        x = rest_pose.view(B, V * 3)   # (B, 3V)
        return self.net(x)             # (B, H)


# ---------------------------------------------------------
# Semantic encoder (motion category vector)
# ---------------------------------------------------------

class SemanticEncoder(nn.Module):
    """
    Encodes semantic category vector, e.g. [0,0,0,0,1,0]
    CATEGORIES = ["walking", "running", "turnleft", "turnright", "raisehands", "jump"]

    Input:
        sem_vec: (B, S)  (S = num_semantic_dims)
    Output:
        sem_lat: (B, H)
    """

    def __init__(self, num_semantic_dims: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_semantic_dims, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),

            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
        )

    def forward(self, sem_vec: torch.Tensor) -> torch.Tensor:
        return self.net(sem_vec)  # (B, H)


# ---------------------------------------------------------
# Attention / fusion module
# ---------------------------------------------------------

class Attention(nn.Module):
    """
    Fusion module between motion sequence and rest pose embedding.

    Input:
        motion_seq: (B, F, H)
        rest_embed: (B, H)
    Output:
        fused_seq: (B, F, H)
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Tanh(),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, motion_seq: torch.Tensor, rest_embed: torch.Tensor) -> torch.Tensor:
        B, F, H = motion_seq.shape
        rest_expanded = rest_embed.unsqueeze(1).expand(-1, F, -1)   # (B, F, H)
        combined = torch.cat([motion_seq, rest_expanded], dim=-1)   # (B, F, 2H)
        fused = self.fc(combined)                                   # (B, F, H)
        return self.norm(fused + motion_seq)


# ---------------------------------------------------------
# Vertex decoder (per-vertex displacements)
# ---------------------------------------------------------

class VertexDecoder(nn.Module):
    """
    Decodes fused motion+shape features into per-frame vertex displacements.

    Input:
        fused_seq: (B, F, H)
    Output:
        pred_disp: (B, F, V, 3)
    """

    def __init__(self, hidden_size: int, num_layers: int, num_vertices: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vertices = num_vertices

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_vertices * 3)

    def forward(self, fused_seq: torch.Tensor) -> torch.Tensor:
        B, F, H = fused_seq.shape

        out, _ = self.lstm(fused_seq)           # (B, F, H)
        out = self.layer_norm(out)              # (B, F, H)

        out_flat = out.contiguous().view(B * F, H)        # (B*F, H)
        disp_flat = self.fc_out(out_flat)                 # (B*F, V*3)
        disp = disp_flat.view(B, F, self.num_vertices, 3) # (B, F, V, 3)
        return disp


# ---------------------------------------------------------
# Full model: VolumetricModel (with semantic conditioning)
# ---------------------------------------------------------

class VolumetricModel(nn.Module):
    """
    Full model:
        - Motion encoder over bones sequence
        - Rest-pose encoder over vertices
        - Semantic encoder over category vector
        - Fusion (attention-like) between motion & shape
        - Decoder to per-frame vertex displacements

    Forward:
        bone_matrices:   (B, F, B_max, 3, 4)
        rest_pose:       (B, V_max, 3)
        semantic_vector: (B, S)   (S = num_semantic_dims)

    Output:
        pred_disp:       (B, F, V_max, 3)
    """

    def __init__(
        self,
        max_vertices: int,
        max_bones: int,
        hidden_size: int,
        num_layers: int,
        num_semantic_dims: int = 6,
    ):
        super().__init__()
        self.max_vertices = max_vertices
        self.max_bones = max_bones
        self.hidden_size = hidden_size

        bone_feat_dim = max_bones * 12  # 3x4 per bone

        self.motion_encoder = MotionEncoder(
            bone_feat_dim=bone_feat_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        self.rest_encoder = RestPoseVertexEncoder(
            num_vertices=max_vertices,
            hidden_size=hidden_size,
        )
        self.semantic_encoder = SemanticEncoder(
            num_semantic_dims=num_semantic_dims,
            hidden_size=hidden_size,
        )
        self.attention = Attention(feature_dim=hidden_size)
        self.decoder = VertexDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_vertices=max_vertices,
        )

    def forward(
        self,
        bone_matrices: torch.Tensor,    # (B, F, B_max, 3, 4)
        rest_pose: torch.Tensor,        # (B, V_max, 3)
        semantic_vector: torch.Tensor,  # (B, S)
    ) -> torch.Tensor:
        B, F, B_max, _, _ = bone_matrices.shape

        # (B, F, B_max, 3, 4) → (B, F, B_max, 12)
        bones_flat = bone_matrices.view(B, F, B_max, 12)

        # Collapse bones into a per-frame feature vector: (B, F, B_max*12)
        bone_seq = bones_flat.view(B, F, B_max * 12)

        # Motion encoding
        motion_seq = self.motion_encoder(bone_seq)      # (B, F, H)

        # Semantic encoding
        sem_lat = self.semantic_encoder(semantic_vector)   # (B, H)
        sem_lat = sem_lat.unsqueeze(1).expand(-1, F, -1)   # (B, F, H)
        motion_seq = motion_seq + sem_lat                  # semantic-conditioned motion

        # Rest pose encoding (vertices)
        rest_embed = self.rest_encoder(rest_pose)       # (B, H)

        # Fusion / attention-like conditioning
        fused_seq = self.attention(motion_seq, rest_embed)  # (B, F, H)

        # Decode to per-frame vertex displacements
        pred_disp = self.decoder(fused_seq)             # (B, F, V_max, 3)

        return pred_disp

    # -------------------------------
    # Training utilities
    # -------------------------------

    def train_one_epoch(
        self,
        loader,
        optimizer,
        device,
        loss_fn,
    ):
        """
        loss_fn(pred_disp, target_disp, batch) -> scalar loss

        target_disp = mesh_vertices - rest_pose (broadcast)
        batch must contain:
          - "bone_matrices"
          - "rest_pose"
          - "mesh_vertices"
          - "semantic_vector"
        """
        self.train()
        total = 0.0
        totalDisPer = 0.0

        for batch in loader:
            bone_matrices = batch["bone_matrices"].to(device)   # (B, F, B_max, 3, 4)
            rest_pose     = batch["rest_pose"].to(device)       # (B, V_max, 3)
            mesh_vertices = batch["mesh_vertices"].to(device)   # (B, F, V_max, 3)
            semantic_vec  = batch["semantic_vector"].to(device) # (B, S)

            rest_expanded = rest_pose.unsqueeze(1)              # (B, 1, V, 3)
            target_disp   = mesh_vertices - rest_expanded       # (B, F, V, 3)

            pred_disp = self(bone_matrices, rest_pose, semantic_vec)  # (B, F, V, 3)

            main_loss = loss_fn(pred_disp, target_disp, batch)

            if pred_disp.shape[1] > 1:
                smooth_t = ((pred_disp[:, 1:] - pred_disp[:, :-1]) ** 2).mean()
            else:
                smooth_t = torch.zeros((), device=pred_disp.device)

            smooth_v = ((pred_disp - pred_disp.mean(dim=2, keepdim=True)) ** 2).mean()

            loss = main_loss + 0.005 * smooth_t + 0.0005 * smooth_v

            lossDisPer = disper_loss(pred_disp, target_disp)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            totalDisPer += lossDisPer.item()

        return total / len(loader), totalDisPer / len(loader)

    def validate_one_epoch(
        self,
        loader,
        device,
        loss_fn,
    ):
        self.eval()
        total = 0.0
        totalDisPer = 0.0

        with torch.no_grad():
            for batch in loader:
                bone_matrices = batch["bone_matrices"].to(device)
                rest_pose     = batch["rest_pose"].to(device)
                mesh_vertices = batch["mesh_vertices"].to(device)
                semantic_vec  = batch["semantic_vector"].to(device)

                rest_expanded = rest_pose.unsqueeze(1)
                target_disp   = mesh_vertices - rest_expanded

                pred_disp = self(bone_matrices, rest_pose, semantic_vec)

                loss = loss_fn(pred_disp, target_disp, batch)
                lossDisPer = disper_loss(pred_disp, target_disp)
                total += loss.item()
                totalDisPer += lossDisPer.item()

        return total / len(loader), totalDisPer / len(loader)

    def train_model(
        self,
        train_loader,
        val_loader,
        optimizer,
        num_epochs,
        device,
        loss_fn,
        save_path,
    ):
        best_val = float("inf")

        for epoch in range(num_epochs):
            train_loss, train_dis_per = self.train_one_epoch(
                train_loader, optimizer, device, loss_fn
            )

            if val_loader is None:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train: {train_loss:.4f} | Train Dis Per: {train_dis_per:.4f} | Val: SKIPPED "
                )
                continue

            val_loss, val_dis_per = self.validate_one_epoch(
                val_loader, device, loss_fn
            )

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train: {train_loss:.4f} | Train Dis Per: {train_dis_per:.4f} | "
                f"Val: {val_loss:.4f} | Val Dis Per: {val_dis_per:.4f}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.state_dict(), save_path)
                print(f"  -> Saved best model to {save_path}")
