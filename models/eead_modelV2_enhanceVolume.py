import torch
import torch.nn as nn
from loss_function import disper_loss, metric_erms, metric_maxavg



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
        # bone_seq: (B, F, D)
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
        # rest_pose: (B, V, 3)
        B, V, _ = rest_pose.shape
        x = rest_pose.view(B, V * 3)   # (B, 3V)
        return self.net(x)             # (B, H)


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
        # residual + norm για σταθερότητα και καλύτερη γενίκευση
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
        # fused_seq: (B, F, H)
        B, F, H = fused_seq.shape

        out, _ = self.lstm(fused_seq)           # (B, F, H)
        out = self.layer_norm(out)              # (B, F, H)

        out_flat = out.contiguous().view(B * F, H)        # (B*F, H)
        disp_flat = self.fc_out(out_flat)                 # (B*F, V*3)
        disp = disp_flat.view(B, F, self.num_vertices, 3) # (B, F, V, 3)
        return disp


# ---------------------------------------------------------
# Full model: VolumetricModel (displacement-based)
# ---------------------------------------------------------

class VolumetricModel(nn.Module):
    """
    Full model:
        - Motion encoder over bones sequence
        - Rest-pose encoder over vertices
        - Fusion (attention-like) between motion & shape
        - Decoder to per-frame vertex displacements

    Forward:
        bone_matrices:   (B, F, B_max, 3, 4)
        rest_pose:       (B, V_max, 3)

    Output:
        pred_disp:       (B, F, V_max, 3)

    Training:
        target_disp = mesh_vertices - rest_pose (broadcast)
        loss_fn(pred_disp, target_disp, batch)
    """

    def __init__(
        self,
        max_vertices: int,
        max_bones: int,
        hidden_size: int,
        num_layers: int,
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
        self.attention = Attention(feature_dim=hidden_size)
        self.decoder = VertexDecoder(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_vertices=max_vertices,
        )

    def forward(
        self,
        bone_matrices: torch.Tensor,   # (B, F, B_max, 3, 4) OR (B, F, B_max*12)
        rest_pose: torch.Tensor,       # (B, V_max, 3)
    ) -> torch.Tensor:

        # Case 1: input is already (B, F, B_max, 3, 4)
        if bone_matrices.dim() == 5:
            B, F, B_max, _, _ = bone_matrices.shape

        # Case 2: input is (B, F, B_max*12)
        elif bone_matrices.dim() == 3:
            B, F, flat = bone_matrices.shape
            B_max = flat // 12
            bone_matrices = bone_matrices.reshape(B, F, B_max, 3, 4)

        else:
            raise ValueError(f"Unexpected bone_matrices shape: {bone_matrices.shape}")

        # (B, F, B_max, 3, 4) → (B, F, B_max, 12)
        bones_flat = bone_matrices.reshape(B, F, B_max, 12)

        # Collapse bones into per-frame feature vector: (B, F, B_max*12)
        bone_seq = bones_flat.reshape(B, F, B_max * 12)

        # Motion encoding
        motion_seq = self.motion_encoder(bone_seq)      # (B, F, H)

        # Rest pose encoding
        rest_embed = self.rest_encoder(rest_pose)       # (B, H)

        # Fusion
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

        Εδώ:
          - target_disp = mesh_vertices - rest_pose (broadcast)
          - batch περιέχει ό,τι χρειάζεται η loss_fn (αν το χρησιμοποιεί)
        """
        self.train()
        total = 0.0
        totalDisPer = 0.0
        totalERMS = 0.0          ### NEW
        totalMaxAvg = 0.0        ### NEW

        for batch in loader:
            bone_matrices = batch["bone_matrices"].to(device)   # (B, F, B_max, 3, 4)
            rest_pose = batch["rest_pose"].to(device)           # (B, V_max, 3)
            mesh_vertices = batch["mesh_vertices"].to(device)   # (B, F, V_max, 3)

            # Compute target displacements: mesh - rest
            rest_expanded = rest_pose.unsqueeze(1)              # (B, 1, V, 3)
            target_disp = mesh_vertices - rest_expanded         # (B, F, V, 3)

            pred_disp = self(bone_matrices, rest_pose)          # (B, F, V, 3)

            # main loss
            main_loss = loss_fn(pred_disp, target_disp, batch)

            # temporal smoothness: frame-to-frame consistency
            if pred_disp.shape[1] > 1:
                smooth_t = ((pred_disp[:, 1:] - pred_disp[:, :-1]) ** 2).mean()
            else:
                smooth_t = torch.zeros((), device=pred_disp.device)

            # spatial smoothness: per-vertex noise reduction
            smooth_v = ((pred_disp - pred_disp.mean(dim=2, keepdim=True)) ** 2).mean()

            # combined loss (ήπια weights για να μην χαλάει το fit)
            loss = main_loss + 0.005 * smooth_t + 0.0005 * smooth_v

            lossDisPer = disper_loss(pred_disp, target_disp)
            lossERMS = metric_erms(pred_disp, target_disp)          ### NEW
            lossMaxAvg = metric_maxavg(pred_disp, target_disp)      ### NEW
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total += loss.item()
            totalDisPer += lossDisPer.item()
            totalERMS += lossERMS
            totalMaxAvg += lossMaxAvg

        return (
            total / len(loader),
            totalDisPer / len(loader),
            totalERMS / len(loader),        ### NEW
            totalMaxAvg / len(loader)       ### NEW
        )


    def validate_one_epoch(
        self,
        loader,
        device,
        loss_fn,
    ):
        self.eval()
        total = 0.0
        totalDisPer = 0.0
        totalERMS = 0.0          ### NEW
        totalMaxAvg = 0.0        ### NEW

        with torch.no_grad():
            for batch in loader:
                bone_matrices = batch["bone_matrices"].to(device)
                rest_pose = batch["rest_pose"].to(device)
                mesh_vertices = batch["mesh_vertices"].to(device)

                rest_expanded = rest_pose.unsqueeze(1)
                target_disp = mesh_vertices - rest_expanded

                pred_disp = self(bone_matrices, rest_pose)

                loss = loss_fn(pred_disp, target_disp, batch)
                lossDisPer = disper_loss(pred_disp, target_disp)
                lossERMS = metric_erms(pred_disp, target_disp)          ### NEW
                lossMaxAvg = metric_maxavg(pred_disp, target_disp)      ### NEW

                total += loss.item()
                totalDisPer += lossDisPer.item()
                totalERMS += lossERMS
                totalMaxAvg += lossMaxAvg

        return (
            total / len(loader),
            totalDisPer / len(loader),
            totalERMS / len(loader),        ### NEW
            totalMaxAvg / len(loader)       ### NEW
        )


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
            train_loss, train_dis_per, train_erms, train_maxavg = self.train_one_epoch(   ### NEW
                train_loader, optimizer, device, loss_fn
            )

            if val_loader is None:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train: {train_loss:.4f} | "
                    f"Train Dis Per: {train_dis_per:.4f} | "
                    f"Train ERMS: {train_erms:.4f} | "          ### NEW
                    f"Train MaxAvg: {train_maxavg:.4f} | "      ### NEW
                    f"Val: SKIPPED "
                )
                continue

            val_loss, val_dis_per, val_erms, val_maxavg = self.validate_one_epoch(       ### NEW
                val_loader, device, loss_fn
            )

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train: {train_loss:.4f} | "
                f"Train Dis Per: {train_dis_per:.4f} | "
                f"Train ERMS: {train_erms:.4f} | "              ### NEW
                f"Train MaxAvg: {train_maxavg:.4f} | "          ### NEW
                f"Val: {val_loss:.4f} | "
                f"Val Dis Per: {val_dis_per:.4f} | "
                f"Val ERMS: {val_erms:.4f} | "                  ### NEW
                f"Val MaxAvg: {val_maxavg:.4f}"                 ### NEW
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.state_dict(), save_path)
                print(f"  -> Saved best model to {save_path}")

