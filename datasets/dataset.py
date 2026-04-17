import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    """
    Loads:
        <anim>_Bones.npy      -> (F, B, 12) or (F, B, 3, 4)
        <anim>_FullMesh.npy   -> (F, V, 3)
        rest_pose.npy         -> (V, 3)  GLOBAL T-POSE
        semantic_labels.json  -> {anim_name: [6D semantic vector]}

    Returns:
        bone_matrices: (F, B, 3, 4)
        rest_pose:     (V, 3)
        mesh_vertices: (F, V, 3)
        semantic_vector: (6,)
    """

    def __init__(self, anim_dirs, semantic_json, root_dir):
        self.samples = []

        # Load global rest pose
        rest_path = os.path.join(root_dir, "rest_pose.npy")
        if not os.path.exists(rest_path):
            raise FileNotFoundError(f"Missing global rest pose file: {rest_path}")

        self.global_rest_pose = np.load(rest_path).astype(np.float32)  # (V, 3)

        # Load semantic labels
        with open(semantic_json, "r") as f:
            self.semantic_labels = json.load(f)

        # Scan animation folders
        for d in anim_dirs:
            if not os.path.isdir(d):
                continue

            anim_name = os.path.basename(d)

            bones_path = os.path.join(d, f"{anim_name}_Bones.npy")
            mesh_path  = os.path.join(d, f"{anim_name}_FullMesh.npy")

            if not (os.path.exists(bones_path) and os.path.exists(mesh_path)):
                print("Skipping incomplete animation:", d)
                continue

            if anim_name not in self.semantic_labels:
                print("Missing semantic label for:", anim_name)
                continue

            self.samples.append({
                "bones": bones_path,
                "mesh": mesh_path,
                "semantic": self.semantic_labels[anim_name],
                "name": anim_name
            })

        print(f"Loaded {len(self.samples)} animations.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load numpy arrays
        bones = np.load(s["bones"]).astype(np.float32)   # (F, B, 12) or (F, B, 3, 4)
        if bones.ndim == 2:
            F, flat = bones.shape
            B = flat // 12
            bones = bones.reshape(F, B, 3, 4)
        mesh  = np.load(s["mesh"]).astype(np.float32)    # (F, V, 3)
        semantic = np.array(s["semantic"], dtype=np.float32)

        # Normalize bone shape to (F, B, 3, 4)
        if bones.ndim == 3 and bones.shape[-1] == 12:
            F, B, _ = bones.shape
            bones = bones.reshape(F, B, 3, 4)

        elif bones.ndim == 4 and bones.shape[-2:] == (4, 4):
            bones = bones[:, :, :3, :]   # drop last row

        # Use global rest pose
        rest_pose = self.global_rest_pose  # (V, 3)
        # print("Loaded bones shape:", bones.shape)
        # print("Loaded mesh shape:", mesh.shape)
        # print("Loaded rest pose shape:", self.global_rest_pose.shape)
        return {
            "bone_matrices": torch.from_numpy(bones),     # (F, B, 3, 4)
            "rest_pose": torch.from_numpy(rest_pose),     # (V, 3)
            "mesh_vertices": torch.from_numpy(mesh),      # (F, V, 3)
            "semantic_vector": torch.from_numpy(semantic),
            "name": s["name"]
        }
