import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PaddedMeshSequenceDataset(Dataset):
    """
    Dataset laods:
      - rest_pose_surface_padded.npy
      - mesh_vertices_padded.npy
      - bone_matrices_padded.npy

    returns
      {
        "bone_matrices": (F, B, 3, 4)
        "rest_pose": (V, 3)
        "mesh_vertices": (F, V, 3)
      }
    """

    def __init__(self, anim_dirs):
        self.samples = []

        for d in anim_dirs:
            if not os.path.isdir(d):
                continue

            files = {
                "rest_surface": os.path.join(d, "rest_pose_surface_padded.npy"),
                "mesh": os.path.join(d, "mesh_vertices_padded.npy"),
                "bones": os.path.join(d, "bone_matrices_padded.npy"),
            }

            # Check all files exist
            if all(os.path.exists(f) for f in files.values()):
                self.samples.append(files)
            else:
                print("Skipping incomplete animation folder:", d)

        print(f"Loaded {len(self.samples)} animation sequences.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]

        # Load numpy arrays
        rest_surface = np.load(paths["rest_surface"]).astype(np.float32)  # (V, 3)
        mesh         = np.load(paths["mesh"]).astype(np.float32)          # (F, V, 3)
        bones        = np.load(paths["bones"]).astype(np.float32)         # (F, B, 3, 4)

        # Convert to torch
        return {
            "bone_matrices": torch.from_numpy(bones),
            "rest_pose": torch.from_numpy(rest_surface),
            "mesh_vertices": torch.from_numpy(mesh),
        }
