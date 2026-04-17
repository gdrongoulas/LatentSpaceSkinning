# import numpy as np
# import os

# DATA_DIR = "/home/gdrongoulas/Documents/paddingsolution/newsol/blender_export"

# faces = np.load(os.path.join(DATA_DIR, "faces.npy"))
# clusters = np.load(os.path.join(DATA_DIR, "clusters.npy"))

# active_clusters = np.unique(clusters)
# cluster_faces = {cid: [] for cid in active_clusters}

# for i, (a, b, c) in enumerate(faces):
#     ca = clusters[a]
#     cb = clusters[b]
#     cc = clusters[c]

#     if ca == cb == cc:
#         cluster_faces[ca].append(i)

# # Convert lists to arrays
# cluster_faces = {cid: np.array(idxs, dtype=np.int32) for cid, idxs in cluster_faces.items()}

# np.save(os.path.join(DATA_DIR, "cluster_faces.npy"), cluster_faces, allow_pickle=True)

# print("Saved cluster_faces.npy")

import numpy as np
import os

DATA = "/home/gdrongoulas/Documents/paddingsolution/newsol/blender_export"

faces = np.load(os.path.join(DATA, "faces.npy"))
clusters = np.load(os.path.join(DATA, "clusters.npy"))

print("faces max vertex index:", faces.max())
print("clusters length:", len(clusters))