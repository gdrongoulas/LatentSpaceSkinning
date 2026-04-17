import numpy as np
import os

def print_array_info(name, arr):
    print(f"\n{name}:")
    print(f"  type: {type(arr)}")
    print(f"  dtype: {arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")
    print(f"  shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
    print(f"  sample values:\n{arr[:10] if hasattr(arr, '__getitem__') else arr}")

def main():
    root_dir = "/home/gdrongoulas/Documents/paddingsolution/newsol"
    volume_dir = os.path.join(root_dir, "blender_export")

    # -----------------------------
    # Load clusters.npy
    # -----------------------------
    clusters_path = os.path.join(volume_dir, "clusters.npy")
    clusters = np.load(clusters_path)
    print_array_info("clusters.npy", clusters)

    # -----------------------------
    # Load cluster_faces.npy
    # -----------------------------
    cluster_faces_path = os.path.join(volume_dir, "cluster_faces.npy")
    cluster_faces = np.load(cluster_faces_path, allow_pickle=True).item()

    print("\ncluster_faces.npy:")
    print(f"  type: {type(cluster_faces)}")
    print(f"  number of clusters: {len(cluster_faces)}")

    # print first 5 clusters
    for cid in list(cluster_faces.keys())[:5]:
        faces = cluster_faces[cid]
        print(f"  cluster {cid}: {len(faces)} faces → sample: {faces[:10]}")

    # -----------------------------
    # Load faces.npy
    # -----------------------------
    faces_path = os.path.join(volume_dir, "faces.npy")
    faces = np.load(faces_path)
    print_array_info("faces.npy", faces)

    # -----------------------------
    # Load adjacency_matrix.npy
    # -----------------------------
    adjacency_path = os.path.join(volume_dir, "adjacency_matrix.npy")
    adjacency = np.load(adjacency_path)
    print_array_info("adjacency_matrix.npy", adjacency)

    print("\nDone.")

if __name__ == "__main__":
    main()
