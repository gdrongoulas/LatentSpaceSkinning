import numpy as np
import os
from collections import defaultdict

DATA_DIR = "/home/gdrongoulas/Documents/paddingsolution/newsol/blender_export"

print("📌 Loading faces...")
faces = np.load(os.path.join(DATA_DIR, "faces.npy"))
V = np.max(faces) + 1

print(f"➡️ Total vertices: {V}")
print(f"➡️ Total faces: {len(faces)}")

# ---------------------------------------------------------
# 1-HOP ADJACENCY
# ---------------------------------------------------------
print("\n🔧 Building 1-hop adjacency...")
neighbors_1hop = defaultdict(set)

for a, b, c in faces:
    neighbors_1hop[a].update([b, c])
    neighbors_1hop[b].update([a, c])
    neighbors_1hop[c].update([a, b])

adj_1hop = [list(neighbors_1hop[i]) for i in range(V)]

# ---------------------------------------------------------
# CHECK 1: Any vertex with NO 1-hop neighbors?
# ---------------------------------------------------------
print("\n🔍 Checking for isolated vertices (should be 0)...")
isolated = [i for i in range(V) if len(adj_1hop[i]) == 0]

if len(isolated) == 0:
    print("✅ No isolated vertices found.")
else:
    print("❌ ERROR: Found isolated vertices:", isolated)

# ---------------------------------------------------------
# 2-HOP ADJACENCY
# ---------------------------------------------------------
print("\n🔧 Building 2-hop adjacency...")

neighbors_2hop = defaultdict(set)

for i in range(V):
    for j in adj_1hop[i]:
        neighbors_2hop[i].update(adj_1hop[j])

    neighbors_2hop[i].discard(i)
    neighbors_2hop[i] -= set(adj_1hop[i])

adj_2hop = [list(neighbors_2hop[i]) for i in range(V)]

# ---------------------------------------------------------
# STATS
# ---------------------------------------------------------
print("\n📊 Adjacency statistics:")
deg1 = [len(n) for n in adj_1hop]
deg2 = [len(n) for n in adj_2hop]

print(f"➡️ 1-hop: min={min(deg1)}, max={max(deg1)}, avg={np.mean(deg1):.2f}")
print(f"➡️ 2-hop: min={min(deg2)}, max={max(deg2)}, avg={np.mean(deg2):.2f}")

# ---------------------------------------------------------
# COMBINE 1-HOP + 2-HOP
# ---------------------------------------------------------
print("\n🔧 Combining 1-hop + 2-hop adjacency...")
adjacency = []
for i in range(V):
    adjacency.append(list(set(adj_1hop[i]) | set(adj_2hop[i])))

# ---------------------------------------------------------
# RANDOM SPOT CHECKS
# ---------------------------------------------------------
print("\n🔍 Performing random spot checks...")
for _ in range(5):
    vid = np.random.randint(0, V)
    print(f"\nVertex {vid}:")
    print(f"  1-hop ({len(adj_1hop[vid])}): {adj_1hop[vid][:10]}{'...' if len(adj_1hop[vid])>10 else ''}")
    print(f"  2-hop ({len(adj_2hop[vid])}): {adj_2hop[vid][:10]}{'...' if len(adj_2hop[vid])>10 else ''}")
    print(f"  Combined ({len(adjacency[vid])}): {adjacency[vid][:10]}{'...' if len(adjacency[vid])>10 else ''}")

# ---------------------------------------------------------
# SAVE adjacency.npy
# ---------------------------------------------------------
np.save(os.path.join(DATA_DIR, "adjacency.npy"), np.array(adjacency, dtype=object))
print("\n💾 Saved adjacency.npy")

# ---------------------------------------------------------
# BUILD adjacency_matrix.npy
# ---------------------------------------------------------
print("\n🔧 Building adjacency_matrix.npy...")

max_deg = max(len(n) for n in adjacency)
print(f"➡️ Max combined neighbors: {max_deg}")

adj_mat = -np.ones((V, max_deg), dtype=np.int32)

for i, neigh in enumerate(adjacency):
    adj_mat[i, :len(neigh)] = neigh

# ---------------------------------------------------------
# SANITY CHECKS
# ---------------------------------------------------------
print("\n🔍 Checking adjacency_matrix correctness...")

errors = 0
for _ in range(5):
    vid = np.random.randint(0, V)
    original = set(adjacency[vid])
    matrix_vals = set(adj_mat[vid][adj_mat[vid] != -1])

    if original != matrix_vals:
        print(f"❌ Mismatch at vertex {vid}")
        print("Original:", original)
        print("Matrix:  ", matrix_vals)
        errors += 1
    else:
        print(f"✅ Vertex {vid} OK")

if errors == 0:
    print("\n🎉 All matrix checks passed.")
else:
    print(f"\n⚠️ Found {errors} mismatches. Check adjacency generation.")

# ---------------------------------------------------------
# SAVE adjacency_matrix.npy
# ---------------------------------------------------------
np.save(os.path.join(DATA_DIR, "adjacency_matrix.npy"), adj_mat)
print("\n💾 Saved adjacency_matrix.npy")

print("\n🎉 DONE — adjacency + adjacency_matrix ready for fast curvature loss.")
