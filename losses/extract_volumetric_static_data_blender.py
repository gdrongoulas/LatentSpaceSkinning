import bpy
import numpy as np
import json
import os

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
OUTPUT_DIR = "/home/gdrongoulas/Documents/paddingsolution/newsol/blender_export"
ARMATURE_NAME = "Armature"

os.makedirs(OUTPUT_DIR, exist_ok=True)

arm_obj = bpy.data.objects[ARMATURE_NAME]

# ---------------------------------------------------------
# 1. Unified bone list
# ---------------------------------------------------------
bone_names = [b.name for b in arm_obj.data.bones]
bone_index = {name: i for i, name in enumerate(bone_names)}
B = len(bone_names)

print("Total bones:", B)

# ---------------------------------------------------------
# 2. Find mesh objects skinned to armature
# ---------------------------------------------------------
mesh_objects = []
for obj in bpy.data.objects:
    if obj.type == 'MESH' and len(obj.data.vertices) > 0:
        for mod in obj.modifiers:
            if mod.type == 'ARMATURE' and mod.object == arm_obj:
                mesh_objects.append(obj)
                break

print("Meshes:", [o.name for o in mesh_objects])

# ---------------------------------------------------------
# 3. Export
# ---------------------------------------------------------
depsgraph = bpy.context.evaluated_depsgraph_get()

all_vertices = []
all_faces = []
all_weights = []
all_clusters = []

vertex_offset = 0

for mesh_obj in mesh_objects:
    print("Processing:", mesh_obj.name)

    eval_obj = mesh_obj.evaluated_get(depsgraph)

    # -----------------------------
    # 3.1 vertices (original indexing, evaluated deformation)
    # -----------------------------
    verts = []
    for v in mesh_obj.data.vertices:
        co = eval_obj.matrix_world @ eval_obj.data.vertices[v.index].co
        verts.append(co)
    verts = np.array(verts, dtype=np.float32)
    all_vertices.append(verts)

    # -----------------------------
    # 3.2 faces (triangulate quads)
    # -----------------------------
    faces = []
    for poly in mesh_obj.data.polygons:
        v = poly.vertices
        if len(v) == 3:
            faces.append([v[0], v[1], v[2]])
        elif len(v) == 4:
            faces.append([v[0], v[1], v[2]])
            faces.append([v[0], v[2], v[3]])
        else:
            print("WARNING: polygon with", len(v), "vertices ignored")

    faces = np.array(faces, dtype=np.int32) + vertex_offset
    all_faces.append(faces)

    # -----------------------------
    # 3.3 weights (V, B)
    # -----------------------------
    V = len(mesh_obj.data.vertices)
    weights = np.zeros((V, B), dtype=np.float32)

    for v in mesh_obj.data.vertices:
        for g in v.groups:
            vg_name = mesh_obj.vertex_groups[g.group].name
            if vg_name in bone_index:
                weights[v.index, bone_index[vg_name]] = g.weight

    weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-8)
    all_weights.append(weights)

    # -----------------------------
    # 3.4 clusters
    # -----------------------------
    clusters = np.argmax(weights, axis=1).astype(np.int32)
    all_clusters.append(clusters)

    vertex_offset += verts.shape[0]

# ---------------------------------------------------------
# 4. Concatenate
# ---------------------------------------------------------
vertices = np.concatenate(all_vertices, axis=0)
faces = np.concatenate(all_faces, axis=0)
weights = np.concatenate(all_weights, axis=0)
clusters = np.concatenate(all_clusters, axis=0)

# ---------------------------------------------------------
# 5. Save
# ---------------------------------------------------------
np.save(f"{OUTPUT_DIR}/vertices.npy", vertices)
np.save(f"{OUTPUT_DIR}/faces.npy", faces)
np.save(f"{OUTPUT_DIR}/weights.npy", weights)
np.save(f"{OUTPUT_DIR}/clusters.npy", clusters)

with open(f"{OUTPUT_DIR}/bone_names.json", "w") as f:
    json.dump(bone_names, f, indent=2)

print("=== EXPORT COMPLETE ===")
print("Vertices:", vertices.shape)
print("Faces:", faces.shape)
print("Weights:", weights.shape)
print("Clusters:", clusters.shape)
