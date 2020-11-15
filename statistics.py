import bpy
import json
from mathutils import Vector
import numpy as np
from scipy.spatial import distance

def compute_distances(objects):
    vertices_objects = []
    for obj in objects:
        verts = np.array([list(obj.matrix_world @ v.co) for v in obj.data.vertices])
        verts = verts[np.random.choice(len(verts), 200, replace=False)] if len(verts) > 200 else verts
        print(verts.shape)
        vertices_objects.append(verts)
    minimum_distances = np.zeros((len(objects), len(objects)), dtype=np.float32)
    vertices_objects = np.array(vertices_objects)
    for i, verts_a in enumerate(vertices_objects):
        for j, verts_b in enumerate(vertices_objects):
            if i == j: continue
            minimum_distances[i, j] = np.min(distance.cdist(verts_a, verts_b))
    return minimum_distances.tolist()

def inside_box(p, min, max):
    return all([l <= x <= h for (x, l,h) in zip(p, min, max)])

def thingi10k_test(obj):
    name = obj.name
    try:
        for d in name:
            int(d)
    except Exception as e:
        return False
    return True

fixed_furniture = [ff for ff in bpy.data.objects if "FixedFurniture" in ff.name or "sink" in ff.name or "mirror" in ff.name]
furniture_group = [fg for fg in bpy.data.objects if fg.parent and "root" in fg.parent.name and not thingi10k_test(fg)]

furnitures = fixed_furniture + furniture_group

stats_file = bpy.path.abspath("//statistics.json")

rooms = [f for f in bpy.data.objects if f.name.startswith("Space") and ".Ceiling" in f.name]
room_stats = []
for ceil in rooms:
    floor = bpy.data.objects[ceil.name.replace("Ceiling", "Floor")]
    bb_floor =  [floor.matrix_world @ Vector(v) for v in floor.bound_box]
    bb_ceil  =  [ceil.matrix_world @ Vector(v) for v in ceil.bound_box]
    bb = np.array(bb_floor + bb_ceil)
    min = np.min(bb,axis=0)
    max = np.max(bb,axis=0)
    c_s = []
    for cam in [o for o in bpy.data.objects if o.type=="CAMERA"]:
        if inside_box(cam.location, min, max): 
            c_s.append({"id": cam.name, "pos": list(cam.location.xyz)})
    
    f_s = []
    f_of_room = []
    for furniture in furnitures:
        bb_f  =  [furniture.matrix_world @ Vector(v) for v in furniture.bound_box]
        min_f = np.min(bb_f,axis=0)
        max_f = np.max(bb_f,axis=0)
        center = min_f + (max_f - min_f) * 0.5
        if inside_box(center, min, max): 
            f_of_room.append(furniture)
            f_s.append({
                "name": furniture.name,
                "bbox": {"min": np.around(min_f, 3).tolist(), "max": np.around(max_f, 3).tolist()}
            })        
    s = {
    "name": ceil.name.replace(".Ceiling",""),
    "bbox": {"min": np.around(min, 3).tolist(), "max": np.around(max, 3).tolist()},
    "furniture": f_s,
    "cameras": c_s,
    "distances": compute_distances(f_of_room)
    }
    room_stats.append(s)

stats = {
"n_rooms": len([1 for f in bpy.data.objects if f.name.startswith("Space") and ".Ceiling" in f.name]),
"n_cameras": len([1 for c in bpy.data.objects if c.type == 'CAMERA']),
"rooms": room_stats
}

with open(stats_file, "w") as json_file:
    json.dump(stats, json_file)