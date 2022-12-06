import bpy
import numpy as np
from math import tan
from mathutils import Vector
import os
import sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
​
print(".........................Start.......................")
​
def point_cloud(depth,cam):
    
    # Distance factor from the camera focal angle
    factor = 2.0 * tan(cam.data.angle_x/2.0)
    
    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    
    # Valid depths are defined by the camera clipping planes
    valid = (depth > cam.data.clip_start) & (depth < cam.data.clip_end)
    
    # Negate Z (the camera Z is at the opposite)
    z = -np.where(valid, depth, np.nan)
    # Mirror X
    # Center c and r relatively to the image size cols and rows
    ratio = max(rows,cols)
    x = -np.where(valid, factor * z * (c - (cols / 2)) / ratio, 0)
    y = np.where(valid, factor * z * (r - (rows / 2)) / ratio, 0)
    
    return np.dstack((x, y, z))
​
​
​
bpy.data.scenes['Scene'].frame_current = 1
framenumber = "0001"
# Render Image so depth and image can be output
base_path = r'/Users/tspc/Downloads/untitled folder 18/'
color_path = base_path + 'color{}.jpg'.format(framenumber)
depth_path = base_path + 'depth{}.exr'.format(framenumber)
​
#print(color_name)
#setting the save path in render nodes
bpy.data.scenes['Scene'].node_tree.nodes['color_output'].base_path = base_path
bpy.data.scenes['Scene'].node_tree.nodes['depth_output'].base_path = base_path
​
#render
bpy.ops.render.render()
​
#Frontface culling
mesh_object=0
principle_shader=0
​
for ob in bpy.data.objects:
     if ob.type == 'MESH':
         mesh_object+=1
         currentMaterial = ob.active_material
         currentMaterial.use_nodes = True
         nodes = currentMaterial.node_tree.nodes
         geo_node = nodes.new(type = 'ShaderNodeNewGeometry')
         for n in nodes:
             if n.type == 'BSDF_PRINCIPLED':
                 principle_shader+=1
                 currentMaterial.node_tree.links.new(geo_node.outputs["Backfacing"], n.inputs["Alpha"])
                 break
         if principle_shader == 0:
             print("No shader found for mesh ",ob.name)
             principle_shader=0
             
if mesh_object==0:
    print("No meshes found")
​
​
#Save backface Depth
framenumber = "0002"
color_back_path = base_path + 'color{}.jpg'.format(framenumber)
depth_back_path = base_path + 'depth{}.exr'.format(framenumber)
​
bpy.data.scenes['Scene'].frame_current = 2
bpy.ops.render.render()
​
​
# Read depth 
#print(depth_path)
front_depth = cv2.imread(depth_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
back_depth = cv2.imread(depth_back_path,  cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
front_depth = front_depth[:,:,1]
back_depth = back_depth[:,:,1]
​
​
# Read color
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="0"
color = cv2.imread(color_path)
​
# Get the camera
cam = bpy.data.objects['Camera']
​
# Calculate the points
front_points = point_cloud(front_depth, cam)
back_points = point_cloud(back_depth, cam)
​
# Get the camera matrix for location conversion
cam_mat = cam.matrix_world
print(cam_mat)
​
# Translate the points
front_verts = np.array([cam_mat @ Vector(p) for r in front_points for p in r])
back_verts = np.array([cam_mat @ Vector(p) for r in back_points for p in r])
​
# Convert color from 0-255 to 0-1, with datatype float64
#color = np.divide(color.astype(np.float64), 255)
​
# Reshape from img shape to shape (width*height, 3), (like 1080, 1920, 3) -> 1080*1920,3 
color = np.reshape(color, (front_verts.shape[0], 3))
​
​
#print shapes
print(color.shape)
print(front_verts.shape)
​
​
#save numpy
colorn_name = 'color{}.npy'.format(framenumber)
depthn_name = 'depth{}.npy'.format(framenumber)
colorn_path = base_path + colorn_name
depthn_path = base_path + depthn_name
​
#np.save(colorn_path,color)
#np.save(depthn_path,verts)
​
# Set Pointcloud outputpath, create a pointcloud with depth and color information and save it
ply_file_path = base_path + '/{}data.ply'.format(framenumber)
points = []
for v in range(color.shape[0]):
    if not np.isnan(front_verts[v,0]):
        points.append("%f %f %f %d %d %d 0\n"%(front_verts[v,0],front_verts[v,1],front_verts[v,2],color[v,2],color[v,1],color[v,0]))
    if not np.isnan(back_verts[v,0]):
        points.append("%f %f %f %d %d %d 0\n"%(back_verts[v,0],back_verts[v,1],back_verts[v,2],color[v,2],color[v,1],color[v,0]))
​
file = open(ply_file_path,"w")
file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
file.close()
​
print(".........................Finished.......................")
