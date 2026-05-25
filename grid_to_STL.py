import numpy as np
from skimage import measure
from stl import mesh

# 1. Load your 2D design
data = np.load("final_gripper.npy")

# 3. Extrude the 2D data into 3D
thickness = 1
volume = np.zeros((data.shape[0], data.shape[1], thickness), dtype=np.uint8)
for i in range(thickness):
    volume[:, :, i] = data

# --- THE FIX ---
# 4. Pad the entire 3D array with a 1-pixel border of 0s
padded_volume = np.pad(volume, pad_width=1, mode='constant', constant_values=0)

# 5. Use Marching Cubes on the PADDED volume
verts, faces, normals, values = measure.marching_cubes(padded_volume, level=0.5)

# 6. Create the STL mesh object
gripper_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        gripper_mesh.vectors[i][j] = verts[f[j], :]

# 7. Save the file
gripper_mesh.save("gripper_closed.stl")