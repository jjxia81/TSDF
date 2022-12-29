import torch
import os
import numpy as np
from io import StringIO 
import open3d as o3d
import mcubes
import matplotlib.pyplot as plt
from skimage import measure
import cv2

data_dir = "/home/jjxia/hdd/scene0241_data/"
depth_intrinsic_dir = '/home/jjxia/hdd/scene0241_data/intrinsic'

def meshwrite(filename, verts, faces, norms, colors):
  """Save a 3D mesh to a polygon .ply file.
  """
  # Write header
  ply_file = open(filename,'w')
  ply_file.write("ply\n")
  ply_file.write("format ascii 1.0\n")
  ply_file.write("element vertex %d\n"%(verts.shape[0]))
  ply_file.write("property float x\n")
  ply_file.write("property float y\n")
  ply_file.write("property float z\n")
  ply_file.write("property float nx\n")
  ply_file.write("property float ny\n")
  ply_file.write("property float nz\n")
  ply_file.write("property uchar red\n")
  ply_file.write("property uchar green\n")
  ply_file.write("property uchar blue\n")
  ply_file.write("element face %d\n"%(faces.shape[0]))
  ply_file.write("property list uchar int vertex_index\n")
  ply_file.write("end_header\n")

  # Write vertex list
  for i in range(verts.shape[0]):
    ply_file.write("%f %f %f %f %f %f %d %d %d\n"%(
      verts[i,0], verts[i,1], verts[i,2],
      norms[i,0], norms[i,1], norms[i,2],
      colors[i,0], colors[i,1], colors[i,2],
    ))

  # Write face list
  for i in range(faces.shape[0]):
    ply_file.write("3 %d %d %d\n"%(faces[i,0], faces[i,1], faces[i,2]))

  ply_file.close()

def read_camera_pose(pose_path):
    pose_file = open(pose_path, "r")
    pose_data = pose_file.read()
    # print(depth_intrinsic_data)
    return np.loadtxt(StringIO(pose_data))

def create_point_clouds_bbox_from_depth(depth_img, cam_pose, cam_intr):
    depth_img = np.transpose(depth_img)
    img_shape = depth_img.shape
    img_x, img_y = np.meshgrid(
            range(img_shape[0]),
            range(img_shape[1]),
            indexing='ij'
        )
    img_x = (img_x - cam_intr[0,2]) / cam_intr[0,0]
    img_y = (img_y - cam_intr[1,2]) / cam_intr[1,1]
    R = cam_pose[:3,:3]
    # R = np.transpose(R)
    new_img_x = img_x * R[0][0] + img_y * R[0][1] + depth_img * R[0][2] + cam_pose[0][3]
    new_img_y = img_x * R[1][0] + img_y * R[1][1] + depth_img * R[1][2] + cam_pose[1][3]
    new_img_z = img_x * R[2][0] + img_y * R[2][1] + depth_img * R[2][2] + cam_pose[2][3]
    min_p = (np.min(new_img_x), np.min(new_img_y), np.min(new_img_z))
    max_p = (np.max(new_img_x), np.max(new_img_y), np.max(new_img_z))
    return min_p, max_p

def get_intrinsic(use_predicted_depth=False):
    if use_predicted_depth:
        depth_intrinsic_path  = os.path.join(depth_intrinsic_dir, 'intrinsic_color.txt')
        # width = 1296
        # height = 968
    else:
        depth_intrinsic_path  = os.path.join(depth_intrinsic_dir, 'intrinsic_depth.txt')
        # width = 640
        # height = 480
    depth_intrinsic_mat = read_camera_pose(depth_intrinsic_path)
    return depth_intrinsic_mat

# cpu version of tsdf with numpy implementation
class TSDF_np:
    def __init__(self, intrinsic, volumn_bounds, voxel_len) -> None:
        self.voxel_len = voxel_len
        self.origin = volumn_bounds[:,0].copy(order='C').astype(np.float32)
        self.volumn_bounds = volumn_bounds
        print('volumn_bounds:',  self.volumn_bounds)
        self.volumn_shape = np.ceil((volumn_bounds[:, 1] - volumn_bounds[:,0]) / voxel_len).astype(int)
        
        print('v shape :',  self.volumn_shape)
        # self.volumn_size = volumn_size
        # self.volumn_shape = (volumn_size, volumn_size, volumn_size)
        self.tsdf = np.ones(self.volumn_shape)
        self.tsdf_color = np.zeros(self.volumn_shape)
        self.tsdf_weights = np.zeros(self.volumn_shape)
        
        self.fx = intrinsic[0][0]
        self.fy = intrinsic[1][1]
        self.cx = intrinsic[0][2]
        self.cy = intrinsic[1][2]
        self.img_width = 640
        self.img_height = 480
        self.trunc_dist = self.voxel_len * 5
        self.create_volumn()
        self.voxel_R = np.zeros(self.volumn_shape).astype(float)
        self.voxel_G = np.zeros(self.volumn_shape).astype(float)
        self.voxel_B = np.zeros(self.volumn_shape).astype(float) 
        
    def create_volumn(self):
        # print(volumn_x)
        # Get voxel grid coordinates
        self.v_x, self.v_y, self.v_z = np.meshgrid(
            range(self.volumn_shape[0]),
            range(self.volumn_shape[1]),
            range(self.volumn_shape[2]),
            indexing='ij'
        )
        print('self.v_x shape :' , self.v_x.shape)
        self.v_x_world = self.v_x * self.voxel_len + self.origin[0]
        self.v_y_world = self.v_y * self.voxel_len + self.origin[1] 
        self.v_z_world = self.v_z * self.voxel_len + self.origin[2]
        
    def project_voxel_to_img_plane(self, camera_mat):
        x_trans = self.v_x_world - camera_mat[0][3]
        y_trans = self.v_y_world - camera_mat[1][3]
        z_trans = self.v_z_world - camera_mat[2][3]
        pose = camera_mat[:3,:3]
        pose = np.linalg.inv(pose)
        self.img_x = x_trans * pose[0][0]  \
                        + y_trans * pose[0][1]  \
                        + z_trans * pose[0][2] 
        self.img_y = x_trans * pose[1][0]  \
                        + y_trans * pose[1][1]  \
                        + z_trans * pose[1][2] 
        self.img_z = x_trans * pose[2][0]  \
                        + y_trans * pose[2][1]  \
                        + z_trans * pose[2][2]
        self.img_x = np.round(self.img_x / self.img_z * self.fx + self.cx ).astype(int)
        self.img_y = np.round(self.img_y / self.img_z * self.fy + self.cy ).astype(int)
                        
        self.voxel_mask = np.where(self.img_x >= 0, 1, 0)
        # print('non zero : ', np.count_nonzero(self.voxel_mask==1))
        self.voxel_mask = np.where(self.img_x < self.img_width, self.voxel_mask, 0)
        # print('non zero : ', np.count_nonzero(self.image_mask==1))
        self.voxel_mask = np.where(self.img_y >= 0, self.voxel_mask, 0)
        # print('non zero : ', np.count_nonzero(self.image_mask==1))
        self.voxel_mask = np.where(self.img_y < self.img_height, self.voxel_mask, 0)
        # print('non zero : ', np.count_nonzero(self.image_mask==1))
        self.voxel_mask = np.where(self.img_z > 0, self.voxel_mask, 0)
        
    def integrate_tsdf(self, depth_img, color_img):
        img_depth_array = depth_img.flatten()
        # print('non zero : ', np.count_nonzero(self.voxel_mask==1))
        voxel_index = np.where(self.voxel_mask == 1, self.img_y * self.img_width + self.img_x, 0)
        self.voxel_depth = np.where(self.voxel_mask == 1, img_depth_array[voxel_index], 0)
        self.voxel_mask = np.where(self.voxel_depth > 0, self.voxel_mask, 0)
        img_r = color_img[:,:,0].flatten()
        img_g = color_img[:,:,1].flatten()
        img_b = color_img[:,:,2].flatten()
        
        voxel_color_r = np.where(self.voxel_mask == 1, img_r[voxel_index]/255, 0)
        voxel_color_g = np.where(self.voxel_mask == 1, img_g[voxel_index]/255, 0)
        voxel_color_b = np.where(self.voxel_mask == 1, img_b[voxel_index]/255, 0)
        
        cur_tsdf = np.where(self.voxel_mask == 1, self.voxel_depth - self.img_z, 0)/ self.trunc_dist
        self.voxel_mask = np.where(cur_tsdf < -1, 0, self.voxel_mask)
        cur_tsdf = np.where(cur_tsdf > 1, 1, cur_tsdf)
        cur_tsdf = np.where(cur_tsdf < -1, -1, cur_tsdf)
        
        cur_weights = self.voxel_mask #np.where(self.voxel_mask == 1, self.img_z / np.sqrt(self.img_x * self.img_x + self.img_y * self.img_y + self.img_z * self.img_z) , 0)
        
        self.tsdf = np.where(self.voxel_mask == 1, self.tsdf + cur_weights * cur_tsdf, self.tsdf)
        self.tsdf_weights = self.tsdf_weights + cur_weights
        
        self.voxel_R = self.voxel_R + voxel_color_r * cur_weights
        self.voxel_G = self.voxel_G + voxel_color_g * cur_weights
        self.voxel_B = self.voxel_B + voxel_color_b * cur_weights
        
    def generate_mesh(self):
        # tsdf_vol, color_vol = self.get_volume()
        # Marching cubes
        self.tsdf = np.where(self.tsdf_weights > 0, self.tsdf / self.tsdf_weights, self.tsdf)
        self.voxel_R = np.where(self.tsdf_weights > 0, np.fmin(255, self.voxel_R / self.tsdf_weights * 255), 0).astype(int)
        self.voxel_G = np.where(self.tsdf_weights > 0, np.fmin(255, self.voxel_G / self.tsdf_weights * 255), 0).astype(int)
        self.voxel_B = np.where(self.tsdf_weights > 0, np.fmin(255, self.voxel_B / self.tsdf_weights * 255), 0).astype(int)
        
        verts, faces, norms, vals = measure.marching_cubes(self.tsdf, level=0)
        verts_ind = np.round(verts).astype(int)
        verts = verts * self.voxel_len + self.origin  # voxel grid coordinates to world coordinates
        # Get vertex colors
        vert_r = self.voxel_R[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
        vert_g = self.voxel_G[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]
        vert_b = self.voxel_B[verts_ind[:,0], verts_ind[:,1], verts_ind[:,2]]  
        colors = np.asarray([vert_r,vert_g,vert_b]).T
        colors = colors.astype(np.uint8)
        return verts, faces, norms, colors
    
if __name__ == "__main__":
    # volumn_size = 256
    pose_dir = data_dir + 'pose'
    depth_dir = data_dir + 'depth'
    color_dir = data_dir + 'color_resize' 
    use_predicted_depth = False
    if use_predicted_depth:
        color_dir = data_dir + 'color'
        depth_dir = data_dir + 'output_depth'
    intrinsic = get_intrinsic(use_predicted_depth)
    
    color_imgs = os.listdir(color_dir)
    vol_bnds = np.zeros((3,2))
    for i in range(len(color_imgs)):
        # Read depth image and camera pose
        if not use_predicted_depth:
            depth = o3d.io.read_image(os.path.join(depth_dir, str(i) + '.png'))
            img_depth = np.asarray(depth) / 1000.0
        # else:
        #     depth = o3d.io.read_image(os.path.join(depth_dir, str(i) + '-depth_raw.png'))
        #     img_depth = np.asarray(depth) / 10000.0
        img_depth[img_depth == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
        # cam_pose = np.loadtxt("data/frame-%06d.pose.txt"%(i))  # 4x4 rigid transformation matrix
        cam_pose  = read_camera_pose (os.path.join(pose_dir , str(i) + '.txt'))
        min_p, max_p = create_point_clouds_bbox_from_depth(img_depth,cam_pose, intrinsic)
        vol_bnds[:,0] = np.fmin(vol_bnds[:,0], min_p)
        vol_bnds[:,1] = np.fmax(vol_bnds[:,1], max_p)

    voxel_len = 0.05
    tsdf = TSDF_np(intrinsic, vol_bnds, voxel_len)
    print("Initializing voxel volume...")
    # tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.04)
    for i in range(len(color_imgs)):
        print('integrate {}th image'.format(i))
        if not use_predicted_depth:
            depth = o3d.io.read_image(os.path.join(depth_dir, str(i) + '.png'))
            depth_img = np.asarray(depth) / 1000.0
        # else:
        #     depth = o3d.io.read_image(os.path.join(depth_dir, str(i) + '-depth_raw.png'))
        #     depth_img = np.asarray(depth) / 10000.0
            
        color_img_path = os.path.join(color_dir, color_imgs[i])
        color_img = cv2.cvtColor(cv2.imread(color_img_path), cv2.COLOR_BGR2RGB)
        # img_depth = np.transpose(img_depth)
        pose  = read_camera_pose (os.path.join(pose_dir , str(i) + '.txt'))
        tsdf.project_voxel_to_img_plane(pose)
        tsdf.integrate_tsdf(color_img=color_img, depth_img= depth_img)
   
    print("Saving mesh to mesh.ply...")
    # verts, faces, norms, colors = tsdf_vol.get_mesh()
    verts, faces, norms, colors = tsdf.generate_mesh()
    meshwrite("mesh.ply", verts, faces, norms, colors)
  