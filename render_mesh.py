import os
import os.path as osp
import numpy as np
import json
from pathlib import Path
from typing import List

from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

from glob import glob
import cv2
import open3d as o3d
import trimesh
import json


def load_o3d_cam_pose(cam_pose_file):
    with open(cam_pose_file, 'r') as ifs:
        trajectory = json.load(ifs)
        assert( trajectory['class_name'] == "PinholeCameraParameters")

        T_w_c = np.eye(4)
        intrinsic_matrix = np.eye(3)

        extrinsics = trajectory['extrinsic']
        T_w_c[:,0] = extrinsics[:4]
        T_w_c[:,1] = extrinsics[4:8]
        T_w_c[:,2] = extrinsics[8:12]
        T_w_c[:,3] = extrinsics[12:]
        # print(T_w_c)

        intrinsics = trajectory['intrinsic']
        img_width = intrinsics['width']
        img_height = intrinsics['height']
        intrin_mat = intrinsics['intrinsic_matrix']
        intrinsic_matrix[:,0] = intrin_mat[:3]
        intrinsic_matrix[:,1] = intrin_mat[3:6]
        intrinsic_matrix[:,2] = intrin_mat[6:]
        # print(intrinsic_matrix)
        return T_w_c, intrinsic_matrix


def interp_and_convert_o3d_render_json(v_cam_pose_files:List, save_o3d_camera_traj_filepath:str):
    v_rotation_in = np.zeros([0, 4])
    v_pos_x_in = []
    v_pos_y_in = []
    v_pos_z_in = []
    for cam_pose_file in v_cam_pose_files:
        T_w_c, intrinsic = load_o3d_cam_pose(cam_pose_file)
        v_rotation_in = np.append(v_rotation_in, [Rotation.from_matrix(T_w_c[:3,:3]).as_quat()], axis=0)
        v_pos_x_in.append(T_w_c[0,3])
        v_pos_y_in.append(T_w_c[1,3])
        v_pos_z_in.append(T_w_c[2,3])

    in_times = np.arange(0, len(v_rotation_in)).tolist()
    out_times = np.linspace(0, len(v_rotation_in)-1, len(v_rotation_in)*30).tolist()
    print(f'in_times: {(in_times)}')
    print(f'out_times: {(out_times)}')
    v_rotation_in = Rotation.from_quat(v_rotation_in)
    slerp = Slerp(in_times, v_rotation_in)
    v_interp_rotation = slerp(out_times)

    fx = interp1d(in_times, np.array(v_pos_x_in), kind='quadratic')
    fy = interp1d(in_times, np.array(v_pos_y_in), kind='quadratic')
    fz = interp1d(in_times, np.array(v_pos_z_in), kind='quadratic')
    v_interp_xs = fx(out_times)
    v_interp_ys = fy(out_times)
    v_interp_zs = fz(out_times)

    root_node = {}
    root_node["class_name"] = "PinholeCameraTrajectory"

    intrinsic_node = {}
    intrinsic_node['width'] = 720
    intrinsic_node['height'] = 544
    cam_intrinsic = np.array([[471.11781966, 0., 359.5],
                                [0., 471.11781966, 271.5],
                                [0.,  0., 1. ]], dtype=np.float32)
    cam_intrinsic_params = []
    cam_intrinsic_params += cam_intrinsic[:,0].tolist()
    cam_intrinsic_params += cam_intrinsic[:,1].tolist()
    cam_intrinsic_params += cam_intrinsic[:,2].tolist()
    intrinsic_node['intrinsic_matrix'] = cam_intrinsic_params

    v_cam_nodes = []
    for idx in range(len(out_times)):
        cam_node = {}
        cam_node['class_name'] = "PinholeCameraParameters"
        cam_node['version_major'] = 1
        cam_node['version_minor'] = 0

        rot_matrix = v_interp_rotation[idx].as_matrix()
        trans = np.array([v_interp_xs[idx], v_interp_ys[idx], v_interp_zs[idx]])
        cam_ext = np.eye(4)
        cam_ext[:3,:3] = rot_matrix
        cam_ext[:3,3] = trans
        cam_ext_params = []
        cam_ext_params += cam_ext[:,0].tolist()
        cam_ext_params += cam_ext[:,1].tolist()
        cam_ext_params += cam_ext[:,2].tolist()
        cam_ext_params += cam_ext[:,3].tolist()
        cam_node['extrinsic'] = cam_ext_params
        cam_node['intrinsic'] = intrinsic_node
        v_cam_nodes.append(cam_node)

    root_node['parameters'] = v_cam_nodes
    root_node["version_major"] = 1
    root_node["version_minor"] = 0

    with open(save_o3d_camera_traj_filepath, 'w') as fc:
        json.dump(root_node, fc)

    return root_node

def custom_draw_geometry_with_camera_trajectory(pcd, render_option_path,
                                                camera_trajectory_path, render_output_path: str):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.frame_idx = 0
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    image_path = os.path.join(render_output_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(render_output_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
        # The run loop calls the function, then re-render
        # So the sequence in this function is to:
        # 1. Capture frame
        # 2. index++, check ending criteria
        # 3. Set camera
        # 4. (Re-render)
        ctr = vis.get_view_control()
        glb = custom_draw_geometry_with_camera_trajectory
        if glb.index >= 0:
            # if glb.index % 3 == 0:
            print("Capture image {:05d}".format(glb.frame_idx))
            depth = vis.capture_depth_float_buffer(False)
            image = vis.capture_screen_float_buffer(False)
            plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(glb.frame_idx)),
                       np.asarray(depth),
                       dpi=1)
            plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.frame_idx)),
                       np.asarray(image),
                       dpi=1)
            glb.frame_idx += 1

            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window(width=720, height=544)
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()

def merge_image_sequences_to_video(img_seq1_folder:str, img_seq2_folder:str, output_video_filepath:str):

    if not osp.isdir(img_seq1_folder):
        print(f'folder {img_seq1_folder} doesnt exist!')
        exit(-1)

    if not osp.isdir(img_seq2_folder):
        print(f'folder {img_seq2_folder} doesnt exist!')
        exit(-1)

    v_rgb_img_file = [img_f for img_f in os.listdir(
        img_seq1_folder) if img_f.endswith('.png')]
    v_rgb_img_file.sort(key=lambda x: int(x.split('.')[0]))
    print(v_rgb_img_file)
    v_sem_img_file = [img_f for img_f in os.listdir(
        img_seq2_folder) if img_f.endswith('.png')]
    v_sem_img_file.sort(key=lambda x: int(x.split('.')[0]))

    assert len(v_rgb_img_file) == len(
        v_sem_img_file) and (len(v_rgb_img_file) > 0)
    # 根据图片的大小，创建写入对象 （文件名，支持的编码器，5帧，视频大小（图片大小））
    size = (720, 544)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWrite = cv2.VideoWriter(output_video_filepath, fourcc, 25.0, size)

    img_num = len(v_rgb_img_file)
    for img_idx in range(img_num):
        rgb_img_filepath = osp.join(img_seq1_folder, v_rgb_img_file[img_idx])
        sem_img_filepath = osp.join(img_seq2_folder, v_sem_img_file[img_idx])
        rgb_img = cv2.imread(rgb_img_filepath)  # 读取第一张图片
        sem_img = cv2.imread(sem_img_filepath)  # 读取第一张图片
        img_height, img_width, _ = rgb_img.shape
        # print(f'rgb_img.shape: {rgb_img.shape}')

        merge_width = int(img_width/2)
        merge_img = np.concatenate(
            (rgb_img[:, 0:merge_width], sem_img[:, merge_width:]), axis=1)
        # print(f'merge_img.shape: {merge_img.shape}')
        if img_idx == 0:
            cv2.imwrite(
                '/home/ziqianbai/Projects/vlab/Open3D/examples/test_data/merge_img.png', merge_img)

        videoWrite.write(merge_img)  # 将图片写入所创建的视频对象

    videoWrite.release()
    print('end!')


if __name__ == '__main__':
    mesh_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/videoclip/scene_03000_238/model.obj'

    input_folderpath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/videoclip/scene_03000_238/'

    save_o3d_camera_traj_filepath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/videoclip/scene_03000_238/camera_trajectory.json'
    render_output_folderpath = '/mnt/nas_3dv/hdd1/datasets/fangchuan/codes/videoclip/scene_03000_238/render_output'

    render_option_path = 'render.json'
    v_cam_pose_files = glob(osp.join(input_folderpath, 'DepthCamera*.json'))
    # v_cam_pose_files = ['/Users/fc/Desktop/DepthCamera_2023-04-13-16-15-32.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-15-46.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-16-15.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-16-28.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-16-57.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-17-18.json',
    # '/Users/fc/Desktop/DepthCamera_2023-04-13-16-17-50.json',]
    v_cam_pose_files.sort(key=lambda x:osp.basename(x))
    print(v_cam_pose_files)

    # interpolated_camera_trajectory  and save to json
    interp_and_convert_o3d_render_json(v_cam_pose_files, save_o3d_camera_traj_filepath)


    pcd_flipped = o3d.io.read_triangle_mesh(mesh_filepath, True)
    # pcd_flipped.compute_vertex_normals()
    # # Flip it, otherwise the pointcloud will be upside down
    # pcd_flipped.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
    #                        [0, 0, 0, 1]])

    print("6. Customized visualization playing a camera trajectory")
    custom_draw_geometry_with_camera_trajectory(pcd_flipped, render_option_path, save_o3d_camera_traj_filepath,render_output_folderpath)