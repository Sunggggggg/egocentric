import cv2
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from torchvision.utils import make_grid
from smplx import SMPLH
import configs.constant as _C
from ..vis.renderer import Renderer, look_at_view_transform

width, height = 500, 500
focal_len = (width ** 2 + height ** 2) ** 0.5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
faces = SMPLH(_C.SMPL.SMPLH_MODEL_PATH, num_betas=10, ext='pkl').faces
renderer = Renderer(width, height, focal_len, device, faces)

def single_pose(vertices):
    """ vertices : [6890, 3]
    """
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    cam_r, cam_t = look_at_view_transform(dist=3., elev=0., azim=0.)
    renderer.create_camera(cam_r, cam_t)
    img = renderer.render_mesh(vertices.to(device), background.copy(), )
    return img

@torch.no_grad()
def visualize_mesh(input_batch, output_batch, save_name, nb_iter):
    pred_vertices = input_batch['body_vertices'][:, 0]             # [T, 6890, 3]
    gt_vertices = output_batch['pred_body_vertices'][:, 0]         # [T, 6890, 3]
    background = np.ones((height, width, 3), dtype=np.uint8) * 255
    batch_size = len(gt_vertices)
    
    cam1_r, cam1_t = look_at_view_transform(dist=3., elev=45., azim=0.)
    cam2_r, cam2_t = look_at_view_transform(dist=2., elev=45., azim=90.)
    renderer.create_camera(cam1_r, cam1_t)
    
    imgs = []
    for t in range(batch_size)[:8] :
        gt_vert = gt_vertices[t].cuda()
        pred_vert = pred_vertices[t].cuda()
        
        gt_img = renderer.render_mesh(gt_vert, background.copy(), )
        pred_img = renderer.render_mesh(pred_vert, background.copy(), )
        
        img = np.hstack([gt_img, pred_img]) # [H, 2W, 3]
        imgs.append(img)
    
    rend_img = np.vstack(imgs)
    rend_img = cv2.cvtColor(rend_img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(os.path.join(save_name, f'rendered_{nb_iter:06}.jpg'), rend_img)
    
    return rend_img
    

SMPL_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (0, 3), # pelvis -> hips, spine
    (1, 4), (2, 5), # hips -> knees
    (3, 6), (6, 9), # spine1 -> spine2 -> spine3
    (4, 7), (5, 8), # knees -> ankles
    (7, 10), (8, 11), # ankles -> feet
    (12, 13), # spine3 -> neck, shoulders
    (13, 16), # shoulders -> elbows
    (16, 18), (17, 19), # elbows -> wrists
    (18, 20), (15, 17), (12, 15)
]

def motion_contact_gif(motion_data, contact_labels):
    """
    motion_data     : [T, J, 3]
    contact_labels  : [T, J]
    """
    if isinstance(motion_data, torch.Tensor):
        motion_data = motion_data.detach().cpu().numpy()
    if isinstance(contact_labels, torch.Tensor):
        contact_labels = contact_labels.detach().cpu().numpy()
    T, J = motion_data.shape[:2]
    
    x_min, x_max = motion_data[:, :, 0].min(), motion_data[:, :, 0].max()
    y_min, y_max = motion_data[:, :, 1].min(), motion_data[:, :, 1].max()
    z_min, z_max = motion_data[:, :, 2].min(), motion_data[:, :, 2].max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    padding = 0.1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    ax.set_zlim(z_min - padding * z_range, z_max + padding * z_range)

    initial_joints = motion_data[0]
    sc = ax.scatter(initial_joints[:, 0], initial_joints[:, 1], initial_joints[:, 2], c='gray', s=50)
    
    initial_line_segments = []
    for joint_pair in SMPL_SKELETON_CONNECTIONS:
        start_joint = initial_joints[joint_pair[0]]
        end_joint = initial_joints[joint_pair[1]]
        initial_line_segments.append([start_joint, end_joint])

    lines = Line3DCollection(initial_line_segments, colors='blue', linewidths=2)
    ax.add_collection3d(lines)
    
    ax.set_title('Motion with Contact Visualization')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    texts = [ax.text(initial_joints[j, 0], initial_joints[j, 1], initial_joints[j, 2], str(j)) for j in range(J)]
    X_ground = np.arange(x_min - padding, x_max + padding, 0.5)
    Y_ground = np.arange(y_min - padding, y_max + padding, 0.5)
    X_ground, Y_ground = np.meshgrid(X_ground, Y_ground)

    Z_ground = np.full(X_ground.shape, z_min - 0.05)
    ax.plot_surface(X_ground, Y_ground, Z_ground, alpha=0.3, color='grey', rstride=10, cstride=10)

    def update(frame):
        current_joints = motion_data[frame]
        current_contacts = contact_labels[frame]
        
        colors = np.full(J, 'gray')
        contact_indices = np.where(current_contacts == 1)[0]
        colors[contact_indices] = 'red'

        sc._offsets3d = (current_joints[:, 0], current_joints[:, 1], current_joints[:, 2])
        sc.set_color(colors)
        ax.set_title(f'Frame: {frame}/{T-1}')
        
        line_segments = []
        for joint_pair in SMPL_SKELETON_CONNECTIONS:
            start_joint = current_joints[joint_pair[0]]
            end_joint = current_joints[joint_pair[1]]
            line_segments.append([start_joint, end_joint])
        lines.set_segments(line_segments)
        
        for j in range(J):
            texts[j].set_position((current_joints[j, 0], current_joints[j, 1]))
            texts[j].set_3d_properties(current_joints[j, 2], 'z')
        return sc, lines
    
    ani = FuncAnimation(fig, update, frames=T, interval=50, blit=False)
    ani.save('motion_with_contacts.mp4', writer='ffmpeg', fps=20)
    
    plt.close()