from __future__ import division
import os, sys, glob
import numpy as np
import pydicom as dicom
from skimage.draw import polygon
np.set_printoptions(threshold = 513)
import time
import open3d as o3d
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  
import cv2
from stl import mesh
from pathlib import Path


def image_show(image, nrows=1, ncols=1, cmap='gray'):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    return fig, ax

def read_structure(structure):
    contours = []
    for i in range(len(structure.ROIContourSequence)):
        contour = {}
        contour['color'] = structure.ROIContourSequence[i].ROIDisplayColor
        contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
        contour['contours'] = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
        contours.append(contour)
    return contours

def get_middle_elements(label):
    non_zero_slices = np.any(label, axis=(0, 1))
    non_zero_indices = np.where(non_zero_slices)[0].tolist()
    n = len(non_zero_indices)
    if n >= 3:
        mid_index = n // 2
        return non_zero_indices[mid_index-1:mid_index+2]
    else:
        fill_value = non_zero_indices[-1] + 1 if non_zero_indices else 0
        return non_zero_indices + [fill_value] * (3 - n)


def get_mask(contours, slices):
    z = [round(s.ImagePositionPatient[2],1) for s in slices]
    pos_r = slices[0].ImagePositionPatient[1]
    spacing_r = slices[0].PixelSpacing[1]
    pos_c = slices[0].ImagePositionPatient[0]
    spacing_c = slices[0].PixelSpacing[0]

    H, W = slices[0].Rows, slices[0].Columns

    contours = sorted(contours, key=lambda x: x['color'][0], reverse=True)
    
    labels = []
    for idx, con in enumerate(contours):
        label = np.zeros((H, W, len(slices)), dtype=np.uint8)
        for c in con['contours']:
            nodes = np.array(c).reshape((-1, 3))
            assert np.amax(np.abs(np.diff(nodes[:, 2]))) == 0
            z_index = z.index(np.around(nodes[0, 2], 1))
            r = (nodes[:, 1] - pos_r) / spacing_r
            c = (nodes[:, 0] - pos_c) / spacing_c
            rr, cc = polygon(r, c)
            label[rr, cc, z_index] = idx + 1
        labels.append(label)

    colors = tuple(np.array([con['color'] for con in contours]) / 255.0)
    return labels, colors

def contours_cnt(image):

    int8_image = np.uint8(image)
    ret, binary = cv2.threshold(int8_image, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours)

def contours_ind(label):
    rslt = []
    for i in range(label.shape[-1]):
        if np.count_nonzero(label[..., i]) > 0:
            rslt.append(i)
    
    return rslt 

def add_applicator_to_lines(lines, another_mesh, ):
    # Calculate center of line start/end points
    start_centers = np.mean([line[0] for line in lines], axis=0)
    end_centers = np.mean([line[1] for line in lines], axis=0)

    center_line_direction = end_centers - start_centers

    height = np.linalg.norm(center_line_direction)
    center_line_direction /= height

    # Create rotation matrix from z-axis to direction
    axis = np.cross([0, 0, 1], center_line_direction)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        theta = np.arccos(np.dot([0, 0, 1], center_line_direction))
        rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
    else:
        rot_matrix = np.eye(3) if np.dot([0, 0, 1], center_line_direction) > 0 else -np.eye(3)
    rot_matrix = np.vstack((rot_matrix, np.zeros(3)))
    rot_matrix = np.column_stack((rot_matrix, np.zeros(4)))
    rot_matrix[-1, -1] = 1

    print(rot_matrix)
    print('rot_matrix')

    another_mesh = another_mesh.transform(rot_matrix)

    # Translate to match line start points
    translation_matrix = np.eye(4)
    translation_matrix[:-1, -1] = start_centers
    another_mesh = another_mesh.transform(translation_matrix)
    
    return another_mesh, rot_matrix, translation_matrix




def read_ct(path):
    """Read CT files and structure data"""
    dcms = glob.glob(os.path.join(path, '*.dcm'))
    if RT_rs == '0':
        dcms_ct = [x for x in dcms if 'rtss' not in x]
        dcms_rs = [x for x in dcms if 'rtss' in x]
    elif RT_rs == '1':
        dcms_ct = [x for x in dcms if 'RS' not in x]
        dcms_rs = [x for x in dcms if 'RS' in x]
    else:
        raise ValueError(f'Invalid CT type flag:{RT_rs}. Valid values: 0-->rtss; 1-->RS')

    # Read structure data
    structure = dicom.read_file(dcms_rs[0])
    contours = read_structure(structure)

    # Read CT slices
    slices = [dicom.read_file(dcm) for dcm in dcms_ct]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    labels, colors = get_mask(contours, slices)
    con_ind = contours_ind(labels[0])

    return image, labels, colors, slices, con_ind

def get_app_type(image, labels, con_ind):
    """Determine applicator type"""
    con_image = image.copy()
    for i in range(image.shape[-1]):
        mask_label = labels[0][..., i] == 0
        mask_threshold = con_image[..., i] <= 950
        con_image[..., i][mask_label] = 0
        con_image[..., i][mask_threshold] = 0
        
    contours_cnt_list = []
    for i in con_ind:
        contours_cnt_list.append(contours_cnt(con_image[..., i]))

    if contours_cnt_list[-1] + contours_cnt_list[-2] > 2:
        print('The vaginal applicator is determined to be a goat horn shape')
        return 'goat'
    else:
        print('The vaginal applicator is determined to be cylindrical')
        return 'cylinder'
    
def get_app_path(app_folder, app_type):

    # folder_path = app_folder.text()
    folder_path = app_folder
    keyword = app_type
    tmp_list = []

    for filename in os.listdir(folder_path):
        if keyword in filename:
            full_path = os.path.join(folder_path, filename)
            tmp_list.append(full_path)        
    tmp_list = sorted(tmp_list)

    return tmp_list[0]

def scale_images(slices, image, labels, colors):
    # Scale images
    scale_factors = [slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], 1]
    image = zoom(image, scale_factors, order=0)
    for i in range(len(labels)):
        labels[i] = zoom(labels[i], scale_factors, order=0)

    num_images = image.shape[2]
    for i in range(num_images):
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image[..., i], cmap="gray")
        if len(labels) > 1:
            ax.contour(labels[0][..., i], levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], colors=colors)
            ax.contour(labels[1][..., i], levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], colors=colors)
        else:
            ax.contour(labels[0][..., i], levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], colors=colors)
        ax.axis('off')
        fig.savefig(f"{output_image_path}/{i}.png")
        plt.close(fig)
    return slices, image, labels

def mask_get(image,
             labels):
    
    if len(labels) > 1:
        image_cancer = np.copy(image)
        image_mold = np.copy(image)
        
        for i in range(image.shape[2]):
            mask1 = labels[1][...,i] == 0
            image_cancer[...,i][mask1] = 0

        for i in range(image.shape[2]):
            mask0 = labels[0][...,i] == 0
            image_mold[...,i][mask0] = 0

        return image_cancer, image_mold    

    else:
        image_mold = np.copy(image)
        for i in range(image.shape[2]):
            mask0 = labels[0][...,i] == 0
            image_mold[...,i][mask0] = 0
    
        return None, image_mold

def save_plt_rst(image, 
                 label, 
                 colors,
                 ind,
                 output_dir,
                 ):
    plt.ioff()
    for i in ind:
        # plt.subplot(5, 5, i + 1)
        plt.imshow(image[..., i], cmap="gray")
        plt.contour(label[..., i], levels=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5], colors=colors)
        plt.savefig(rf"{output_dir}/combined_image_{i}.png")  # Modify this path as needed
    plt.axis('off')
    plt.clf()  # Clear the figure after saving
    plt.close()
    return [rf"{output_dir}/combined_image_{ind[0]}.png", rf"{output_dir}/combined_image_{ind[1]}.png", rf"{output_dir}/combined_image_{ind[2]}.png", ]

def interpolate_slices(image, num_slices=10):
    new_shape = (image.shape[0], image.shape[1], image.shape[2] + num_slices * (image.shape[2] - 1))
    new_image = np.zeros(new_shape)
    
    for z in range(image.shape[2] - 1):
        for i in range(num_slices + 1):
            interpolated_slice = (1 - i / (num_slices + 1)) * image[:, :, z] + (i / (num_slices + 1)) * image[:, :, z + 1]
            new_image[:, :, z * (num_slices + 1) + i] = interpolated_slice                   
                    
    new_image[:, :, -1] = image[:, :, -1]  # Add the last slice
    # new_image[:,:,-1] = np.zeros(image[:,:,-1].shape)
    return new_image


def bin_and_classify_3d_image(image, num_bins):
    flattened_image = image.flatten()
    bin_edges = np.linspace(flattened_image.min(), flattened_image.max(), num_bins + 1)
    bin_indices = np.digitize(flattened_image, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, None)
    return bin_indices.reshape(image.shape)

def image_to_point_cloud(image, z_scale, centroids_ratio=10000000):
    # z_scale == z_scale/num_slices
    print('image_to_point_cloud start')
    print(f"image.shape is ({image.shape[0]}, {image.shape[1]}, {image.shape[2]})")
    x, y, z = np.nonzero(image)
    print(f'temp_1 ok')
    non_zero_points = np.column_stack((x, y, z * z_scale))
    print('image_to_point_cloud finished')
    
    
    
    # cal_num_centroids
    num_centroids = int(len(non_zero_points) // (centroids_ratio * z_scale))+1
        
    kmeans = KMeans(n_clusters=num_centroids)
    kmeans.fit(non_zero_points)
    return np.array(non_zero_points), kmeans.cluster_centers_


def find_largest_non_zero_slice_z(arr):
    z_max = arr.shape[2]
    for z in range(z_max-1, -1, -1):
        if not np.all(arr[:, :, z] == 0):
            return z
    return -1  
def find_smallest_non_zero_slice_z(arr):
    z_max = arr.shape[2]
    for z in range(z_max):
        if not np.all(arr[:, :, z] == 0):
            return z
    return -1  

def find_non_zero_bounds_per_slice(image):
    bounds = []
    for z in range(image.shape[2]):
        non_zero_points = np.nonzero(image[:,:,z])
        if non_zero_points[0].size != 0:  
            x_min, x_max = np.min(non_zero_points[0]), np.max(non_zero_points[0])
            y_min, y_max = np.min(non_zero_points[1]), np.max(non_zero_points[1])
            bounds.append((x_min, x_max, y_min, y_max))
        else:
            bounds.append(None)
    return bounds

def create_lines_from_centroids(centroids, image_mold, image_cancer, com_image, last_slice, cancer_last_slice, z_scale, num_slices, z_extension=0.1):
    lines = []
    z_max = last_slice
    for centroid in centroids:
        centroid_x, centroid_y, centroid_z = centroid
        slice_z = cancer_last_slice + 3
        x_range, y_range = np.where(com_image[:, :, slice_z] != 0)
        x_min, x_max = np.min(x_range), np.max(x_range)
        y_min, y_max = np.min(y_range), np.max(y_range)
        end_x_range, end_y_range = np.where(image_mold[:, :, z_max] != 0)
        end_x_min, end_x_max = np.min(end_x_range), np.max(end_x_range)
        end_y_min, end_y_max = np.min(end_y_range), np.max(end_y_range)
        relative_x = (centroid_x - x_min) / (x_max - x_min)
        relative_y = (centroid_y - y_min) / (y_max - y_min)
        new_x = end_x_min + (end_x_max - end_x_min) * relative_x
        new_y = end_y_min + (end_y_max - end_y_min) * relative_y
        new_z = last_slice - z_scale * z_extension
        lines.append((centroid, np.array([new_x, new_y, new_z])))
    return lines



def find_orthogonal_vector(vec):
    if vec[0] == 0 and vec[1] == 0:
        if vec[2] == 0:
            raise ValueError("Zero vector")
        return np.array([1, 0, 0])
    return np.array([-vec[1], vec[0], 0])
    
def rotation_matrix_from_axis_angle(axis, angle):

    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    return rotation_matrix


def add_lines_to_mesh(mesh, lines):
    # Create a new mesh to store the original mesh with added line segments
    new_mesh = mesh + o3d.geometry.TriangleMesh()
    
    cylinders = []  # Create a list to store all cylinder objects
    
    for line in lines:
        # Get the start and end points of the line segment
        start_point, end_point = line

        # Calculate direction vector from start to end point
        # direction = -(end_point - start_point)  # Alternative direction calculation
        direction = end_point - start_point
        # Reverse the last dimension directly
        # direction[..., -1] = -direction[..., -1]
        
        # Normalize the direction vector and get segment height
        height = np.linalg.norm(direction)
        direction /= height
        
        # Create cylinder geometry representing the line segment
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=1, height=height)

        # Create rotation matrix from z-axis to target direction
        axis = np.cross([0, 0, 1], direction)
        if np.linalg.norm(axis) > 1e-6:
            # Normalize axis and calculate rotation angle
            axis = axis / np.linalg.norm(axis)
            theta = np.arccos(np.dot([0, 0, 1], direction))
            rot_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * theta)
        else:
            # Handle parallel/anti-parallel cases with z-axis
            rot_matrix = np.eye(3) if np.dot([0, 0, 1], direction) > 0 else -np.eye(3)
        
        # Convert to 4x4 homogeneous transformation matrix
        rot_matrix = np.vstack((rot_matrix, np.zeros(3)))
        rot_matrix = np.column_stack((rot_matrix, np.zeros(4)))
        rot_matrix[-1, -1] = 1
        
        print(rot_matrix)
        print('Rotation matrix:')
        
        # Apply rotation to cylinder
        cylinder = cylinder.transform(rot_matrix)

        # Alternative scaling approach (commented out)
        # scale_matrix = np.eye(4)
        # scale_matrix[0, 0] = 1
        # scale_matrix[1, 1] = 1
        # scale_matrix[2, 2] = height
        # cylinder = cylinder.transform(scale_matrix)

        # Translate cylinder to match line segment start point
        translation_matrix = np.eye(4)
        # Ensure cylinder base aligns with line start point
        translation_matrix[:-1, -1] = start_point
        cylinder = cylinder.transform(translation_matrix)

        # Add the cylinder to the new mesh
        new_mesh += cylinder
        cylinders.append(cylinder)
        
    return new_mesh, cylinders

def process_image(image, z_scale, num_slices, num_bins, centroids_ratio, get_centroid=False):
    t0 = time.time()
    print(f'Current time: {t0}')
    
    # last_slice = find_largest_non_zero_slice_z(image)
    # print('Last slice index:')
    # print(last_slice)
    
    first_slice = find_smallest_non_zero_slice_z(image)
    print('First slice index:')
    print(first_slice)
    
    new_image = interpolate_slices(image, num_slices)
    tc = time.time()
    print(f"Interpolation runtime: {tc-t0} seconds", flush=True)
    
    new_image = bin_and_classify_3d_image(new_image, num_bins)
    
    # Extract non-zero pixels as a point cloud
    non_zero_points, centroid = image_to_point_cloud(new_image, z_scale/num_slices, centroids_ratio)
    t1 = time.time()
    print(f"Runtime up to t1: {t1-t0} seconds", flush=True)
    
    # Create an empty point cloud dataset
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(non_zero_points)
    t2 = time.time()
    print(f"Runtime up to t2: {t2-t1} seconds", flush=True)

    # Skip downsampling
    # -----------
    down_sampled_point_cloud = point_cloud
    t3 = time.time()
    print(f"Runtime up to t3: {t3-t2} seconds", flush=True)
    # -----------
    
    # Generate a 3D mesh from the point cloud data
    distances = down_sampled_point_cloud.compute_nearest_neighbor_distance()
    avg_distance = np.mean(distances)
    radius = 1 * avg_distance
    t4 = time.time()
    print(f"Runtime up to t4: {t4-t3} seconds", flush=True)

    # Estimate normals for the point cloud
    down_sampled_point_cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=40)
    )
    t5 = time.time()
    print(f"Runtime up to t5: {t5-t4} seconds", flush=True)

    # Build a 3D mesh using the Ball Pivoting Algorithm
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        down_sampled_point_cloud,
        o3d.utility.DoubleVector([radius*1.2, radius*2.4])
    )
    t6 = time.time()
    print(f"Runtime up to t6: {t6-t5} seconds", flush=True)

    # Compute and apply mesh normals
    bpa_mesh.compute_vertex_normals()
    vertex_normals = np.asarray(bpa_mesh.vertex_normals)
    bpa_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    t7 = time.time()
    print(f"Runtime up to t7: {t7-t6} seconds", flush=True)

    # Smooth the mesh surface using Laplacian smoothing
    smoothed_mesh = bpa_mesh.filter_smooth_laplacian(number_of_iterations=2, lambda_filter=0.5)
    # Alternative: smoothed_mesh = bpa_mesh.filter_smooth_simple(number_of_iterations=3)

    # Compute and apply normals for the smoothed mesh
    smoothed_mesh.compute_vertex_normals()
    vertex_normals = np.asarray(smoothed_mesh.vertex_normals)
    smoothed_mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
    t8 = time.time()
    print(f"Runtime up to t8: {t8-t7} seconds", flush=True)
    
    print(f"STL generation step completed")
    # Save as STL file
    # o3d.io.write_triangle_mesh(f"output_cancer.stl", smoothed_mesh)

    return smoothed_mesh, centroid, first_slice

def stl_generate(image_mold,
                 image_cancer,
                 z_scale,
                 num_slices,
                 num_bins,
                 centroids_ratio,
                 path,
                 ):
    # Process both images and generate smooth 3D meshes
    try:
        image_cancer
    except NameError:
        print('Starting mold-only processing')
        smoothed_mesh_mold, centroid_mold, last_slice = process_image(image_mold, z_scale, num_slices, num_bins, centroids_ratio)
        
        # Save as STL file
        o3d.io.write_triangle_mesh("output_mold_only.stl", smoothed_mesh_mold)
        print('Mold-only processing completed')
        
    else:
        print('Starting combined processing')
        smoothed_mesh_cancer, centroid_cancer, cancer_last_slice = process_image(image_cancer, z_scale, num_slices, num_bins, centroids_ratio, get_centroid=True)
        smoothed_mesh_mold, centroid_mold, mold_last_slice = process_image(image_mold, z_scale, num_slices, num_bins, centroids_ratio)
        com_image = np.logical_or(image_mold, image_cancer).astype(int)

        ori_lines = create_lines_from_centroids(centroid_cancer, image_mold, image_cancer, com_image, mold_last_slice, cancer_last_slice, z_scale, num_slices)
        lines = ori_lines.copy()

        for i in range(len(lines)):
            # Calculate line direction
            direction = lines[i][0] - lines[i][1]
            # Scale direction to 70% of original length and add to start point
            new_start = lines[i][1] + direction * 0.7
            
            # Update start point
            lines[i] = (new_start, lines[i][1])

        # Combine both meshes into one
        combined_mesh = smoothed_mesh_cancer + smoothed_mesh_mold
        
        mold_mesh_with_lines, lines_mesh_mold = add_lines_to_mesh(smoothed_mesh_mold, lines)

        # Compute vertex normals
        mold_mesh_with_lines.compute_vertex_normals()
        vertex_normals_mold = np.asarray(mold_mesh_with_lines.vertex_normals)
        mold_mesh_with_lines.vertex_normals = o3d.utility.Vector3dVector(vertex_normals_mold)

        # Add lines to the cancer mesh
        cancer_mesh_with_lines, lines_mesh_cancer = add_lines_to_mesh(smoothed_mesh_cancer, lines)
        cancer_mesh_with_lines.compute_vertex_normals()
        vertex_normals_cancer = np.asarray(cancer_mesh_with_lines.vertex_normals)
        cancer_mesh_with_lines.vertex_normals = o3d.utility.Vector3dVector(vertex_normals_cancer)

        # Add lines to the combined mesh
        combined_mesh_with_lines, lines_mesh_combined = add_lines_to_mesh(combined_mesh, lines)
        combined_mesh_with_lines.compute_vertex_normals()
        vertex_normals_combined = np.asarray(combined_mesh_with_lines.vertex_normals)
        combined_mesh_with_lines.vertex_normals = o3d.utility.Vector3dVector(vertex_normals_combined)

        # Save STL files
        o3d.io.write_triangle_mesh(rf"{path}\output_cancer_only.stl", smoothed_mesh_cancer)
        print(rf"{path}\output_cancer_only.stl")
        print('Cancer mesh saved')
        o3d.io.write_triangle_mesh(rf"{path}\output_mold_only.stl", smoothed_mesh_mold)
        print(rf"{path}\output_mold_only.stl")
        print('Mold mesh saved')
        o3d.io.write_triangle_mesh(rf"{path}\output_combined.stl", combined_mesh)
        print(rf"{path}\output_combined.stl")
        print('Combined mesh saved')
        o3d.io.write_triangle_mesh(rf"{path}\output_mold_with_lines.stl", mold_mesh_with_lines)
        print(rf"{path}\output_mold_with_lines.stl")
        print('Mold mesh with lines saved')
        o3d.io.write_triangle_mesh(rf"{path}\output_cancer_with_lines.stl", cancer_mesh_with_lines)
        print(rf"{path}\output_cancer_with_lines.stl")
        print('Cancer mesh with lines saved')
        o3d.io.write_triangle_mesh(rf"{path}\output_combined_with_lines.stl", combined_mesh_with_lines)
        print(rf"{path}\output_combined_with_lines.stl")
        print('Combined mesh with lines saved')

    return cancer_mesh_with_lines, smoothed_mesh_cancer, combined_mesh_with_lines, lines_mesh_combined, lines

def main_genete_app_stl(applicator_path,
                        lines,
                        output_dir,
                        combined_mesh_with_lines,
                        lines_mesh_combined,
                        distance=-25,
                        ):
    # Create a line segment geometry
    another_mesh = o3d.io.read_triangle_mesh(applicator_path)
    # Calculate rotation matrix (angles in radians)
    rot_mat_short_axis = another_mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))  # Rotate 180 degrees around Y-axis
    another_mesh.rotate(rot_mat_short_axis, center=another_mesh.get_center())

    # Get original head center coordinates
    temp_vertices = np.asarray(another_mesh.vertices)

    another_mesh, rot_mat, tran_mat = add_applicator_to_lines(lines, another_mesh)

    # Get axis-aligned bounding box of the point cloud
    lines_points = [point for line in lines for point in line]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(lines_points)
    bbox = point_cloud.get_axis_aligned_bounding_box()

    bbox_center = np.mean(np.concatenate(lines), axis=0)
    bbox_extent = bbox.get_max_bound() - bbox.get_min_bound()

    mesh_center = another_mesh.get_center()
    mesh_extent = another_mesh.get_max_bound() - another_mesh.get_min_bound()

    # Calculate scaling factor and translation vector
    scale_factor = np.linalg.norm(mesh_extent) / np.linalg.norm(bbox_extent) 
    translation_vector = bbox_center - mesh_center

    # Scale and translate the mesh
    another_mesh.scale(scale_factor, center=mesh_center)

    # Translate along the major axis
    obb = another_mesh.get_oriented_bounding_box()
    major_vector = obb.R[:, obb.extent.argmax()]
    translation_vector = major_vector / np.linalg.norm(major_vector) * distance
    another_mesh.translate(translation_vector)

    # Save the adjusted mesh
    another_mesh_with_lines = another_mesh
    another_mesh = combined_mesh_with_lines + another_mesh
    another_mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(rf"{output_dir}/adjusted_mesh.stl", another_mesh)
    
    for item in lines_mesh_combined:
        another_mesh_with_lines += item
    another_mesh_with_lines.compute_vertex_normals()
    o3d.io.write_triangle_mesh(rf"{output_dir}/applicator_with_lines.stl", another_mesh_with_lines)

    return rf"{output_dir}/adjusted_mesh.stl", rf"{output_dir}/applicator_with_lines.stl"

def main_show_image(path):
    
    output_dir = os.path.dirname(path) + '/output'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    image, labels, colors, slices, con_ind = read_ct(path)

    #0820 
    #==========
    app_type = get_app_type(image, labels, con_ind)
    #==========

    show_image_ind = get_middle_elements(labels[0])
    slices, image, labels = scale_images(slices, image, labels, colors)
    image_cancer, image_mold = mask_get(image, labels)
    # plt_output_path_lst = save_plt_rst(image, labels[0], colors, show_image_ind, output_dir)

    return slices, image_mold, image_cancer, output_dir, '', app_type

def main_genete_stl(slices, 
                    image_mold, 
                    image_cancer, 
                    output_dir
                    ):
    z_scale_mapping = {2.5:11, 3: 14, 5: 23}
    
    z_scale = slices[1].ImagePositionPatient[-1] - slices[0].ImagePositionPatient[-1]
    num_slices = z_scale_mapping.get(abs(z_scale), int((z_scale-2.5)*2.5*11 + 11))
    num_bins = 1000
    centroids_ratio = 4e5
    

    cancer_mesh_with_lines, smoothed_mesh_cancer, combined_mesh_with_lines, lines_mesh_combined, lines = stl_generate(image_mold,
                                                                                                                        image_cancer,
                                                                                                                        z_scale, 
                                                                                                                        num_slices, 
                                                                                                                        num_bins, 
                                                                                                                        centroids_ratio, 
                                                                                                                        output_dir)
    return rf"{output_dir}/output_cancer_with_lines.stl", combined_mesh_with_lines, lines_mesh_combined, lines


if __name__ == '__main__':

    for i, arg in enumerate(sys.argv):
        print(f"Argument {i}: {arg}")

    RT_rs = sys.argv[1]
    input_ct_path = sys.argv[2]
    app_db_path = sys.argv[3]
    output_image_path = sys.argv[4]
    output_stl_path = sys.argv[5]

    

    main_slices, main_image_mold, main_image_cancer, main_output_dir, main_plt_output_path_lst, main_app_type = main_show_image(input_ct_path)
    main_app_path = get_app_path(app_db_path, main_app_type)
    main_stlPath1, combined_mesh_with_lines, lines_mesh_combined, main_lines = main_genete_stl(main_slices, main_image_mold, main_image_cancer, output_stl_path,)
    main_appPathmold, main_appPath = main_genete_app_stl(main_app_path, main_lines, output_stl_path, combined_mesh_with_lines, lines_mesh_combined)



