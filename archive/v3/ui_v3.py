from __future__ import division
# import tkinter as tk
import os, sys, glob
import numpy as np
import pydicom as dicom
# import dicom
from skimage.draw import polygon
# from skimage.transform import resize
# import h5py
# from constants import *
# from utils import *
np.set_printoptions(threshold = 513)
# from stl import mesh
import time
import open3d as o3d
from scipy.ndimage import zoom
from sklearn.cluster import KMeans
# from collections import Counter
import matplotlib.pyplot as plt
# from pathlib import Path
# # from pyqtgraph.Qt import QtCore, QtGui
# from pyqtgraph.Qt import QtGui
# # import pyqtgraph as pg
# import pyqtgraph.opengl as gl

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  



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
        # contour['name'] = structure.StructureSetROISequence[i].ROIName
        # assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
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

def read_ct(path):
    dcms = glob.glob(os.path.join(path, '*.dcm'))
    dcms_ct = [x for x in dcms if 'rtss' not in x]
    dcms_rs = [x for x in dcms if 'rtss' in x]

    # dcms_ct = [x for x in dcms if 'RS' not in x]
    # dcms_rs = [x for x in dcms if 'RS' in x]

    # rs
    structure = dicom.read_file(dcms_rs[0])
    contours = read_structure(structure)

    # ct
    slices = [dicom.read_file(dcm) for dcm in dcms_ct]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    image = np.stack([s.pixel_array for s in slices], axis=-1)
    labels, colors = get_mask(contours, slices)

    return image, labels, colors, slices

def scale_images(slices,
                 image,
                 labels):
    
    scale_factors = [slices[0].PixelSpacing[0], slices[0].PixelSpacing[1], 1]

    image = zoom(image, scale_factors, order=0)
    for i in range(len(labels)):
        labels[i] = zoom(labels[i], scale_factors, order=0)

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

def main(path):
    
    output_dir = os.path.dirname(path) + '/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    image, labels, colors, slices = read_ct(path)
    show_image_ind = get_middle_elements(labels[0])
    slices, image, labels = scale_images(slices, image, labels)
    image_cancer, image_mold = mask_get(image, labels)
    plt_output_path_lst = save_plt_rst(image, labels[0], colors, show_image_ind, output_dir)
    # if image_cancer == None:
    #     return 'Invalid input: No Cancer ploted!'

    z_scale_mapping = {2.5:11, 3: 14, 5: 23}
    
    z_scale = slices[1].ImagePositionPatient[-1] - slices[0].ImagePositionPatient[-1]
    # interpolate_epochs = 3
    # num_slices = 11 
    num_slices = z_scale_mapping.get(abs(z_scale), int((z_scale-2.5)*2.5*11 + 11))
    num_bins = 1000
    centroids_ratio = 4e5

    cancer_mesh_with_lines, smoothed_mesh_cancer = stl_generate(image_mold,
                                                                image_cancer,
                                                                z_scale, 
                                                                num_slices, 
                                                                num_bins, 
                                                                centroids_ratio, 
                                                                output_dir)
    return rf"{output_dir}/output_cancer_with_lines.stl", plt_output_path_lst



from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *  
import numpy as np
from stl import mesh
from pathlib import Path
        
class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.setGeometry(0, 0, 1600, 1000)  # Adjust window size as needed
        self.setAcceptDrops(True)
        
        self.initUI()
        
    

        self.currentSTL = [None, None]
        self.lastDir = None
        self.droppedFilename = None

    def initUI(self):
        centerWidget = QWidget()
        self.setCentralWidget(centerWidget)

        layout = QHBoxLayout()
        centerWidget.setLayout(layout)

        # Create the buttons layout
        buttonsLayout = QVBoxLayout()
        layout.addLayout(buttonsLayout)


        
        # Add "Import CT Path" button
        btnImportCT = QPushButton(text="Import CT Path")
        btnImportCT.setMaximumWidth(150)  # Set maximum width
        btnImportCT.setFixedHeight(90)  # Set fixed height
        btnImportCT.clicked.connect(self.importCTPath)
        buttonsLayout.addWidget(btnImportCT)

        # Add "Generate and Preview CT" button
        btnGenerateCT = QPushButton(text="Generate and Preview")
        btnGenerateCT.setMaximumWidth(150)  # Set maximum width
        btnGenerateCT.setFixedHeight(90)  # Set fixed height
        btnGenerateCT.clicked.connect(self.generateAndPreviewCT)
        buttonsLayout.addWidget(btnGenerateCT)

        # Add text field for path
        self.ctPathEdit = QLineEdit()
        self.ctPathEdit.setFixedWidth(150)
        buttonsLayout.addWidget(self.ctPathEdit)

        # Add "Dose" button
        btnDose = QPushButton(text="Dose")
        btnDose.setMaximumWidth(150)  # Set maximum width
        btnDose.setFixedHeight(90)  # Set fixed height
        # Add your function here
        buttonsLayout.addWidget(btnDose)

        # Add "3D Print" button
        btn3DPrint = QPushButton(text="3D Print")
        btn3DPrint.setMaximumWidth(150)  # Set maximum width
        btn3DPrint.setFixedHeight(90)  # Set fixed height
        # Add your function here
        buttonsLayout.addWidget(btn3DPrint)

        # Add stretch to push elements to top
        buttonsLayout.addStretch()

        # Create the view windows layout
        viewLayout = QHBoxLayout()
        layout.addLayout(viewLayout)
        layout.setStretch(1, 1)  # Set stretch factor of viewLayout to be more than buttonsLayout

        # Add view windows
        self.viewer = [gl.GLViewWidget()]#, gl.GLViewWidget()]
        for v in self.viewer:
            viewLayout.addWidget(v, 1)
            v.setWindowTitle('STL Viewer')
            v.setCameraPosition(distance=40)

            g = gl.GLGridItem()
            g.setSize(200, 200)
            g.setSpacing(5, 5)
            v.addItem(g)

        # Add view window for images
        self.imageLayout = QGridLayout()
        viewLayout.addLayout(self.imageLayout)
        self.images = [QGraphicsView(), QGraphicsView(), QGraphicsView()]  # Create QGraphicsView objects
        for i, img_view in enumerate(self.images):
            img_view.setScene(QGraphicsScene())  # Set a new QGraphicsScene
            self.imageLayout.addWidget(img_view, i // 1, i % 1)  # Arrange images in a 3x1 grid

        viewLayout.setStretch(0, 1)  # set stretch factor for GLViewWidget
        viewLayout.setStretch(1, 1)  # set stretch factor for imageLayout

    # Other functions...
    # ...

    def importCTPath(self):
        directory = str(Path.home())  # default to user's home directory
        if self.lastDir:
            directory = self.lastDir
        foldername = QFileDialog.getExistingDirectory(self, "Open folder", directory)
        if foldername:
            self.ctPathEdit.setText(foldername)
            self.lastDir = foldername
            print(foldername)
            # image, labels, colors, slices = read_ct(foldername)


    def generateAndPreviewCT(self):
        ctPath = self.ctPathEdit.text()
        print(f"Processing CT file at {ctPath}")

        stlPath1, stlPath2 = main(self.lastDir)
        # Add your own function here to process the CT file, generate STL and preview it
        self.showSTL(stlPath1, 0)
        self.showImage(stlPath2)

    def showDialog(self, viewer_index):
        directory = Path("")
        if self.lastDir:
            directory = self.lastDir
        fname = QFileDialog.getOpenFileName(self, "Open file", str(directory), "STL (*.stl)")
        if fname[0]:
            self.showSTL(fname[0], viewer_index)
            self.lastDir = Path(fname[0]).parent
            
    def showSTL(self, filename, viewer_index):
        if self.currentSTL[viewer_index]:
            self.viewer[viewer_index].removeItem(self.currentSTL[viewer_index])

        points, faces = self.loadSTL(filename)
        meshdata = gl.MeshData(vertexes=points, faces=faces)
        mesh_item = gl.GLMeshItem(meshdata=meshdata, smooth=True, drawFaces=False, drawEdges=True, edgeColor=(0, 1, 0, 1))
        self.viewer[viewer_index].addItem(mesh_item)
        
        self.currentSTL[viewer_index] = mesh_item

    def showImage(self, imagePaths):
        if len(imagePaths) != len(self.images):
            print(f"Number of image paths {len(imagePaths)} does not match number of image views {len(self.images)}")
            return

        for i in range(len(imagePaths)):
            imagePath = imagePaths[i]
            image = QPixmap(imagePath)
            self.images[i].scene().clear()
            self.images[i].scene().addPixmap(image)
            self.images[i].show()

    def loadSTL(self, filename):
        m = mesh.Mesh.from_file(filename)
        shape = m.points.shape
        points = m.points.reshape(-1, 3)
        center = np.mean(points, axis=0)
        points = points - center
        faces = np.arange(points.shape[0]).reshape(-1, 3)
        return points, faces

if __name__ == '__main__':
    app = QtGui.QApplication([])
    window = MyWindow()
    window.show()
    app.exec_()

