import numpy as np
import tifffile
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import RegularGridInterpolator
import pickle
import torch
import torch.nn.functional as F
from datetime import datetime
import json
from pathlib import Path
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import pdist, squareform
from typing import Literal
import os


class Realign:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.image_path = self.config.get('image_path')
        self.crop_W = self.config.get('crop_W')
        self.crop_H = self.config.get('crop_H')
        self.image = tifffile.imread(self.image_path)
        self.image = np.squeeze(self.image)
        self.save_path = self.config.get('save_path')
        self.start_H = self.config.get('start_H')
        self.start_W = self.config.get('start_W')
        self.block_H = self.config.get('block_H')
        self.block_W = self.config.get('block_W')
        self.step = self.config.get('step')
        self.max_threshold_rate = self.config.get('max_threshold_rate')
        self.min_threshold_rate = self.config.get('min_threshold_rate')
        self.step_threshold_rate = self.config.get('step_threshold_rate')
        self.half_size = self.config.get('half_size')
        self.min_centroids_number = self.config.get('min_centroids_number')
        self.min_point_distance = self.config.get('min_point_distance')
        self.near_points_range = self.config.get('near_points_range')
        self.H, self.W = self.image.shape
        self.centerX = self.W // 2 - 1
        self.centerY = self.H // 2 - 1
        self.new_centerX = 0
        self.new_centerY = 0
        self.gt_coor = []
        self.dist_coor = []
        self.offsetX = []
        self.offsetY = []

    """
    The following functions were written by Teacher LWJ:

    1. distort_model
    2. error_function
    3. generate_params
    4. undistort_coor

    These functions provide a capability: by inputting the corresponding coordinates 
    of several points in the distorted and undistorted images, they can output 
    lens distortion parameters based on the lens distortion formula, allowing 
    the original image to be undistorted.
    """
    def distort_model(self, params, x, y):
        fx, fy, cx, cy, k1, k2, k3, p1, p2 = params
        matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        objpoints = np.concatenate((x[:, np.newaxis], y[:, np.newaxis], np.ones_like(y[:, np.newaxis])), axis=1)
        objpoints_rotated = np.matmul(objpoints, matrix)
        objpoints_projected = objpoints_rotated[:, :2] / (objpoints_rotated[:, 2:] + 1e-17)
        shift = objpoints_projected - np.array([cx, cy])

        x_shifted = shift[:, 0]
        y_shifted = shift[:, 1]
        r2 = x_shifted ** 2 + y_shifted ** 2
        x_distorted = x_shifted * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + 2 * p1 * x_shifted * y_shifted + p2 * (
                r2 + 2 * x_shifted ** 2) + cx
        y_distorted = y_shifted * (1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3) + p1 * (
                r2 + 2 * y_shifted ** 2) + 2 * p2 * x_shifted * y_shifted + cy
        return x_distorted, y_distorted

    def error_function(self, params, x, y, mapx, mapy):
        x_predicted, y_predicted = self.distort_model(params, x, y)
        return np.concatenate((x_predicted - mapx, y_predicted - mapy))

    def generate_params(self, x_distorted, y_distorted, x_ideal, y_ideal):
        params_initial = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        result_undist = least_squares(self.error_function, params_initial,
                                      args=(x_distorted, y_distorted, x_ideal, y_ideal,))
        result_dist = least_squares(self.error_function, params_initial,
                                    args=(x_ideal, y_ideal, x_distorted, y_distorted,))

        return {"undistort": result_undist.x, "inv_undistort": result_dist.x}

    def undistort_coor(self, params):
        H, W = (self.H, self.W)
        gty, gtx = np.mgrid[:H, :W]
        gtxy = np.c_[gtx.ravel(), gty.ravel()]
        x_undistorted, y_undistorted = self.distort_model(params['inv_undistort'], (gtxy[:, 0] - W // 2) / 100,
                                                          (gtxy[:, 1] - H // 2) / 100)
        x_undistorted = x_undistorted * 100 + W // 2
        y_undistorted = y_undistorted * 100 + H // 2
        return x_undistorted, y_undistorted

    def findMinCoord(self, block):
        """
        Find the minimum value and its coordinates in a given block.
        """
        min_value = block.min()
        min_coords = np.unravel_index(np.argmin(block), block.shape)
        return min_value, min_coords

    def findMaxCoord(self, block):
        """
        Find the maximum value and its coordinates in a given block.
        """
        max_value = block.max()
        max_coords = np.unravel_index(np.argmax(block), block.shape)
        return max_value, max_coords

    def generate_center_coor(self, centroids):
        """
        The identified points are the black regions at the corners of the lens,
        which are exactly 7.5 pixels away from the lens center (15/2 = 7.5)
        """
        offsets = np.array([
            [7.5, 7.5],
            [-7.5, 7.5],
            [7.5, -7.5],
            [-7.5, -7.5]
        ])

        # Generate new coordinates for each original coordinate.
        center_coor = []
        for centroid in centroids:
            for offset in offsets:
                new_coord = centroid + offset
                center_coor.append(new_coord)

        return center_coor

    def check_valid_centroids(self, centroids):
        """
        This function checks the validity of identified centroids based on two criteria:
        1. There should be enough centroids to ensure reliable identification.
        2. Adjacent centroids must not be too close to avoid noise interference.
        """
        if len(centroids) < self.min_centroids_number:
            return False, len(centroids)
        else:
            centroids = np.array(centroids)
            distances = pdist(centroids)
            all_distances_greater_than_min = np.all(distances > self.min_point_distance)
            return all_distances_greater_than_min, len(centroids)

    def generate_gt_coor(self):
        """
        generate gt_coor
        One example:
        np.array([[1368,1151],[1368,5156],[1368,9146],[1368,13151],
                  [5373,1151],[5373,5156],[5373,9146],[5373,13151],
                  [9378,1151],[9378,5156],[9378,9146],[9378,13151]])
        """
        center_offset, _ = self.get_offset([0, 0], [self.centerY, self.centerX], self.half_size)

        self.new_centerY = round(center_offset[0] + self.centerY)
        self.new_centerX = round(center_offset[1] + self.centerX)

        self.new_centerY = int(round(self.new_centerY))
        self.new_centerX = int(round(self.new_centerX))

        print(f"new_centerY: {self.new_centerY}, new_centerX: {self.new_centerX}")

        offsetY = self.generate_offset(self.new_centerY, self.start_H, self.block_H)
        offsetX = self.generate_offset(self.new_centerX, self.start_W, self.block_W)
        print(f"offsetY: {offsetY}, offsetX: {offsetX}")
        self.offsetX = np.array(offsetX)
        self.offsetY = np.array(offsetY)

        self.gt_coor = np.array([[self.new_centerY + dy, self.new_centerX + dx] for dy in offsetY for dx in offsetX
                                 ])
        np.save(self.save_path + '/temp/gt_coor.npy', self.gt_coor)
        print(f"gt_coor: {self.gt_coor}")

    def generate_offset(self, center, start_H, block_H):
        """
        Note: This function provides an alternative way to store ground truth coordinates
        (gt_coor).
        """
        diff = center - start_H

        first_element = diff - (diff % 15)

        last_element = -first_element

        block_H_plus_one = block_H + 1
        step = (first_element - last_element) // (block_H_plus_one - 1)
        step = (step // 15) * 15

        offset = [first_element - i * step for i in range(block_H_plus_one)]

        offset = offset[::-1]

        if len(offset) % 2 != 0:
            middle_index = len(offset) // 2
            offset[middle_index] = 0

        return offset

    def deep_generate_dist_coor(self):
        """
        Generates distorted coordinates (dist_coor) by starting from the center
        and progressively identifying micro-lenses in various directions.
        If there is an offset, it is applied in subsequent identifications.
        """

        if len(self.gt_coor) % 2 == 0:
            raise Exception("gt_coor is not a matrix of even length")
        offsets = []
        nx = len(self.offsetX)
        first_offsetX = self.offsetX[:(nx // 2)][::-1]
        last_offsetX = self.offsetX[-(nx // 2):]
        ny = len(self.offsetY)
        first_offsetY = self.offsetY[:(ny // 2) + 1][::-1]
        last_offsetY = self.offsetY[-(ny // 2):]
        self.dist_coor = np.zeros_like(self.gt_coor, dtype=np.float64)

        center_offset, _ = self.get_offset([0, 0], [self.new_centerY, self.new_centerX], self.half_size)
        self.dist_coor[nx // 2 + (ny // 2) * nx] = [self.new_centerY + center_offset[0],
                                                    self.new_centerX + center_offset[1]]

        current_Y = self.new_centerY
        current_X = self.new_centerX
        first_offset = np.array([0, 0])
        for i, negative_offsetY in enumerate(first_offsetY):
            i = i - 1
            if negative_offsetY != 0:
                path = self.generate_path(current_Y, current_X, self.new_centerY + negative_offsetY, self.new_centerX,
                                          'vertical')
                for coordinate in path:
                    first_offset, _ = self.get_offset(first_offset, coordinate, self.half_size)
                self.dist_coor[nx // 2 + (ny // 2 - i - 1) * nx] = [
                    self.new_centerY + negative_offsetY + first_offset[0],
                    self.new_centerX + first_offset[1]]

            current_Y = self.new_centerY + negative_offsetY
            current_X = self.new_centerX
            second_offset = np.array(first_offset)
            for j, negative_offsetX in enumerate(first_offsetX):
                path = self.generate_path(current_Y, current_X, self.new_centerY + negative_offsetY,
                                          self.new_centerX + negative_offsetX, 'horizontal')
                for coordinate in path:
                    second_offset, _ = self.get_offset(second_offset, coordinate, self.half_size)
                self.dist_coor[nx // 2 - (j + 1) + (ny // 2 - i - 1) * nx] = [
                    self.new_centerY + negative_offsetY + second_offset[0],
                    self.new_centerX + negative_offsetX + second_offset[1]]
                current_Y = self.new_centerY + negative_offsetY
                current_X = self.new_centerX + negative_offsetX

            current_Y = self.new_centerY + negative_offsetY
            current_X = self.new_centerX
            second_offset = np.array(first_offset)
            for j, positive_offsetX in enumerate(last_offsetX):
                path = self.generate_path(current_Y, current_X, self.new_centerY + negative_offsetY,
                                          self.new_centerX + positive_offsetX, 'horizontal')
                for coordinate in path:
                    second_offset, _ = self.get_offset(second_offset, coordinate, self.half_size)
                self.dist_coor[nx // 2 + (j + 1) + (ny // 2 - i - 1) * nx] = [
                    self.new_centerY + negative_offsetY + second_offset[0],
                    self.new_centerX + positive_offsetX + second_offset[1]]
                current_Y = self.new_centerY + negative_offsetY
                current_X = self.new_centerX + positive_offsetX

            current_Y = self.new_centerY + negative_offsetY
            current_X = self.new_centerX

        current_Y = self.new_centerY
        current_X = self.new_centerX
        first_offset = np.array([0, 0])
        for i, postive_offsetY in enumerate(last_offsetY):
            path = self.generate_path(current_Y, current_X, self.new_centerY + postive_offsetY, self.new_centerX,
                                      'vertical')
            for coordinate in path:
                first_offset, _ = self.get_offset(first_offset, coordinate, self.half_size)
            self.dist_coor[nx // 2 + (ny // 2 + i + 1) * nx] = [
                self.new_centerY + postive_offsetY + first_offset[0],
                self.new_centerX + first_offset[1]]

            current_Y = self.new_centerY + postive_offsetY
            current_X = self.new_centerX
            second_offset = np.array(first_offset)
            for j, negative_offsetX in enumerate(first_offsetX):
                path = self.generate_path(current_Y, current_X, self.new_centerY + postive_offsetY,
                                          self.new_centerX + negative_offsetX, 'horizontal')
                for coordinate in path:
                    second_offset, _ = self.get_offset(second_offset, coordinate, self.half_size)
                self.dist_coor[nx // 2 - (j + 1) + (ny // 2 + i + 1) * nx] = [
                    self.new_centerY + postive_offsetY + second_offset[0],
                    self.new_centerX + negative_offsetX + second_offset[1]]
                current_Y = self.new_centerY + postive_offsetY
                current_X = self.new_centerX + negative_offsetX

            current_Y = self.new_centerY + postive_offsetY
            current_X = self.new_centerX
            second_offset = np.array(first_offset)
            for j, positive_offsetX in enumerate(last_offsetX):
                path = self.generate_path(current_Y, current_X, self.new_centerY + postive_offsetY,
                                          self.new_centerX + positive_offsetX, 'horizontal')
                for coordinate in path:
                    second_offset, _ = self.get_offset(second_offset, coordinate, self.half_size)
                self.dist_coor[nx // 2 + (j + 1) + (ny // 2 + i + 1) * nx] = [
                    self.new_centerY + postive_offsetY + second_offset[0],
                    self.new_centerX + positive_offsetX + second_offset[1]]
                current_Y = self.new_centerY + postive_offsetY
                current_X = self.new_centerX + positive_offsetX

            current_Y = self.new_centerY + postive_offsetY
            current_X = self.new_centerX

        print(f"dist_coor: {self.dist_coor}")
        offsets = self.dist_coor - self.gt_coor
        print(f"offsets: {offsets}")

        os.makedirs(os.path.dirname(self.save_path + '/temp/dist_coor.npy'), exist_ok=True)
        os.makedirs(os.path.dirname(self.save_path + '/temp/offsets.py'), exist_ok=True)
        np.save(self.save_path + '/temp/dist_coor.npy', self.dist_coor)
        np.save(self.save_path + '/temp/offsets.npy', offsets)

    def generate_path(self, current_Y, current_X, next_Y, next_X, type: Literal['vertical', 'horizontal']):
        """
        Generates a series of coordinates between the start point (current_Y, current_X)
        and the end point (next_Y, next_X) based on the specified direction (type).
        If type is 'vertical', it moves along the height; if 'horizontal', it moves along the width.
        """
        coordinates = []

        if type == 'vertical':
            start_Y = current_Y
            offset = next_Y - current_Y
            stop_Y = current_Y + offset
            step = self.step if offset > 0 else -self.step
            Y_values = np.arange(start_Y, stop_Y, step)
            if Y_values[-1] != stop_Y:
                Y_values = np.append(Y_values, stop_Y)
            X_values = np.full(Y_values.shape, current_X)
            coordinates = np.column_stack((Y_values, X_values))
            coordinates = coordinates[1:]
            return coordinates

        elif type == 'horizontal':
            start_X = current_X
            offset = next_X - current_X
            stop_X = current_X + offset
            step = self.step if offset > 0 else -self.step
            X_values = np.arange(start_X, stop_X, step)
            if X_values[-1] != stop_X:
                X_values = np.append(X_values, stop_X)
            Y_values = np.full(X_values.shape, current_Y)
            coordinates = np.column_stack((Y_values, X_values))
            return coordinates

    def get_offset(self, former_offset, gt, half_size=30):
        """
        Retrieves the offset from the given coordinates by identifying distinct black regions
        (microlens corners) in the image.

        The process involves:
        1. Identifying independent black regions using a threshold that increases to find
           larger connected areas.
        2. Calculating the lens center based on valid centroid detections.

        Note: Generally, only the first half of the code is used. If it fails to find valid
        regions, a dilation operation is applied to mitigate noise, and the process is repeated.
        """
        former_offset = np.array(former_offset).astype(int)
        gt = np.array(gt).astype(int)
        block = self.image[gt[0] - half_size + former_offset[0]:gt[0] + half_size + 1 + former_offset[0],
                gt[1] - half_size + former_offset[1]:gt[1] + half_size + 1 + former_offset[1]]

        height, width = block.shape
        center = [height // 2, width // 2]
        max_value, max_coords = self.findMaxCoord(block)
        max_threshold = max_value * self.max_threshold_rate
        min_threshold = max_value * self.min_threshold_rate
        step = max_value * self.step_threshold_rate
        thresholds = np.arange(min_threshold, max_threshold, step)

        former_length = 0
        former_valid = False
        former_centroids = []
        for threshold in thresholds:
            high_value_regions = block < threshold

            # Connected region analysis, marking each connected high-value block
            labeled_array, num_features = label(high_value_regions)

            # Create boundary mask
            boundary_mask = np.zeros_like(block, dtype=bool)
            boundary_mask[0, :] = True
            boundary_mask[-1, :] = True
            boundary_mask[:, 0] = True
            boundary_mask[:, -1] = True

            # Determine connected regions that intersect with the boundary
            regions_to_exclude = np.unique(labeled_array[boundary_mask])
            regions_to_exclude = regions_to_exclude[regions_to_exclude > 0]  # 0 is the background, no need to exclude

            # Exclude the labeled regions at the boundary
            valid_regions_mask = np.isin(labeled_array, regions_to_exclude, invert=True)

            # Update connected region analysis results
            valid_labeled_array, valid_num_features = label(np.logical_and(high_value_regions, valid_regions_mask))

            # Calculate the centroid of each valid connected region
            centroids = center_of_mass(block, valid_labeled_array, range(1, valid_num_features + 1))
            valid, length = self.check_valid_centroids(centroids)
            if valid == False and former_valid == True:
                break
            if valid == True and length < former_length and former_valid == True:
                break
            former_centroids = centroids
            former_length = length
            former_valid = valid

        if former_valid:
            center_coor = self.generate_center_coor(former_centroids)

            adjusted_coords = []

            # Calculate offsets relative to the center point
            for coord in center_coor:
                adjusted_coord = [coord[0] - center[0], coord[1] - center[1]]
                adjusted_coords.append(adjusted_coord)

            # Calculate the coordinates closest to the origin
            distances = np.linalg.norm(adjusted_coords, axis=1)
            min_index = np.argmin(distances)
            closest_coord = adjusted_coords[min_index]

            # Use the closest coordinate as the center and calculate the average of nearby coordinates
            adjusted_coords = np.array(adjusted_coords)
            closest_coord = np.array(closest_coord)
            distances_to_closest = np.linalg.norm(adjusted_coords - closest_coord, axis=1)
            close_points = adjusted_coords[distances_to_closest <= self.near_points_range]
            final_offset = np.mean(close_points, axis=0) + former_offset
            final_offset = final_offset
            return final_offset, np.array(former_centroids) - np.array(center)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        block = cv2.dilate(block, kernel, iterations=1)

        height, width = block.shape
        center = [height // 2, width // 2]

        former_length = 0
        former_valid = False
        former_centroids = []
        for threshold in thresholds:
            high_value_regions = block < threshold

            # Connected region analysis, marking each connected high-value block
            labeled_array, num_features = label(high_value_regions)

            # Create boundary mask
            boundary_mask = np.zeros_like(block, dtype=bool)
            boundary_mask[0, :] = True
            boundary_mask[-1, :] = True
            boundary_mask[:, 0] = True
            boundary_mask[:, -1] = True

            # Determine connected regions that intersect with the boundary
            regions_to_exclude = np.unique(labeled_array[boundary_mask])
            regions_to_exclude = regions_to_exclude[regions_to_exclude > 0]  # 0 is the background, no need to exclude

            # Exclude the labeled regions at the boundary
            valid_regions_mask = np.isin(labeled_array, regions_to_exclude, invert=True)

            # Update connected region analysis results
            valid_labeled_array, valid_num_features = label(np.logical_and(high_value_regions, valid_regions_mask))

            # Calculate the centroid of each valid connected region
            centroids = center_of_mass(block, valid_labeled_array, range(1, valid_num_features + 1))
            center_coor = []
            valid, length = self.check_valid_centroids(centroids)
            if valid == False and former_valid == True:
                break
            if valid == True and length < former_length and former_valid == True:
                break
            former_centroids = centroids
            former_length = length
            former_valid = valid

        if former_valid:
            center_coor = self.generate_center_coor(former_centroids)

            adjusted_coords = []

            # Calculate offsets relative to the center point
            for coord in center_coor:
                adjusted_coord = [coord[0] - center[0], coord[1] - center[1]]
                adjusted_coords.append(adjusted_coord)

            # Calculate the coordinates closest to the origin
            distances = np.linalg.norm(adjusted_coords, axis=1)
            min_index = np.argmin(distances)
            closest_coord = adjusted_coords[min_index]

            # Use the closest coordinate as the center and calculate the average of nearby coordinates
            adjusted_coords = np.array(adjusted_coords)
            closest_coord = np.array(closest_coord)
            distances_to_closest = np.linalg.norm(adjusted_coords - closest_coord, axis=1)
            close_points = adjusted_coords[distances_to_closest <= self.near_points_range]
            final_offset = np.mean(close_points, axis=0) + former_offset
            final_offset = final_offset
            return final_offset, np.array(former_centroids) - np.array(center)

        raise Exception(
            f"Unrecognized microlens, center coordinates {gt}, valid {former_valid}, centroids {former_centroids}, length {former_length}")

    def main_function(self):
        self.generate_gt_coor()
        self.deep_generate_dist_coor()
        x_ideal = (self.gt_coor[:, 1] - self.new_centerX) / 100
        y_ideal = (self.gt_coor[:, 0] - self.new_centerY) / 100

        x_distorted = (self.dist_coor[:, 1] - self.centerX) / 100
        y_distorted = (self.dist_coor[:, 0] - self.centerY) / 100

        params_new = self.generate_params(x_distorted, y_distorted, x_ideal, y_ideal)
        x_undistorted, y_undistorted = self.undistort_coor(params_new)

        x_undistorted = x_undistorted.reshape(self.H, self.W)
        y_undistorted = y_undistorted.reshape(self.H, self.W)
        x_undistorted = x_undistorted / self.W * 2 - 1
        y_undistorted = y_undistorted / self.H * 2 - 1
        grid = torch.stack(
            [torch.from_numpy(x_undistorted.astype(np.float32)), torch.from_numpy(y_undistorted.astype(np.float32))],
            dim=2).unsqueeze(0)  # 1,h,w,2
        image_undistort = F.grid_sample(torch.from_numpy(self.image.astype(np.float32)).unsqueeze(0).unsqueeze(0), grid,
                                        mode='bilinear', padding_mode='zeros', align_corners=False)
        print(f'image_undistort.shape: {image_undistort.shape}')

        output_file = self.save_path + "/image/result/undistort_" + os.path.basename(self.image_path)
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)
        tifffile.imwrite(output_file, image_undistort.numpy().astype(np.uint16))
        with open(self.save_path + "/param/undistort_params_dict_points_" + datetime.now().strftime("%y%m%d") + ".pkl",
                  'wb') as f:
            pickle.dump(params_new, f)