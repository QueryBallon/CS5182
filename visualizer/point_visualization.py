import os
import numpy as np
import plotly.express as px
import argparse
import random

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Plot 3D Point Cloud')
    parser.add_argument('--path', type=str, required=True, help='path')
    parser.add_argument('--num_point', type=int, default=2048, help='point Number')

    return parser.parse_args()

def plot_3d_shape(shape):
    x = shape[:, 0]
    y = shape[:, 1]
    z = shape[:, 2]
    fig = px.scatter_3d(x=x, y=y, z=z, opacity=0.3)
    fig.show()

def point_cloud_noise(data, mean=0, std_dev=None):
    if std_dev is None:
        std_dev = random.uniform(0.005, 0.01)

    # noise for x, y, and z coordinates
    noise_xyz = np.random.normal(mean, std_dev, size=data[:, :3].shape)

    # noise to the original x, y, and z coordinates
    noisy_data = np.column_stack((data[:, :3] + noise_xyz, data[:, 3:]))

    return noisy_data

def point_cloud_transform(data, rotate=(0, 0, 0), scale=(1, 1, 1), shift=(0, 0, 0)):
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]

        points = np.column_stack((x, y, z, np.ones(len(x))))

        # rotation angles (in radians)
        r_x = np.radians(rotate[0])
        r_y = np.radians(rotate[1])
        r_z = np.radians(rotate[2])

        # rotation matrices along X Y Z
        r_x_matrix = np.array([ [1, 0, 0, 0],
                                [0, np.cos(r_x), -np.sin(r_x), 0],
                                [0, np.sin(r_x), np.cos(r_x), 0],
                                [0, 0, 0, 1]])

        r_y_matrix = np.array([ [np.cos(r_y), 0, np.sin(r_y), 0],
                                [0, 1, 0, 0],
                                [-np.sin(r_y), 0, np.cos(r_y), 0],
                                [0, 0, 0, 1]])

        r_z_matrix = np.array([ [np.cos(r_z), -np.sin(r_z), 0, 0],
                                [np.sin(r_z), np.cos(r_z), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

        # rotation
        combined_rotation_matrix = np.dot(r_x_matrix, np.dot(r_y_matrix, r_z_matrix))
        rotated_points = np.dot(points, combined_rotation_matrix.T)

        # scaling
        scaled_points = rotated_points * np.array([scale[0], scale[1], scale[2], 1])

        # transition
        shifted_points = scaled_points + np.array([shift[0], shift[1], shift[2], 0])

        # cartesian coordinates
        fx = shifted_points[:, 0] / shifted_points[:, 3]
        fy = shifted_points[:, 1] / shifted_points[:, 3]
        fz = shifted_points[:, 2] / shifted_points[:, 3]

        return np.column_stack((fx, fy, fz, data[:, 3:]))


def main(args):
    paths = []
    if os.path.isdir(args.path):
        for pts in os.listdir(args.path):
            paths.append(os.path.join(args.path, pts))
    else:
        paths.append(args.path)

    np.random.shuffle(paths)
    for path in paths[:3]:
        data = np.genfromtxt(path, delimiter=' ')

        new_data = point_cloud_transform(data, (0, 0, 0), (1, 1, 1), (0, 0, 0))
        # new_data = point_cloud_noise(new_data)
        # new_data = rotate_points(data, (-90, 0, -22), (1, 1, 1), (-0.1, -0.4, -0.1))
        plot_3d_shape(new_data[:args.num_point])

        # np.savetxt('./l_transformed_data.txt', new_data, fmt='%f')


if __name__ == '__main__':
    args = parse_args()
    main(args)


