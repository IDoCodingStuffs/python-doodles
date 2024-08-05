import open3d as o3d
import pydicom
import numpy as np
import glob
import os
import pgzip
from tqdm.contrib.concurrent import process_map


def read_series_as_pcd(dir_path):
    pcds_xyz = []
    pcds_d = []

    paths = glob.glob(os.path.join(dir_path, "*.dcm"))
    for path in paths:
        dicom_slice = pydicom.read_file(path)
        img = np.expand_dims(dicom_slice.pixel_array, -1)
        pcd = o3d.geometry.PointCloud()
        x, y, z = np.where(img)

        index_voxel = np.vstack((x, y, z))
        grid_index_array = index_voxel.T
        pcd.points = o3d.utility.Vector3dVector(grid_index_array)

        vals = np.array([img[x, y, z] for x, y, z in grid_index_array])

        dX, dY = dicom_slice.PixelSpacing
        X = np.array(list(dicom_slice.ImageOrientationPatient[:3]) + [0]) * dX
        Y = np.array(list(dicom_slice.ImageOrientationPatient[3:]) + [0]) * dY
        S = np.array(list(dicom_slice.ImagePositionPatient) + [1])

        transform_matrix = np.array([X, Y, np.zeros(len(X)), S]).T
        transformed_pcd = pcd.transform(transform_matrix)

        pcds_xyz.extend(transformed_pcd.points)
        pcds_d.extend(vals)

        pcd.clear()
        transformed_pcd.clear()

    pcd_xyzd = np.hstack((pcds_xyz, np.expand_dims(pcds_d, -1)))

    return pcd_xyzd


def read_series_as_voxel_grid(dir_path):
    cache_path = os.path.join(dir_path, "cached_grid.npy.gz")
    f = None
    if os.path.exists(cache_path):
        try:
            f = pgzip.PgzipFile(cache_path, "r")
            ret = np.load(f)
            f.close()
            return ret
        except Exception as e:
            print(dir_path, "\n", e)
            if f:
                f.close()
            os.remove(cache_path)

    pcd_xyzd = read_series_as_pcd(dir_path)

    pcd_overall = o3d.geometry.PointCloud()
    pcd_overall.points = o3d.utility.Vector3dVector(pcd_xyzd[:, :3])

    paths = glob.glob(os.path.join(dir_path, "*.dcm"))
    dicom_slice = pydicom.read_file(paths[0])
    dX, dY = dicom_slice.PixelSpacing

    voxel_grid = o3d.geometry.VoxelGrid().create_from_point_cloud(pcd_overall, dX)

    coords = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    vals = pcd_xyzd[:, 3]

    pcd_overall.clear()
    voxel_grid.clear()

    size = np.max(coords, axis=0) + 1
    grid = np.zeros((size[1], size[0], size[2]))

    for index, coord in enumerate(coords):
        grid[(coord[1], coord[0], coord[2])] = vals[index]

    f = pgzip.PgzipFile(cache_path, "w")
    np.save(f, grid)
    f.close()

    del pcd_overall
    del voxel_grid

    return grid

if __name__ == "__main__":
    dirs = glob.glob("./data/rsna-2024-lumbar-spine-degenerative-classification/train_images/*/*/")
    process_map(read_series_as_voxel_grid, dirs, chunksize=2, max_workers=4)
