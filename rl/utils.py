# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Changyu Hu

# Commons Clause addition:
# This software is provided for non-commercial use only. See LICENSE file for details.

from scipy.sparse import csr_array
import pathlib
import struct
import shutil
import numpy as np

np_integer, np_real = np.int32, np.float32


def to_real_array(val):
    return np.array(val, dtype=np_real).copy()


def to_integer_array(val):
    return np.array(val, dtype=np_integer).copy()


def eulerAnglesToRotationMatrix3D(theta):
    """
    Convert euler angles to rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def create_folder(folder_name, exist_ok):
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=exist_ok)


def delete_folder(folder_name):
    shutil.rmtree(folder_name)


def delete_file(file_name):
    pathlib.Path(file_name).unlink()


def file_exist(file_name):
    return pathlib.Path(file_name).is_file()


# Load Eigen matrices and vectors.


def load_real_vector(file_name):
    with open(file_name, "rb") as f:
        content = f.read()
        # The first 8 bytes are the vec size.
        num = np_integer(struct.unpack("=q", content[:8])[0])
        data = struct.unpack("={:d}d".format(num), content[8:])
        return to_real_array(data).ravel()


def sparse_matrix_to_triplets(mat):
    row_num, col_num = mat.shape
    row_num = np_integer(row_num)
    col_num = np_integer(col_num)
    nonzeros_num = mat.nnz
    nonzeros_num = np_integer(nonzeros_num)

    triplets = []
    for r in range(row_num):
        cols = mat.indices[mat.indptr[r]:mat.indptr[r + 1]]
        data = mat.data[mat.indptr[r]:mat.indptr[r + 1]]
        for c, v in zip(cols, data):
            # Triplet (r, c, v).
            triplets.append((np_integer(r), np_integer(c), np_real(v)))
    return triplets


def triplets_to_sparse_matrix(row_num, col_num, triplets):
    row_num = np_integer(row_num)
    col_num = np_integer(col_num)
    row_idx = []
    col_idx = []
    data = []
    for r, c, v in triplets:
        row_idx.append(r)
        col_idx.append(c)
        data.append(v)
    row_idx = to_integer_array(row_idx)
    col_idx = to_integer_array(col_idx)
    data = to_real_array(data)
    # CSR matrix format.
    return csr_array((data, (row_idx, col_idx)), (row_num, col_num))


def load_real_sparse_matrix(file_name):
    with open(file_name, "rb") as f:
        content = f.read()
        # Row and col numbers.
        row_num = np_integer(struct.unpack("=q", content[:8])[0])
        col_num = np_integer(struct.unpack("=q", content[8:16])[0])
        nonzeros_num = np_integer(struct.unpack("=q", content[16:24])[0])
        row_idx = []
        col_idx = []
        data = []
        byte_cnt = 24
        for _ in range(nonzeros_num):
            r = np_integer(struct.unpack(
                "=q", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            c = np_integer(struct.unpack(
                "=q", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            v = np_real(struct.unpack("=d", content[byte_cnt:byte_cnt + 8])[0])
            byte_cnt += 8
            row_idx.append(r)
            col_idx.append(c)
            data.append(v)
        row_idx = to_integer_array(row_idx)
        col_idx = to_integer_array(col_idx)
        data = to_real_array(data)
        # CSR matrix format.
        return csr_array((data, (row_idx, col_idx)), (row_num, col_num))


def save_real_sparse_matrix(file_name, mat):
    triplets = sparse_matrix_to_triplets(mat)
    with open(file_name, "wb") as f:
        # Write row and col numbers.
        row_num, col_num = mat.shape
        row_num = np_integer(row_num)
        col_num = np_integer(col_num)
        nonzeros_num = mat.nnz
        nonzeros_num = np_integer(nonzeros_num)
        f.write(struct.pack("=q", row_num))
        f.write(struct.pack("=q", col_num))
        f.write(struct.pack("=q", nonzeros_num))

        for r, c, v in triplets:
            f.write(struct.pack("=q", np_integer(r)))
            f.write(struct.pack("=q", np_integer(c)))
            f.write(struct.pack("=d", np_real(v)))
