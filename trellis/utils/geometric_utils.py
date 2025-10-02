import numpy as np


def quaternion_to_matrix(quaternion: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices

    Args:
        quaternion (np.ndarray): shape (..., 4), the quaternions to convert

    Returns:
        np.ndarray: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(
        min=eps
    )
    w, x, y, z = (
        quaternion[..., 0],
        quaternion[..., 1],
        quaternion[..., 2],
        quaternion[..., 3],
    )
    zeros = np.zeros_like(w)
    I = np.eye(3, dtype=quaternion.dtype)
    xyz = quaternion[..., 1:]
    A = (
        xyz[..., :, None] * xyz[..., None, :]
        - I * (xyz**2).sum(axis=-1)[..., None, None]
    )
    B = np.stack([zeros, -z, y, z, zeros, -x, -y, x, zeros], axis=-1).reshape(
        *quaternion.shape[:-1], 3, 3
    )
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def matrix_to_quaternion(rot_mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    Args:
        rot_mat (np.ndarray): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        np.ndarray: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = [
        rot_mat[..., i, j] for i in range(3) for j in range(3)
    ]

    diag = np.diagonal(rot_mat, axis1=-2, axis2=-1)
    M = np.array(
        [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=rot_mat.dtype
    )
    wxyz = 0.5 * np.clip(1 + diag @ M.T, 0.0, None) ** 0.5
    max_idx = np.argmax(wxyz, axis=-1)
    xw = np.sign(m21 - m12)
    yw = np.sign(m02 - m20)
    zw = np.sign(m10 - m01)
    yz = np.sign(m21 + m12)
    xz = np.sign(m02 + m20)
    xy = np.sign(m01 + m10)
    ones = np.ones_like(xw)
    sign = np.where(
        max_idx[..., None] == 0,
        np.stack([ones, xw, yw, zw], axis=-1),
        np.where(
            max_idx[..., None] == 1,
            np.stack([xw, ones, xy, xz], axis=-1),
            np.where(
                max_idx[..., None] == 2,
                np.stack([yw, xy, ones, yz], axis=-1),
                np.stack([zw, xz, yz, ones], axis=-1),
            ),
        ),
    )
    quat = sign * wxyz
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=eps)
    return quat
