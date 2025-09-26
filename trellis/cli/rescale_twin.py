import pipelime.commands.interfaces as plint
import pydantic.v1 as pyd
from pipelime.commands.piper import PipelimeCommand
from pipelime.piper import PiperPortType
import torch

from dataclasses import dataclass
from typing import Optional, cast, Sequence

import numpy as np
import torch as tr
from skimage import measure
from torch import Tensor
from trimesh import Trimesh


def get_meshgrid(res: Tensor) -> Tensor:
    """Create a 3D meshgrid.

    Args:
        res: resolution of the meshgrid.

    Raises:
        ValueError: If input tensor is not a 3D tensor.

    Returns:
        3D meshgrid.
    """
    if not res.dim() == 1 or not res.shape[0] == 3:
        raise ValueError("Input tensor must have shape (3).")

    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0].item(), dtype=torch.long),
                torch.arange(res[1].item(), dtype=torch.long),
                torch.arange(res[2].item(), dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    )


from pipelime.items import Item


@dataclass
class Intrinsics:
    """Data class representing intrinsic parameters of a pinhole camera model."""

    K: Tensor
    """Camera matrix as a 3x3 tensor with all values in pixels."""
    dist_coeffs: Tensor
    """Distortion coefficients as a 5-element tensor."""
    height: int
    """Image height in pixels."""
    width: int
    """Image width in pixels."""

    @classmethod
    def from_item(cls, item: Item) -> "Intrinsics":
        """Create Intrinsics from a sample dictionary.

        Args:
            sample: Sample dictionary to read from.
        """
        camera = item()
        camera_matrix = torch.tensor(camera["intrinsics"]["camera_matrix"])
        w, h = camera["intrinsics"]["image_size"]
        dist_coeffs = camera["intrinsics"]["dist_coeffs"]
        return cls(
            K=torch.tensor(camera_matrix),
            dist_coeffs=torch.tensor(dist_coeffs),
            height=h,
            width=w,
        )


@dataclass
class VoxelCarvingMesherInput:
    """Mesher Input data."""

    masks: Tensor
    """Masks."""
    extrinsics: Tensor
    """World to camera transformation."""
    intrinsics: Sequence[Intrinsics]
    """Camera intrinsics."""

    @classmethod
    def from_underfolder(
        cls,
        samples,
    ) -> "VoxelCarvingMesherInput":
        """Create a VoxelCarvingMesherInput object from underfolder.

        Args:
            samples: Samples from the underfolder.
            items_keys: Items keys used to read mesh input data.
            track_fn: Function to track progress.
        """
        masks, w_Ts_c, intrinsics = [], [], []
        for i in range(len(samples)):
            masks.append(tr.from_numpy(samples[i]["mask"]()))
            w_Ts_c.append(tr.from_numpy(samples[i]["w2c"]()))
            intrinsics.append(Intrinsics.from_item(samples[i]["camera"]))

        masks = tr.stack(masks)
        w_Ts_c = tr.stack(w_Ts_c)

        return cls(masks=masks, extrinsics=w_Ts_c, intrinsics=intrinsics)


class VoxelCarvingMesher:

    def _mesh2world(
        self, mesh: Trimesh, shift: Tensor, scale: float, resolution: int
    ) -> Trimesh:
        """Scale and shift the mesh to the original size.

        Args:
            mesh: Input mesh.
            shift: Shift.
            scale: Scale.
            resolution: Resolution.

        Returns:
            The scaled and shifted mesh.
        """

        vertices = mesh.vertices
        vertices /= resolution
        vertices *= scale
        vertices += np.array([-0.5, -0.5, -0.5]) * scale
        vertices += shift.cpu().numpy()
        mesh.vertices = vertices
        return mesh

    def _voxel_carving(
        self,
        camera_matrix: Tensor,
        c2w: Tensor,
        masks: Tensor,
        width: int,
        height: int,
        chunk: int = 32**3,
    ) -> Tensor:
        """Carve the voxel grid.


        Args:
            camera_matrix: Camera intrinsics.
            c2w: Camera extrinsics.
            masks: Masks.
            width: Image width.
            height: Image height.
            chunk: Chunk size.
            track_fn: Function to track the progress.

        Returns:
            The voxel grid.
        """
        assert camera_matrix.dim() == 3 and camera_matrix.shape[1:] == (3, 3)
        assert c2w.dim() == 3 and (c2w.shape[1:] == (3, 4) or c2w.shape[1:] == (4, 4))
        assert camera_matrix.shape[0] == c2w.shape[0] or camera_matrix.shape[0] == 1
        assert masks is not None and masks.shape[0] == c2w.shape[0]

        grid_res = tr.tensor([128] * 3)
        cells_per_lvl = int(grid_res.prod().item())
        occ_grid = tr.ones(cells_per_lvl)
        dim = 3
        aabbs = tr.tensor([[-0.5, -0.5, -0.5, 0.5, 0.5, 0.5]], dtype=tr.float32)

        grid_coords = get_meshgrid(grid_res).reshape(cells_per_lvl, dim)

        N_cams = c2w.shape[0]
        w2c_R = c2w[:, :3, :3].transpose(2, 1)  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ c2w[:, :3, 3:]  # (N_cams, 3, 1)

        lvl_indices = [tr.arange(int(grid_res.prod().item()))]

        for lvl, indices in enumerate(lvl_indices):
            grid_coords_indices = grid_coords[indices]

            for i in range(0, len(indices), chunk):
                x = grid_coords_indices[i : i + chunk] / (grid_res - 1)
                indices_chunk = indices[i : i + chunk]
                # voxel coordinates [0, 1]^3 -> world
                xyzs_w = (aabbs[lvl, :3] + x * (aabbs[lvl, 3:] - aabbs[lvl, :3])).T
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = camera_matrix @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)

                uv_x_mask = (uv[:, 0, :] >= 0) & (uv[:, 0, :] < width)
                uv_y_mask = (uv[:, 1, :] >= 0) & (uv[:, 1, :] < height)
                uv_mask = uv_x_mask & uv_y_mask  # (N_cams, chunk)
                uv_x = tr.clamp(uv[:, 0, :], 0, width - 1)  # (N_cams, chunk)
                uv_y = tr.clamp(uv[:, 1, :], 0, height - 1)  # (N_cams, chunk)

                # 0 never seen
                # 1 free (by at least one camera)
                # 2 occupied (by all cameras)
                in_mask = tr.zeros_like(uv_x)  # (N_cams, chunk)
                for cam_idx in range(N_cams):
                    value = masks[cam_idx][uv_y[cam_idx].long(), uv_x[cam_idx].long()]
                    # set to 1 voxels seen as `free` by a camera
                    # set to 2 voxels seen as `occupied` by a camera
                    value = (value > 0.0) + 1
                    # set to 0 voxels `never seen` by a camera
                    value *= uv_mask[cam_idx]
                    in_mask[cam_idx] = value
                # a voxel is occupied if:
                # - it is not seen as `free` by a number of cameras <= outliers_ratio * N_cams
                # - it is not seen as `never seen` by all cameras
                # - it is seen as `occupied` by a number of cameras >= consensus_ratio * N_cams
                in_mask = (
                    ((in_mask == 1).sum(0) <= (N_cams * 0.05))
                    & ~(in_mask == 0).all(0)
                    & ((in_mask == 2).sum(0) >= (N_cams * 0.95))
                )

                cell_ids_base = lvl * cells_per_lvl
                occ_grid[cell_ids_base + indices_chunk] = tr.where(in_mask, 1.0, 0.0)

        occ_grid = occ_grid.reshape(*grid_res)  # type: ignore

        return occ_grid

    def mesh(
        self,
        data: VoxelCarvingMesherInput,
    ) -> Trimesh:
        """Create mesh using the default neural-twin internal method.

        Args:
            data: Input data for the mesher.
            track_fn: Function to track the progress.

        Returns:
            Generated mesh.
        """
        data_in = cast(VoxelCarvingMesherInput, data)

        shift = tr.mean(data_in.extrinsics[..., :3, 3], dim=0)
        scale = 2 * tr.max(tr.abs(data_in.extrinsics[:, :3, 3])).item()

        w_Ts_c_ncd = data_in.extrinsics.clone().to(tr.float32)
        w_Ts_c_ncd[:, :3, 3] -= shift
        w_Ts_c_ncd[:, :3, 3] /= scale

        camera_matrices = tr.stack(
            [intrinsic.K for intrinsic in data_in.intrinsics]
        ).to(tr.float32)
        grid = self._voxel_carving(
            camera_matrices.float(),
            w_Ts_c_ncd,
            data_in.masks,
            data_in.intrinsics[0].width,
            data_in.intrinsics[0].height,
        )

        vertices, faces, _, _ = measure.marching_cubes(grid.numpy(), level=0)

        mesh = Trimesh(vertices=vertices, faces=faces)
        mesh.invert()

        mesh = self._mesh2world(mesh, shift, scale, 128)
        return mesh


class RescaleTwinCommand(PipelimeCommand, title="rescale_twin"):
    """Convert a folder of images to underfolder dataset."""

    input: plint.InputDatasetInterface = plint.InputDatasetInterface.pyd_field(
        piper_port=PiperPortType.INPUT,
        description="Input underfolder of images",
    )
    input_images: plint.InputDatasetInterface = plint.InputDatasetInterface.pyd_field(
        piper_port=PiperPortType.INPUT,
        description="Input underfolder of images",
    )
    output: plint.OutputDatasetInterface = plint.OutputDatasetInterface.pyd_field(
        piper_port=PiperPortType.OUTPUT,
        description="Output undefolder",
    )

    def run(self):
        import pipelime.sequences as pls
        import trimesh
        from eyeo.neural_twin import NeuralTwinItem
        from eyesplat.neural_twin import EyesplatNeuralTwin

        uf = self.input.create_reader()
        nt: EyesplatNeuralTwin = uf[0]["neural_twin"]()

        uf_images = self.input_images.create_reader()
        data = VoxelCarvingMesherInput.from_underfolder(uf_images)
        voxel_carving = VoxelCarvingMesher()
        mesh_target = voxel_carving.mesh(data)
        if not isinstance(mesh_target, trimesh.Trimesh):
            mesh_target = mesh_target.dump(concatenate=True)  # type: ignore

        mesh_source = nt.mesh
        if not isinstance(mesh_source, trimesh.Trimesh):
            mesh_source = mesh_source.dump(concatenate=True)  # type: ignore

        # Rotation to align the meshes
        R_x_neg_90 = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
        mesh_source.apply_transform(R_x_neg_90)
        # Scale based on the bounding box size
        bbox_target = mesh_target.bounding_box_oriented.extents
        bbox_source = mesh_source.bounding_box_oriented.extents
        scale_factor = np.max(bbox_target) / np.max(bbox_source)
        mesh_source_rescaled = mesh_source.copy()
        mesh_source_rescaled.apply_scale(scale_factor)

        # Align the meshes by their centroids
        centroid_target = mesh_target.centroid
        centroid_source = mesh_source_rescaled.centroid
        translation = centroid_target - centroid_source
        mesh_source_rescaled.apply_translation(translation)

        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = R_x_neg_90[:3, :3]
        transform_matrix[:3, :3] *= scale_factor
        transform_matrix[:3, 3] = translation
        nt.apply_transform(
            torch.tensor(transform_matrix, dtype=torch.float32, device="cuda")
        )
        nt.mesh = mesh_source_rescaled

        pls.SamplesSequence.from_list(
            [
                pls.Sample(
                    {
                        "neural_twin": NeuralTwinItem(nt),
                    }
                )
            ]
        ).to_underfolder(self.output.folder, exists_ok=True).run()
