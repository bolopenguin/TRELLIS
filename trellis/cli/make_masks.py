import pipelime.commands.interfaces as plint
import pydantic.v1 as pyd
from pipelime.commands.piper import PipelimeCommand
from pipelime.piper import PiperPortType

from typing import Iterable
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, cast

import cv2
import numpy as np
from calibry.boards.charuco import ChArUcoBoard, Frame, FramePoints

from PIL import Image


from rembg import new_session, remove
from transformers import pipeline

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
import torch


class CutieTracker:

    def __init__(self, mask: np.ndarray, cutie_model_path: str | None = None) -> None:
        from pathlib import Path
        import os

        cutie = get_default_model(cutie_model_path)
        self._processor = InferenceCore(cutie, cfg=cutie.cfg)
        self._processor.max_internal_size = 1024

        mask = (mask > 0).astype(np.uint8)
        self._labels = [1]  # Only one object to track
        self._initial_mask = torch.from_numpy(mask).cuda()

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def mask(self, img_iterator: Iterable) -> np.ndarray:

        masks = []
        for idx, image in enumerate(img_iterator):

            image = torch.from_numpy(image).cuda()
            image = image.permute(2, 0, 1) / 255.0  # HWC to CHW

            if idx == 0:
                output_prob = self._processor.step(
                    image, self._initial_mask, objects=self._labels
                )
            else:
                output_prob = self._processor.step(image, objects=self._labels)

            mask = self._processor.output_prob_to_mask(output_prob)
            mask_np = mask.cpu().numpy().astype(np.uint8) * 255
            masks.append(mask_np)

        return np.stack(masks)

    def __del__(self):
        import hydra

        # When using cutie, the global hydra state is not cleared
        hydra.core.global_hydra.GlobalHydra.instance().clear()


@dataclass
class BoardInput:
    """Masker Input data."""

    images: Sequence[np.ndarray]
    """Images."""
    extrinsics: np.ndarray
    """Camera extrinsics."""
    boards: Sequence[ChArUcoBoard]
    """List of charuco boards."""

    @classmethod
    def from_underfolder(cls, samples) -> "BoardInput":
        """Create a BoardInput object from underfolder.

        Args:
            samples: Samples from the underfolder.
            items_keys: Items keys used to read masker input data.
            track_fn: Function to track progress.
        """

        marker_cfg_keys = ("marker_cfg",)

        images, w2cs, boards = [], [], []
        seen_configs = set()
        for s in samples:
            images.append(s["color"]())
            w2cs.append(s["w2c"]())
            for key in marker_cfg_keys:
                if key in s:
                    board_cfg = s[key]()
                    board_cfg_str = str(sorted(board_cfg.items()))
                    if board_cfg_str not in seen_configs:
                        seen_configs.add(board_cfg_str)
                        boards.append(ChArUcoBoard.parse_obj(board_cfg))

        return cls(images=images, extrinsics=np.stack(w2cs), boards=boards)


class BoardMasker:
    """Opencv masker.

    Find a mask with DepthAnything and Rembg. Ensure the board is not present in the mask.
    """

    def __init__(
        self,
        rembg_model_path: str | None = None,
        depth_anything_model_path: str | None = None,
        cutie_model_path: str | None = None,
    ) -> None:
        from pathlib import Path
        import os

        if rembg_model_path is not None:
            self._rembg_session = new_session(
                model_name="u2net_custom", model_path=rembg_model_path
            )
        else:
            self._rembg_session = new_session(model_name="u2net")

        da_model_path = "depth-anything/Depth-Anything-V2-Large-hf"
        if depth_anything_model_path is not None:
            da_model_path = depth_anything_model_path
        self._da_pipeline = pipeline(
            task="depth-anything/Depth-Anything-V2-Large-hf", model=da_model_path
        )

        self._padding = 10

    def _get_depth(self, image: np.ndarray) -> np.ndarray:
        image_pil = Image.fromarray(image)
        depth_pil = self._da_pipeline(image_pil)["depth"]
        depth_array = np.array(depth_pil)[..., None].repeat(3, axis=2)
        return depth_array

    def _get_rembg_mask(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate a mask from an image using rembg, working only inside the provided mask.

        Args:
            image: Input image
            mask: Binary mask defining the region of interest
            session: rembg session
        Returns:
            Generated mask
        """
        if mask is None:
            pred_mask = remove(
                image,
                only_mask=True,
                post_process_mask=True,
                session=self._rembg_session,
            )
        else:
            mask_coords = np.where(mask > 0)
            if len(mask_coords[0]) == 0:
                return np.zeros_like(mask)

            y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
            x_min, x_max = mask_coords[1].min(), mask_coords[1].max()

            h, w = image.shape[:2]
            y_min = max(0, y_min - self._padding)
            y_max = min(h, y_max + self._padding)
            x_min = max(0, x_min - self._padding)
            x_max = min(w, x_max + self._padding)

            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_mask = mask[y_min:y_max, x_min:x_max]

            pred_mask_cropped = remove(
                cropped_image,
                only_mask=True,
                post_process_mask=True,
                session=self._rembg_session,
            )

            pred_mask_cropped = (pred_mask_cropped > 0).astype(np.uint8) * 255
            pred_mask_cropped = pred_mask_cropped & cropped_mask

            pred_mask = np.zeros_like(mask)
            pred_mask[y_min:y_max, x_min:x_max] = pred_mask_cropped

        return pred_mask

    def _get_initial_mask(
        self,
        images: Sequence[np.ndarray],
        extrinsics: np.ndarray,
        boards: Sequence[ChArUcoBoard],
    ) -> Tuple[int, np.ndarray]:

        z_values = extrinsics[:, 2, 3]
        top_indices = np.argsort(z_values)[-10:]

        masks = []
        valid_indices = []
        for idx in top_indices:
            image = images[idx]
            frame = Frame(index=idx, image=image, shape=image.shape[:2], valid=False)
            points_seq: Sequence[FramePoints] = []
            for board in boards:
                points = board.get_points(frame)
                if points:
                    points_seq.append(points)
            if not points_seq:
                continue
            corners = [points.marker_corners.value for points in points_seq]
            corners = np.concatenate(corners, axis=0).reshape(-1, 2)
            hull_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            hull = cv2.convexHull(corners.astype(int))
            cv2.fillConvexPoly(hull_mask, hull, 255)

            depth = self._get_depth(image)
            mask_depth = self._get_rembg_mask(depth, hull_mask)
            image_masked_depth = image * (mask_depth[..., None] > 0)

            frame_masked_depth = Frame(
                index=idx, image=image_masked_depth, shape=image.shape[:2], valid=False
            )
            points_masked_depth_seq: Sequence[FramePoints] = []
            for board in boards:
                points_masked_depth = board.get_points(frame_masked_depth)
                if points_masked_depth:
                    points_masked_depth_seq.append(points_masked_depth)
            if points_masked_depth_seq:
                continue

            masks.append(mask_depth)
            valid_indices.append(idx)
        # Select the mask with the largest area
        if len(masks) == 0:
            raise RuntimeError("No valid mask found.")
        areas = [np.sum(mask > 0) for mask in masks]
        best_idx = int(np.argmax(areas))
        return (valid_indices[best_idx], masks[best_idx])

    def mask(self, data: BoardInput) -> np.ndarray:

        if not isinstance(data, BoardInput):  # pragma: no cover
            raise ValueError(f"Expected BoardInput, got {type(data)}")

        data_in = cast(BoardInput, data)

        start_idx, first_mask = self._get_initial_mask(
            data_in.images, data_in.extrinsics, data_in.boards
        )
        tracker = CutieTracker(first_mask)

        sequence_one = np.arange(0, start_idx + 1)
        sequence_one = np.flip(sequence_one)
        sequence_two = np.arange(start_idx, len(data_in.images))

        sequence_one_images = [data_in.images[i] for i in sequence_one]
        sequence_two_images = [data_in.images[i] for i in sequence_two]

        sequence_one_masks = tracker.mask(sequence_one_images)
        sequence_two_masks = tracker.mask(sequence_two_images)

        # Invert the sequence one masks
        sequence_one_masks = np.flip(sequence_one_masks, axis=0)[:-1]

        return np.concatenate([sequence_one_masks, sequence_two_masks], axis=0)


class MakeMasksCommand(PipelimeCommand, title="make_masks"):
    """Convert a folder of images to underfolder dataset."""

    input: plint.InputDatasetInterface = plint.InputDatasetInterface.pyd_field(
        piper_port=PiperPortType.INPUT,
        description="Input underfolder of images",
    )
    output: plint.OutputDatasetInterface = plint.OutputDatasetInterface.pyd_field(
        piper_port=PiperPortType.OUTPUT,
        description="Output undefolder",
    )

    rembg_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the rembg model. If None, use the default model.",
    )
    depth_anything_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the depth anything model. If None, use the default model.",
    )
    cutie_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the cutie model. If None, use the default model.",
    )

    def run(self):
        import pipelime.sequences as pls
        import pipelime.items as pli

        uf = self.input.create_reader()

        data = BoardInput.from_underfolder(uf)
        masker = BoardMasker()
        masks = masker.mask(data)

        seq = []
        for s, mask in zip(uf, masks):
            s = s.set_item("mask", pli.PngImageItem(mask))
            seq.append(s)

        pls.SamplesSequence.from_list(seq).to_underfolder(
            self.output.folder, exists_ok=True
        ).run()
