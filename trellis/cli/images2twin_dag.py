import pipelime.commands.interfaces as plint
import pydantic.v1 as pyd
from pipelime.commands.piper import T_NODES, DagBaseCommand
from pipelime.piper import PiperPortType


class Images2TwinDag(
    DagBaseCommand, title="images2twin_dag", schema_extra={"version": "0.0.0"}
):
    """DAG to generate scene Neural Twin."""

    input_folder: plint.InputDatasetInterface = pyd.Field(
        piper_port=PiperPortType.INPUT,
        description="Input underfolder of images",
    )

    input_marker_folder: plint.InputDatasetInterface | None = pyd.Field(
        default=None,
        piper_port=PiperPortType.INPUT,
        description="Path to input underfolder with marker configs.",
    )

    output_folder: plint.OutputDatasetInterface = pyd.Field(
        piper_port=PiperPortType.OUTPUT,
        description="Output underfolder",
    )

    def create_graph(self) -> T_NODES:
        from calibry.cli import calibrate_opencv
        from trellis.cli.images2twin import Images2TwinCommand
        from trellis.cli.make_masks import MakeMasksCommand
        from pipelime.commands import CopySharedItemsCommand
        from trellis.cli.rescale_twin import RescaleTwinCommand

        nodes = {}

        # If no marker folder is provided, skip calibration and masking steps
        # The masking is done with rembg
        if self.input_marker_folder is None:
            nodes["create_twin"] = Images2TwinCommand(
                input=self.input_folder,
                output=self.output_folder,
                image_key="color",
                mask_key="mask",
            )
            return nodes

        uf_with_marker_config = self.folder_debug / "uf_with_marker_config"
        nodes["copy_marker_config"] = CopySharedItemsCommand(
            source=self.input_marker_folder,
            dest=self.input_folder,
            output=uf_with_marker_config,
            key_list=["marker_cfg"],
            force_shared=True,
        )

        uf_calibrated = self.folder_debug / "uf_calibrated"
        nodes["calibrate"] = calibrate_opencv(
            input_folder=uf_with_marker_config,
            output_folder=uf_calibrated,
            image_key="color",
            marker_key="marker_cfg",
            out_pose_key="w2c",
            out_camera_key="camera",
            min_corners=6,
            max_iterations=1,
            min_frames=2,
        )

        uf_calibrated_masked = self.folder_debug / "uf_calibrated_masked"
        nodes["make_masks"] = MakeMasksCommand(
            input=uf_calibrated,
            output=uf_calibrated_masked,
        )

        trellis_neural_twin_folder = self.folder_debug / "trellis_neural_twin"
        nodes["create_twin"] = Images2TwinCommand(
            input=uf_calibrated_masked,
            output=trellis_neural_twin_folder,
            image_key="color",
            mask_key="mask",
        )

        twin_rescaled_folder = self.output_folder
        nodes["rescale_twin"] = RescaleTwinCommand(
            input=trellis_neural_twin_folder,
            input_images=uf_calibrated_masked,
            output=twin_rescaled_folder,
        )

        return nodes
