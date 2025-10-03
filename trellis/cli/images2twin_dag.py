from enum import Enum
import pipelime.commands.interfaces as plint
import pydantic.v1 as pyd
from pipelime.commands.piper import T_NODES, DagBaseCommand
from pipelime.piper import PiperPortType


class ModelPaths(Enum):
    rembg_model_path: str = "/app/weights/u2net.onnx"
    dino_model_path: str = "/app/weights/hub"
    trellis_model_path: str = (
        "/app/weights/models--gqk--TRELLIS-image-large-fork/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96"
    )
    cutie_model_path: str = "/app/weights/cutie-base-mega.pth"
    depth_anything_model_path: str = (
        "/app/weights/models--depth-anything--Depth-Anything-V2-Large-hf/snapshots/7581137eff8d4e94f6e796d3baea0e9fa79b22d2"
    )


class Images2TwinDag(
    DagBaseCommand, title="images2twin_dag", schema_extra={"version": "0.0.0"}
):
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

    rembg_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the rembg model. If None, use the default model.",
    )
    trellis_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the trellis models. If None, use the default models.",
    )
    dino_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the dino model. If None, use the default models.",
    )
    cutie_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the cutie model. If None, use the default model.",
    )
    depth_anything_model_path: str | None = pyd.Field(
        default=None,
        description="Path to the depth anything model. If None, use the default model.",
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
                dino_model_path=self.dino_model_path,
                rembg_model_path=self.rembg_model_path,
                trellis_model_path=self.trellis_model_path,
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
            cutie_model_path=self.cutie_model_path,
            depth_anything_model_path=self.depth_anything_model_path,
            rembg_model_path=self.rembg_model_path,
        )

        trellis_neural_twin_folder = self.folder_debug / "trellis_neural_twin"
        nodes["create_twin"] = Images2TwinCommand(
            input=uf_calibrated_masked,
            output=trellis_neural_twin_folder,
            image_key="color",
            mask_key="mask",
            dino_model_path=self.dino_model_path,
            rembg_model_path=self.rembg_model_path,
            trellis_model_path=self.trellis_model_path,
        )

        twin_rescaled_folder = self.output_folder
        nodes["rescale_twin"] = RescaleTwinCommand(
            input=trellis_neural_twin_folder,
            input_images=uf_calibrated_masked,
            output=twin_rescaled_folder,
        )

        return nodes


class Images2TwinDagProd(
    Images2TwinDag, title="images2twin_dag_prod", schema_extra={"version": "0.0.0"}
):

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
        print(ModelPaths.rembg_model_path)
        return {
            "images2twin_dag": Images2TwinDag(
                input_folder=self.input_folder,
                input_marker_folder=self.input_marker_folder,
                output_folder=self.output_folder,
                rembg_model_path=ModelPaths.rembg_model_path.value,
                dino_model_path=ModelPaths.dino_model_path.value,
                trellis_model_path=ModelPaths.trellis_model_path.value,
                cutie_model_path=ModelPaths.cutie_model_path.value,
                depth_anything_model_path=ModelPaths.depth_anything_model_path.value,
            )
        }
