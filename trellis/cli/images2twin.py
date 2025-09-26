import os
from pathlib import Path
import pipelime.commands.interfaces as plint
import pydantic.v1 as pyd
from trellis.pipelines import TrellisImageTo3DPipeline
from pipelime.commands.piper import PipelimeCommand
from pipelime.piper import PiperPortType
from trellis.utils import postprocessing_utils
import typing as t
from trellis.representations.gaussian.general_utils import inverse_sigmoid

os.environ["SPCONV_ALGO"] = "native"


class Images2TwinCommand(PipelimeCommand, title="images2twin"):
    """Convert a folder of images to underfolder dataset."""

    input: plint.InputDatasetInterface = plint.InputDatasetInterface.pyd_field(
        piper_port=PiperPortType.INPUT,
        description="Input underfolder of images",
    )
    output: plint.OutputDatasetInterface = plint.OutputDatasetInterface.pyd_field(
        piper_port=PiperPortType.OUTPUT,
        description="Output undefolder",
    )

    image_key: str = pyd.Field(
        default="color",
        description="Key for the image in the output dataset",
    )
    mask_key: t.Optional[str] = pyd.Field(
        default="mask",
        description="Key for the mask in the output dataset",
    )

    def run(self):
        import pipelime.sequences as pls
        from PIL import Image
        from rembg import new_session, remove
        import trimesh
        import torch
        from trellis.representations.gaussian.gaussian_model import Gaussian
        from eyesplat.config import TrainConfig
        from eyeo.neural_twin import NeuralTwinItem
        from eyesplat.neural_twin import EyesplatNeuralTwin

        uf = self.input.create_reader()
        # Load images as np arrays
        images = []
        for s in uf:
            image = s[self.image_key]()
            images.append(Image.fromarray(image))

        weights_folder = Path(os.environ.get("TRELLIS_ROOT", "/app/weights"))
        print(f"Using weights folder: {weights_folder}")
        folder = str(weights_folder / "u2net.onnx")
        rembg_session = new_session(model_name="u2net_custom", model_path=folder)

        images_masked = []
        if self.mask_key is not None and self.mask_key in uf[0].keys():
            for s, image in zip(uf, images):
                mask = s[self.mask_key]()
                mask = Image.fromarray(mask).convert("L")  # ensure grayscale
                # Apply mask directly and keep RGBA
                image_masked = image.convert("RGBA")
                image_masked.putalpha(mask)
                images_masked.append(image_masked)
        else:
            for image in images:
                # rembg already outputs RGBA
                image_masked = remove(image, session=rembg_session)
                # Ensure RGBA mode
                image_masked = image_masked.convert("RGBA")
                images_masked.append(image_masked)

        folder = str(
            weights_folder
            / "models--gqk--TRELLIS-image-large-fork/snapshots/25e0d31ffbebe4b5a97464dd851910efc3002d96"
        )

        pipeline = TrellisImageTo3DPipeline.from_pretrained(folder)
        pipeline.cuda()
        outputs = pipeline.run_multi_image(
            images_masked,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )

        gaussians: Gaussian = outputs["gaussian"][0]
        means = gaussians.get_xyz.detach().cpu()
        quats = (gaussians._rotation + gaussians.rots_bias[None, :]).detach().cpu()
        scales = torch.log(gaussians.get_scaling).detach().cpu()
        opacities = inverse_sigmoid(gaussians.get_opacity).detach().cpu().squeeze(-1)
        f_dc = gaussians._features_dc.detach().contiguous().cpu()
        sh_degree = gaussians.sh_degree

        config = TrainConfig()
        config.module.sh_degree = sh_degree
        eyesplat_dict = {
            "shift_scale": {"shift": torch.tensor([0.0, 0.0, 0.0]), "scale": 1.0},
            "info": {"w2obj": torch.eye(4), "w2cs": []},
            "state_dict": {
                "means": means,
                "scales": scales,
                "quats": quats,
                "opacities": opacities,
                "colors": f_dc,
            },
            "train_config": config.dict(),
        }

        nt = EyesplatNeuralTwin(eyesplat_dict, {})

        # Mesh
        mesh = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,  # Ratio of triangles to remove in the simplification process
            texture_size=1024,  # Size of the texture used for the GLB
        )
        vertices = outputs["mesh"][0].vertices
        faces = outputs["mesh"][0].faces
        face_normals = outputs["mesh"][0].face_normal
        mesh = trimesh.Trimesh(
            vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy()
        )
        mesh.visual.face_normals = face_normals.cpu().numpy()
        nt.mesh = mesh

        sample = pls.Sample({"neural_twin": NeuralTwinItem(nt)})
        pls.SamplesSequence.from_list([sample]).to_underfolder(
            self.output.folder, exists_ok=True
        ).run()
