import mimetypes
import json
from typing import Iterator
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow_api.json"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        # Give a list of weights filenames to download during setup
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[],
        )

    def chaos_to_denoise(self, chaos: int):
        denoise_values = {
            0: 1.0,
            1: 0.99,
            2: 0.98,
            3: 0.97,
            4: 0.96,
            5: 0.95,
            6: 0.94,
            7: 0.93,
            8: 0.92,
            9: 0.91,
            10: 0.90,
        }
        return denoise_values.get(chaos)

    def aspect_ratio_to_width_height(self, aspect_ratio: str):
        aspect_ratios = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "21:9": (1536, 640),
            "3:2": (1216, 832),
            "2:3": (832, 1216),
            "4:5": (896, 1088),
            "5:4": (1088, 896),
            "9:16": (768, 1344),
            "9:21": (640, 1536),
        }
        return aspect_ratios.get(aspect_ratio)

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        positive_prompt = workflow["282"]["inputs"]
        positive_prompt["Text"] = kwargs["prompt"]

        sampler = workflow["297"]["inputs"]
        sampler["cfg"] = kwargs["guidance_scale"]
        sampler["denoise"] = kwargs["denoise"]

        empty_latent_image = workflow["342"]["inputs"]
        empty_latent_image["width"] = kwargs["width"]
        empty_latent_image["height"] = kwargs["height"]

        if kwargs["weird"]:
            content_shuffle = workflow["404"]["inputs"]
            content_shuffle["resolution"] = kwargs["width"]
            workflow["396"]["inputs"]["image"] = ["404", 0]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        chaos: int = Input(
            description="Higher values lead to more variation in image outputs",
            le=10,
            ge=0,
            default=5,
        ),
        weird: bool = Input(
            description="If true, the outputs will be a lot weirder because they will be forced into more unusual compositions.",
            default=False,
        ),
        aspect_ratio: str = Input(
            choices=["1:1", "16:9", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"],
            default="1:1",
        ),
        guidance_scale: float = Input(
            description="The guidance scale tells the model how similar the output should be to the prompt.",
            le=20,
            ge=0,
            default=4.5,
        ),
        number_of_images: int = Input(
            description="The number of images to generate",
            le=10,
            ge=1,
            default=1,
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
    ) -> Iterator[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)

        width, height = self.aspect_ratio_to_width_height(aspect_ratio)
        denoise = self.chaos_to_denoise(chaos)

        self.comfyUI.connect()
        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            guidance_scale=guidance_scale,
            denoise=denoise,
            width=width,
            height=height,
            weird=weird,
        )

        returned_files = []

        for i in range(number_of_images):
            self.comfyUI.randomise_seeds(workflow)
            self.comfyUI.run_workflow(workflow)

            all_output_files = self.comfyUI.get_files(OUTPUT_DIR)
            new_files = [
                file
                for file in all_output_files
                if file.name.rsplit(".", 1)[0] not in returned_files
            ]
            optimised_images = optimise_images.optimise_image_files(
                output_format, output_quality, new_files
            )

            for image in optimised_images:
                yield Path(image)

            returned_files.extend(
                [file.name.rsplit(".", 1)[0] for file in all_output_files]
            )
