import numpy as np
import os
from transformers import AutoProcessor
import triton_python_backend_utils as pb_utils

MODEL_PATH = os.environ.get("MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        # Load the Florence 2 processor
        self.processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        trust_remote_code=True)

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the image tensor from the request
            image = pb_utils.get_input_tensor_by_name(request, "image")
            # input_image = np.squeeze(inp.as_numpy())  # Shape (height, width, channels)

            # Get text input from the request
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt")

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Prepare the processed image and text inputs as response
            inference = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("pixel_values", inputs["pixel_values"].numpy()),
                pb_utils.Tensor("input_ids", inputs["input_ids"].numpy())
            ])
            responses.append(inference)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary