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
            image = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()

            # Get text input from the request (prompt & text_input)
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()
            txtinput_tensor = pb_utils.get_input_tensor_by_name(request, "text_input").as_numpy()

            # Decode the strings
            task_prompt = prompt_tensor[0].decode("utf-8")
            txtinput = txtinput_tensor[0].decode("utf-8") if txtinput_tensor.size > 0 else None

            # Add txt input if provided
            if txtinput is None:
                prompt = task_prompt
            else:
                prompt = task_prompt + txtinput

            # Preprocess the image and text using Florence 2 processor
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")

            # Prepare the processed result as a Triton response, passing image dimensions and task prompt
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("pixel_values", inputs["pixel_values"].numpy()),
                pb_utils.Tensor("input_ids", inputs["input_ids"].numpy())
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary