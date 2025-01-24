import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM
import triton_python_backend_utils as pb_utils
import HyperParameters as hp

MODEL_PATH = os.environ.get("MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        # Load the Florence 2 model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Get the preprocessed image and text inputs from the request
            pixel_values = pb_utils.get_input_tensor_by_name(request, "pixel_values").as_numpy()
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()

            # Convert to PyTorch tensors
            pixel_values_tensor = torch.tensor(pixel_values)
            input_ids_tensor = torch.tensor(input_ids)

            # Run inference using the Florence 2 model
            generated_ids = self.model.generate(
                input_ids=input_ids_tensor,
                pixel_values=pixel_values_tensor,
                max_new_tokens=hp.max_new_tokens,
                early_stopping=hp.early_stopping,
                do_sample=hp.do_sample,
                num_beams=hp.num_beams,
            )

            # Prepare the response with the generated ids and additional parameters
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("generated_ids", generated_ids.numpy()),
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary