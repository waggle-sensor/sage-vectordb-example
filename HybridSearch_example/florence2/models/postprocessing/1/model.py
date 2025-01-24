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
            trust_remote_code=True
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            # Extract the generated_ids tensor from the request
            generated_ids = pb_utils.get_input_tensor_by_name(request, "generated_ids").as_numpy()

            # Decode the generated_ids into text (the caption)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Post-process the generated text
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt, #left of here, how to pass down the pipeline?
                image_size=(image_width, image_height)
            )

            # Prepare the processed result as a response
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("parsed_answer", np.array([parsed_answer], dtype=object))
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary
