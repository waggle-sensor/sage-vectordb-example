import numpy as np
import os
from transformers import AutoProcessor
import triton_python_backend_utils as pb_utils
import json

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

            # Get additional parameters: image width, image height, and task prompt
            image_width = pb_utils.get_input_tensor_by_name(request, "image_width").as_numpy()[0]
            image_height = pb_utils.get_input_tensor_by_name(request, "image_height").as_numpy()[0]
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0].decode("utf-8")

            # Decode the generated ids into text
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Post-process the generated text
            answer_dict = self.processor.post_process_generation(
                generated_text,
                task=prompt,
                image_size=(image_width, image_height)
            )

            # Convert the dictionary to a string
            answer_str = json.dumps(answer_dict)

            # Encode
            answer = answer_str.encode("utf-8")

            # Prepare the final parsed answer as a response
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("answer", np.array([answer], dtype=object))
            ])
            responses.append(inference_response)

        return responses

    def finalize(self):
        pass  # Cleanup if necessary
