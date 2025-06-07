import os
import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import HyperParameters as hp

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        # Load Qwen2.5-VL processor (handles vision + language preprocessing)
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True, 
            clean_up_tokenization_spaces=True,
            use_fast=True
        )

        # Load the AWQ-quantized Qwen2.5-VL model
        # torch_dtype="auto" lets HF pick the correct dtype for AWQ; device_map="auto" assigns layers across GPUs
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.float16, # set `torch_dtype=torch.float16` for better efficiency with AWQ.
            low_cpu_mem_usage=True,
            device_map="auto", 
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        responses = [] 
        for request in requests:
            # Get inputs from request
            image_arr = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()

            # Decode bytes → str
            prompt = prompt_tensor[0].decode("utf-8")

            # Preprocess: AutoProcessor will tokenize text and preprocess image into pixel_values
            inputs = self.processor(text=[prompt], images=[image_arr], return_tensors="pt").to(self.device)

            # Run inference using the Qwen2.5-VL model
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=hp.max_new_tokens,
                    early_stopping=hp.early_stopping,
                    do_sample=hp.do_sample,
                    num_beams=hp.num_beams,
                )

            # Decode the token IDs back into text.
            generated_text = self.processor.batch_decode( generated_ids, skip_special_tokens=True)[0]

            # Serialize string → bytes
            answer_bytes = generated_text.encode("utf-8")

            # Prepare the final parsed answer as a response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "answer",
                        np.array([answer_bytes], dtype=object)
                    )
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):
        """
        Called when the model is unloaded. You can release resources here if needed.
        """
        pass
