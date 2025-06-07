import os
import numpy as np
from PIL import Image
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
import HyperParameters as hp
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

MODEL_PATH = os.environ.get("QWEN_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):

        # Load Qwen2.5-VL processor
        self.processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            trust_remote_code=True, 
            clean_up_tokenization_spaces=True,
            use_fast=True,
            min_pixels=hp.min_pixels,
            max_pixels=hp.max_pixels,
        )

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load the AWQ-quantized Qwen2.5-VL model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            quantization_config=bnb,
            local_files_only=True,
            # torch_dtype=torch.float16, # set `torch_dtype=torch.float16` for better efficiency with AWQ.
            low_cpu_mem_usage=True,
            device_map="auto" # assigns layers across GPUs
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def execute(self, requests):
        responses = []
        for request in requests:
            torch.cuda.empty_cache()

            # Unpack inputs
            image_arr = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
            image = Image.fromarray(image_arr.astype('uint8')).convert('RGB')
            prompt_bytes = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            prompt_text = prompt_bytes.decode("utf-8")

            # Build the chat “messages” structure
            messages = [
                {"type": "image", "image": image_arr},
                {"type": "text",  "text": prompt_text},
            ]
            
            # Apply the chat template to get the full token prompt
            text_input = self.processor.apply_chat_template(
                [{"role":"user","content":messages}],
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize + prep vision tensors
            inputs = self.processor(
                text=[text_input],
                images=[image],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            # Generate
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=hp.max_new_tokens,
                    early_stopping=hp.early_stopping,
                    do_sample=hp.do_sample,
                    num_beams=hp.num_beams,
                )

            # Trim off the prompt tokens
            in_ids = inputs["input_ids"]
            trimmed = [
                out_ids[len(in_ids[i]) :] for i, out_ids in enumerate(generated)
            ]

            # Decode and build Triton response
            output_text = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            answer_bytes = output_text.encode("utf-8")

            tensor = pb_utils.Tensor("answer", np.array([answer_bytes], dtype=object))
            responses.append(pb_utils.InferenceResponse(output_tensors=[tensor]))

        return responses

    def finalize(self):
        pass

# class TritonPythonModel:
#     def initialize(self, args):
#         # Load Qwen2.5-VL processor
#         self.processor = AutoProcessor.from_pretrained(
#             MODEL_PATH,
#             local_files_only=True,
#             trust_remote_code=True, 
#             clean_up_tokenization_spaces=True,
#             use_fast=True
#         )

#         # Load the AWQ-quantized Qwen2.5-VL model
#         self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#             MODEL_PATH,
#             local_files_only=True,
#             torch_dtype=torch.float16, # set `torch_dtype=torch.float16` for better efficiency with AWQ.
#             low_cpu_mem_usage=True,
#             device_map="auto" # assigns layers across GPUs
#         )

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def execute(self, requests):
#         responses = [] 
#         for request in requests:
#             # Get inputs from request
#             image_arr = pb_utils.get_input_tensor_by_name(request, "image").as_numpy()
#             prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()

#             # Decode bytes → str
#             prompt = prompt_tensor[0].decode("utf-8")

#             # Preprocess: AutoProcessor will tokenize text and preprocess image into pixel_values
#             inputs = self.processor(text=[prompt], images=[image_arr], return_tensors="pt").to(self.device)

#             # Run inference using the Qwen2.5-VL model
#             with torch.no_grad():
#                 generated_ids = self.model.generate(
#                     **inputs,
#                     max_new_tokens=hp.max_new_tokens,
#                     early_stopping=hp.early_stopping,
#                     do_sample=hp.do_sample,
#                     num_beams=hp.num_beams,
#                 )

#             # Decode the token IDs back into text.
#             generated_text = self.processor.batch_decode( generated_ids, skip_special_tokens=True)[0]

#             # Serialize string → bytes
#             answer_bytes = generated_text.encode("utf-8")

#             # Prepare the final parsed answer as a response
#             inference_response = pb_utils.InferenceResponse(
#                 output_tensors=[
#                     pb_utils.Tensor(
#                         "answer",
#                         np.array([answer_bytes], dtype=object)
#                     )
#                 ]
#             )
#             responses.append(inference_response)

#         return responses

#     def finalize(self):
#         """
#         Called when the model is unloaded. You can release resources here if needed.
#         """
#         pass
