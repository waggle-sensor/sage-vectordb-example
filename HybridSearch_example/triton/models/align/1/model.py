import os
import io
import numpy as np
import triton_python_backend_utils as pb_utils
import torch
from PIL import Image
from transformers import AlignProcessor, AlignModel

MODEL_PATH = os.environ.get("ALIGN_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        """
        Load ALIGN’s processor and model in one shot.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # AlignProcessor handles BOTH text-tokenization and image-preprocessing
        self.processor = AlignProcessor.from_pretrained(
            MODEL_PATH,
            clean_up_tokenization_spaces=True
        )

        # ALIGN model itself (supports get_text_features & get_image_features)
        self.model = AlignModel.from_pretrained(
            MODEL_PATH,
            clean_up_tokenization_spaces=True
        ).to(self.device)
        self.model.eval()

    def execute(self, requests):
        """
        For each incoming request:
          - If it has a “text” tensor, run get_text_features(...) → shape [B, D]
          - If it has an “image” tensor, run get_image_features(...) → shape [B, D]
        """
        responses = []

        for request in requests:
            text_embeddings = None
            image_embeddings = None

            # ── 1) TEXT BRANCH ───────────────────────────────────────
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            if text_tensor is not None:
                # Triton will give us a numpy array of shape (batch_size,), dtype=object (bytes)
                raw_texts = text_tensor.as_numpy()  # e.g. array([b"some text", b"another text"], dtype=object)
                batch_texts = [t.decode("utf-8") for t in raw_texts]

                # AlignProcessor will tokenize
                encoded = self.processor(text=batch_texts, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    feats = self.model.get_text_features(**encoded)  # shape: [B, D]

                # Move to CPU numpy and ensure float32
                text_embeddings = feats.cpu().numpy().astype(np.float32)  # [B, D]

            # ── 2) IMAGE BRANCH ───────────────────────────────────────
            image_tensor = pb_utils.get_input_tensor_by_name(request, "image")
            if image_tensor is not None:
                # Triton gives us a numpy array of bytes
                image_np = image_tensor.as_numpy() 

                # AlignProcessor handles resizing/normalizing
                inputs = self.processor(images=image_np, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    feats = self.model.get_image_features(**inputs)  # shape: [B, D]

                image_embeddings = feats.cpu().numpy().astype(np.float32)  # [B, D]

            # ── 3) PACKAGE THE OUTPUTS ────────────────────────────────
            output_tensors = []

            if text_embeddings is not None:
                # text_embeddings: shape [B, D]
                text_tensor_out = pb_utils.Tensor("text_embedding", text_embeddings)
                output_tensors.append(text_tensor_out)

            if image_embeddings is not None:
                # image_embeddings: shape [B, D]
                img_tensor_out = pb_utils.Tensor("image_embedding", image_embeddings)
                output_tensors.append(img_tensor_out)

            responses.append(
                pb_utils.InferenceResponse(output_tensors=output_tensors)
            )

        return responses

    def finalize(self):
        """No special cleanup required."""
        pass
