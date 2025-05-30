import numpy as np
import os
import triton_python_backend_utils as pb_utils
import torch
from pathlib import Path
from ragatouille import RAGPretrainedModel

MODEL_PATH = os.environ.get("COLBERT_MODEL_PATH")

class TritonPythonModel:
    def initialize(self, args):
        """
        Initialize the Triton Python model by loading the ColBERT model.
        Args:
            args (dict): Initialization arguments.
        """
        # Check if GPU is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load ColBERT
        self.colbert = RAGPretrainedModel.from_pretrained(Path(MODEL_PATH), n_gpu=-1 if self.device == "cuda" else 0)

    def execute(self, requests):
        """
        Perform inference using the ColBERT encoder and return full token-level embeddings.
        """
        responses = []

        for request in requests:
            # Get input tensor (assume input name is "text" and dtype is BYTES)
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text").as_numpy()
            if text_tensor is None:
                raise ValueError("Input 'text' tensor is required.")
            texts = [text_tensor[0].decode("utf-8")]

            # Run ColBERT encoder
            output = self.colbert.encode(texts)
            embeddings = output["embeddings"]  # list of [tokens x 128] arrays

            # Pad to max token length across batch
            # NOTE: need to pad, Triton does not support Jagged tensors
            max_len = max(e.shape[0] for e in embeddings)
            batch_size = len(embeddings)
            emb_dim = embeddings[0].shape[1]
            padded = np.zeros((batch_size, max_len, emb_dim), dtype=np.float32)
            for i, emb in enumerate(embeddings):
                padded[i, :emb.shape[0], :] = emb  # pad with zeros

            # Return padded token-level embeddings and token lengths
            token_lengths = np.array([e.shape[0] for e in embeddings], dtype=np.int32)
            output_tensor = pb_utils.Tensor("embedding", padded)
            length_tensor = pb_utils.Tensor("token_lengths", token_lengths)

            responses.append(pb_utils.InferenceResponse(
                output_tensors=[output_tensor, length_tensor]
            ))

        return responses
    
    def finalize(self):
        """
        Finalize the model, releasing any resources if necessary.
        """
        pass