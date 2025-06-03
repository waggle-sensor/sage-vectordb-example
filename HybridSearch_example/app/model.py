'''This file contains the code to talk to Triton Inference Server'''

import logging
import tritonclient.grpc as TritonClient
import numpy as np
import HyperParameters as hp

def get_colbert_embedding(triton_client, text):
    """
    Embed text using ColBERT encoder served via Triton Inference Server.
    Returns token-level embeddings of shape [num_tokens, 128]
    """
    # Prepare input
    text_bytes = text.encode("utf-8")
    input_tensor = np.array([text_bytes], dtype="object")  # batch size = 1

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("text", input_tensor.shape, "BYTES")
    ]
    outputs = [
        TritonClient.InferRequestedOutput("embedding"),
        TritonClient.InferRequestedOutput("token_lengths")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(input_tensor)

    # Run inference
    try:
        results = triton_client.infer(model_name="colbert", inputs=inputs, outputs=outputs)

        # Retrieve and reshape output
        emb_flat = results.as_numpy("embedding")           # shape: (1, max_len * 128)
        token_lengths = results.as_numpy("token_lengths")  # shape: (1,)
        num_tokens = token_lengths[0]

        # Reshape and unpad
        emb_3d = emb_flat.reshape(1, -1, 128)
        token_embeddings = emb_3d[0, :num_tokens, :]  # shape: [num_tokens, 128]
    except Exception as e:
        logging.error(f"Error during Colbert inference: {str(e)}")
        return None

    return token_embeddings

def get_allign_embeddings(triton_client, text, image=None):
    """
    Embed text and image using ALIGN encoder served via Triton Inference Server.
    Returns one embedding created via embeddings for both modalities.
    """
    def fuse_embeddings( img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Given two L2-normalized vectors img_emb and txt_emb (shape (D,)), 
        returns their weighted sum (alpha * img + (1-alpha) * txt), re-normalized to unit norm.
        """
        if img_emb.shape != txt_emb.shape:
            raise ValueError("img_emb and txt_emb must have the same dimension")

        # Weighted sum
        combined = alpha * img_emb + (1.0 - alpha) * txt_emb

        # Re-normalize
        norm = np.linalg.norm(combined)
        if norm == 0.0:
            # Edge case: if they cancel out exactly (unlikely), fall back to text alone
            return txt_emb.copy()
        return (combined / norm).astype(np.float32)
    
    # Prepare inputs
    text_bytes = text.encode("utf-8")

    # Prepare inputs & outputs for Triton
    inputs = [TritonClient.InferInput("text", [1], "BYTES")]
    outputs = [TritonClient.InferRequestedOutput("text_embedding")]

    # Add tensors
    inputs[0].set_data_from_numpy(np.array([text_bytes], dtype="object"))

    # If image is provided, prepare image input
    if image is not None:
        # Prepare inputs
        image_width, image_height = image.size
        image_np = np.array(image).astype(np.float32)

        # Prepare inputs & outputs for Triton
        inputs.append(TritonClient.InferInput("image", [image_height, image_width, 3], "FP32"))
        outputs.append(TritonClient.InferRequestedOutput("image_embedding"))

        # Add tensors
        inputs[1].set_data_from_numpy(image_np)

    # Run inference
    try:
        results = triton_client.infer(model_name="align", inputs=inputs, outputs=outputs)

        # Retrieve embeddings
        text_embedding = results.as_numpy("text_embedding")[0]
        image_embedding = results.as_numpy("image_embedding")[0] if image is not None else None 
    except Exception as e:
        logging.error(f"Error during ALIGN inference: {str(e)}")
        return None
    
    # fuse embeddings if image is provided
    embedding = text_embedding
    if image_embedding is not None:
        embedding = fuse_embeddings(image_embedding, text_embedding, alpha=hp.align_alpha)

    return embedding
