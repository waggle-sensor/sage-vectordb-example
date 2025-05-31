'''This file contains the code to talk to Triton Inference Server'''

import logging
import tritonclient.grpc as TritonClient
import numpy as np

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