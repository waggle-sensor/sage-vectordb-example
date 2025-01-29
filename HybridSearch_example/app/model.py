'''This file contains the code to run the Florence 2 model'''
#NOTE: This will be deployed on our cloud with a machine with cuda and communication 
#   with our cloud k8s namespace beehive-sage. I will need to deploy florence 2 with Triton Inference Server.
#   This microservice will communicate with the microservice that will deploy data.py

import logging
import HyperParameters as hp
from collections import OrderedDict
from PIL import Image
import tritonclient.grpc as TritonClient
import numpy as np
import json

def triton_run_model(triton_client, task_prompt, image_path, text_input=""):
    """
    takes in a task prompt and image, returns an answer 
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs for Triton
    image_width, image_height = image.size
    image_np = np.array(image).astype(np.float32)
    task_prompt_bytes = task_prompt.encode("utf-8")
    text_input_bytes = text_input.encode("utf-8")

    # Prepare inputs & outputs for Triton
    # NOTE: if you enable max_batch_size, leading number is batch size, example [1,1] 1 is batch size
    inputs = [
        TritonClient.InferInput("image", [image_height, image_width, 3], "FP32"),
        TritonClient.InferInput("prompt", [1], "BYTES"),
        TritonClient.InferInput("text_input", [1], "BYTES"),
        TritonClient.InferInput("image_width", [1], "INT32"),
        TritonClient.InferInput("image_height", [1], "INT32")
    ]
    outputs = [
        TritonClient.InferRequestedOutput("answer")
    ]

    # Add tensors
    inputs[0].set_data_from_numpy(image_np)
    inputs[1].set_data_from_numpy(np.array([task_prompt_bytes], dtype="object"))
    inputs[2].set_data_from_numpy(np.array([text_input_bytes], dtype="object"))
    inputs[3].set_data_from_numpy(np.array([image_width], dtype="int32"))
    inputs[4].set_data_from_numpy(np.array([image_height], dtype="int32"))

    # Perform inference
    try:
        response = triton_client.infer(model_name="florence2base", inputs=inputs, outputs=outputs)

        # Get the result
        answer = response.as_numpy("answer")[0]
        answer_str = answer.decode("utf-8")

        # Convert the JSON string to a dictionary
        answer_dict = json.loads(answer_str)

        return answer_dict
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}")
        return None

def triton_gen_caption(triton_client, image_path):
    """
    Generate image caption using the provided model
    """
    task_prompt = '<MORE_DETAILED_CAPTION>'

    description_text = triton_run_model(triton_client, task_prompt, image_path)
    description_text = description_text[task_prompt]

    #takes those details from the setences and finds labels and boxes in the image
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    boxed_descriptions = triton_run_model(triton_client, task_prompt, image_path, description_text)

    #only prints out labels not bboxes
    descriptions = boxed_descriptions[task_prompt]['labels']
    logging.debug(f'Labels Generated: {descriptions}')

    #finds other things in the image that the description did not explicitly say
    task_prompt = '<DENSE_REGION_CAPTION>'
    labels = triton_run_model(triton_client, task_prompt, image_path)

    #only prints out labels not bboxes
    printed_labels = labels[task_prompt]['labels']

    # Join description_text into a single string
    description_text_joined = "".join(description_text)

    #makes unique list of labels and adds commas
    label_list = descriptions + printed_labels
    unique_labels = list(OrderedDict.fromkeys(label_list))
    labels = ", ".join(unique_labels)

    # Combine all lists into one list
    combined_list = ["DESCRIPTION:"] + [description_text_joined] + ["LABELS:"] + [labels]

    # Join the unique items into a single string with spaces between them
    final_description = " ".join(combined_list)

    logging.debug(f'Final Generated Description: {final_description}')
    return final_description