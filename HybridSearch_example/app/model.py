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

def run_model(model, processor, task_prompt, image_path, text_input=None):
    """
    takes in a task prompt and image, returns an answer 
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Get the dimensions of the image
    image_width, image_height = image.size

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=hp.max_new_tokens,
    early_stopping=hp.early_stopping,
    do_sample=hp.do_sample,
    num_beams=hp.num_beams,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image_width, image_height)
    )

    return parsed_answer
    
def generate_caption(model, processor, image_path):
    """
    Generate image caption using the provided model
    """
    task_prompt = '<MORE_DETAILED_CAPTION>'

    description_text = run_model(model, processor, task_prompt, image_path)
    description_text = description_text[task_prompt]

    #takes those details from the setences and finds labels and boxes in the image
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    boxed_descriptions = run_model(model, processor, task_prompt, image_path, description_text)

    #only prints out labels not bboxes
    descriptions = boxed_descriptions[task_prompt]['labels']
    logging.debug(f'Labels Generated: {descriptions}')

    #finds other things in the image that the description did not explicitly say
    task_prompt = '<DENSE_REGION_CAPTION>'
    labels = run_model(model, processor, task_prompt, image_path)

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

def triton_run_model(triton_client, task_prompt, image_path, text_input=""):
    """
    takes in a task prompt and image, returns an answer 
    """
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Prepare inputs for Triton
    image_width, image_height = image.size
    image_np = np.array(image).astype(np.float32)
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension

    # Prepare inputs & outputs for Triton
    inputs = [
        TritonClient.InferInput("image", [1, image_height, image_width, 3], "FP32"),
        TritonClient.InferInput("prompt", [1], "STRING"),
        TritonClient.InferInput("text_input", [1], "STRING"),
        TritonClient.InferInput("image_width", [1], "INT32"),
        TritonClient.InferInput("image_height", [1], "INT32")
    ]
    outputs = [
        TritonClient.InferRequestedOutput("answer")
    ]

    # Add tensors
    logging.debug(f"task_prompt: {task_prompt}")
    logging.debug(f"text_input: {text_input}")
    inputs[0].set_data_from_numpy(image_np)
    inputs[1].set_data_from_numpy(np.array([task_prompt], dtype="object"))
    inputs[2].set_data_from_numpy(np.array([text_input], dtype="object"))
    inputs[3].set_data_from_numpy(np.array([image_width], dtype="int32"))
    inputs[4].set_data_from_numpy(np.array([image_height], dtype="int32"))

    # Perform inference
    try:
        response = triton_client.infer(model_name="florence2base", inputs=inputs, outputs=outputs)

        # Get the result
        answer_str = response.as_numpy("answer")[0]

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