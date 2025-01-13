import logging
import HyperParameters as hp
from collections import OrderedDict

def run_model(model, processor, task_prompt, image, text_input=None):
    """
    takes in a task prompt and image, returns an answer 
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt")

    image_height, image_width = image.shape[:2]

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
    
def generate_caption(model, processor, image):
    """
    Generate image caption using the provided model
    """
    task_prompt = '<MORE_DETAILED_CAPTION>'

    description_text = run_model(model, processor, task_prompt, image)
    description_text = description_text[task_prompt]

    #takes those details from the setences and finds labels and boxes in the image
    task_prompt = '<CAPTION_TO_PHRASE_GROUNDING>'
    boxed_descriptions = run_model(model, processor, task_prompt, image, description_text)

    #only prints out labels not bboxes
    descriptions = boxed_descriptions[task_prompt]['labels']
    logging.info(descriptions)

    #finds other things in the image that the description did not explicitly say
    task_prompt = '<DENSE_REGION_CAPTION>'
    labels = run_model(model, processor, task_prompt, image)

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

    logging.debug(final_description)
    return final_description