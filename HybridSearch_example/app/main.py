'''This file contains the code to generate the gradio app'''
#NOTE: This is will be replaced with our UI in k8s namespace beekeeper.
#   The UI will use the new queries added to our data API to do the same thing this file is doing.

import gradio as gr
import os
import weaviate
import argparse
import logging
import time
from query import testText, getImage

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
IMAGE_DIR = os.path.join(os.getcwd(), "static", "Images")
UPLOAD_DIR = os.path.join(os.getcwd(), "static", "uploads")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
)

def allowed_file(filename):
    '''
    Check if file is allowed
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_weaviate_client():
    '''
    Intialize weaviate client based on arg or env var
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaviate_host",
        default=os.getenv("WEAVIATE_HOST","127.0.0.1"),
        help="Weaviate host IP.",
    )
    parser.add_argument(
        "--weaviate_port",
        default=os.getenv("WEAVIATE_PORT","8080"),
        help="Weaviate REST port.",
    )
    parser.add_argument(
        "--weaviate_grpc_port",
        default=os.getenv("WEAVIATE_GRPC_PORT","50051"),
        help="Weaviate GRPC port.",
    )
    args = parser.parse_args()

    weaviate_host = args.weaviate_host
    weaviate_port = args.weaviate_port
    weaviate_grpc_port = args.weaviate_grpc_port

    logging.debug(f"Attempting to connect to Weaviate at {weaviate_host}:{weaviate_port}")

    # Retry logic to connect to Weaviate
    while True:
        try:
            client = weaviate.connect_to_local(
                host=weaviate_host,
                port=weaviate_port,
                grpc_port=weaviate_grpc_port
            )
            logging.debug("Successfully connected to Weaviate")
            return client
        except weaviate.exceptions.WeaviateConnectionError as e:
            logging.error(f"Failed to connect to Weaviate: {e}")
            logging.debug("Retrying in 10 seconds...")
            time.sleep(10)

weaviate_client = initialize_weaviate_client()

# TODO: implement testImage() first
# def image_query(file):
#     '''
#     Send image query to testImage() and engineer results to display in Gradio
#     '''
#     # Get the full path of the temporary file
#     temp_file_path = file.name
    
#     # Define the destination file path
#     full_path = os.path.join(UPLOAD_DIR, os.path.basename(temp_file_path))
    
#     # Move the temporary file to the destination directory
#     shutil.move(temp_file_path, full_path)
    
#     dic = testImage({"image": f"{full_path}"}, client)
#     image_paths = dic['objects']
#     certainty = dic['scores']

#     images = [f'{IMAGE_DIR}/{i}' for i in image_paths if any(ext in i for ext in ['.jfif','.jpg','.jpeg','.png'])]

#     return images, certainty

def text_query(description):
    '''
    Send text query to testText() and engineer results to display in Gradio
    '''
    # Get the DataFrame from the testText function
    df = testText(description, weaviate_client)
    
    # Extract the image links and captions from the DataFrame
    images = []
    for _, row in df.iterrows():  # Iterate through the DataFrame rows
        if any(row["filename"].endswith(ext) for ext in [".jfif", ".jpg", ".jpeg", ".png"]):
            # Use getImage to retrieve the image from the URL
            image = getImage(row['link'])
            if image:
                images.append((image, f"{row['uuid']}: {row['caption']}"))

    #drop columns that I dont want to show
    meta = df.drop(columns=["caption", "link", "node"])

    # Return the images along with the entire DataFrame
    return images, meta

# Gradio Interface Setup
def load_interface():
    '''
    Configure Gradio interface
    '''
    #set blocks
    iface_text_description = gr.Blocks()
    iface_upload_image = gr.Blocks()

    # text query tab
    with iface_text_description:

        # set title and description
        gr.Markdown(
        """
        # Text Query
        Enter a text description to find similar images.
        """)

        #set inputs
        query = gr.Textbox(label="Text Query", interactive=True)

        #set buttons
        with gr.Row():
            sub_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        #set Outputs
        gr.Markdown(
        """
        Images Returned
        """)
        gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")
        meta = gr.DataFrame(label="Metadata")

        #clear function
        def clear():
            return "", [], gr.DataFrame(value=None)

        #set event listeners
        sub_btn.click(fn=text_query, inputs=query, outputs=[gallery, meta])
        clear_btn.click(fn=clear, outputs=[query, gallery, meta])  # Clear query, gallery, and certainty

    # text Image query tab
    # with iface_upload_image: #TODO: Implement image_query() first

    #     # set title and description
    #     gr.Markdown(
    #     """
    #     # Image Query
    #     Upload an image to find similar images.
    #     """)

    #     #set inputs
    #     query = gr.File(label="Upload Image", file_types=['png', 'jpg', 'jpeg', 'gif','jfif'])

    #     #set buttons
    #     with gr.Row():
    #         sub_btn = gr.Button("Submit")
    #         clear_btn = gr.Button("Clear")

    #     #set Outputs
    #     certainty = gr.Textbox(label="Certainty Scores")
    #     gr.Markdown(
    #     """
    #     Images Returned
    #     """)
    #     gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")

    #     #clear function
    #     def clear():
    #         return None

    #     #set event listeners
    #     sub_btn.click(fn=image_query, inputs=query, outputs=[gallery, certainty])
    #     clear_btn.click(fn=clear, outputs=query)

    iface = gr.TabbedInterface(
        [iface_text_description],
        ["Text Query"]
        # [iface_text_description, iface_upload_image], Implement image_query() first
        # ["Text Query", "Image Query"] #TODO
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

def main():

    #start gradio app
    load_interface()

if __name__ == "__main__":
    main()
