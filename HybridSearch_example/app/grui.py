'''This file contains the code to generate the gradio app'''
#NOTE: This is will be replaced with our UI in k8s namespace beekeeper.
#   The UI will use the new queries added to our data API to do the same thing this file is doing.

import gradio as gr
import os
import weaviate
import argparse
import logging
import time
from apscheduler.schedulers.background import BackgroundScheduler
from data import load_data, clear_data, check_data, continual_load
from query import testText
import tritonclient.grpc as TritonClient

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

#Continual Loading?
CONT_LOAD = os.environ.get("CONTINUAL_LOADING", "false").lower() == "true"

#Creds
USER = os.environ.get("SAGE_USER")
PASS = os.environ.get("SAGE_PASS")

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

def text_query(description): #TODO: return the links as well
    '''
    Send text query to testText() and engineer results to display in Gradio
    '''
    dic = testText(description, weaviate_client)
    text_results = dic['objects']
    certainty = dic['scores']
    
    # Extract image links with captions from text_results
    images = [
        (f"{IMAGE_DIR}/{obj['filename']}", f"{obj['filename']}: {obj['caption']}")  # Tuple of image link and caption to work with gradio gallery component
        for obj in text_results
        if any(obj["filename"].endswith(ext) for ext in [".jfif", ".jpg", ".jpeg", ".png"])
    ]

    return images, certainty

def set_query(username, token, query):
    '''
    load sage data to IMAGE_DIR and return 'Images Loaded'
    '''
    load_data(username, token, query, weaviate_client, IMAGE_DIR)
    return "Images Loaded"

def rm_data():
    '''
    Clear IMAGE_DIR data and return 'Empty'
    '''
    clear_data(IMAGE_DIR)
    return "Empty"

# Gradio Interface Setup
def load_interface():
    '''
    Configure Gradio interface
    '''
    #set blocks
    iface_load_data = gr.Blocks()
    iface_text_description = gr.Blocks()
    iface_upload_image = gr.Blocks()

    # load sage data tab
    with iface_load_data:
        # set title and description
        gr.Markdown(
        """
        # Load Sage Data
        Upload data from Sage to be vectorized with ImageBind and captioned using Florence 2.
        """)

        #set default code 
        def_query = '''
        df = sage_data_client.query(
            start="-24h",
            #end="2023-02-22T23:00:00.000Z",
            filter={
                "plugin": "registry.sagecontinuum.org/theone/imagesampler.*",
                "vsn": "W088"
                #"job": "imagesampler-top"
            }
        ).sort_values('timestamp')
        '''

        #set cred inputs
        username = gr.Textbox(label="Username",max_lines=1)
        token = gr.Textbox(label="Token",max_lines=1)

        #set code input
        gr.Markdown(
        """
        Enter your Sage query as Python code (output results to df)
        """)
        query = gr.Code(label="Sage Data Client Query",value=def_query,language='python')

        #set button row
        with gr.Row():
            load_btn = gr.Button("Load Data")
            clear_btn = gr.Button("Clear Data")

        #set image dir indicator
        indicator = gr.Textbox(label="Image Directory Status", value=check_data(IMAGE_DIR), max_lines=1)
        
        #set event listeners
        inputs = [username, token, query]
        load_btn.click(fn=set_query, inputs=inputs, outputs=indicator)
        clear_btn.click(fn=rm_data, outputs=indicator)

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
        certainty = gr.JSON(label="Certainty Scores")
        gr.Markdown(
        """
        Images Returned
        """)
        gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")

        #clear function
        def clear():
            return ""

        #set event listeners
        sub_btn.click(fn=text_query, inputs=query, outputs=[gallery, certainty])
        clear_btn.click(fn=clear, outputs=query)

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

    if CONT_LOAD:
        iface = gr.TabbedInterface(
            [iface_text_description],
            ["Text Query"]
        )
    else:
        iface = gr.TabbedInterface(
            [iface_load_data, iface_text_description],
            ["Load Data", "Text Query"]
            # [iface_load_data, iface_text_description, iface_upload_image], Implement image_query() first
            # ["Load Data", "Text Query", "Image Query"] #TODO: 
        ) 
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

def run_continual_load():
    '''
    Run the continual loading function in the background
    '''
    # Initiate Triton client
    triton_client = TritonClient.InferenceServerClient(url="florence2:8001")

    # Start continual loading
    continual_load(USER, PASS, weaviate_client, triton_client)

def main():
    # Initialize the background scheduler
    scheduler = BackgroundScheduler()

    # Schedule the continual_load function to run in the background if CONT_LOAD is enabled
    if CONT_LOAD:
        scheduler.add_job(run_continual_load)  # Run continously

    # Start the scheduler to run jobs in the background
    scheduler.start()

    #start gradio app
    load_interface()

if __name__ == "__main__":
    main()
