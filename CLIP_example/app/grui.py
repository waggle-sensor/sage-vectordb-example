import gradio as gr
import os
import weaviate
import argparse
import shutil
from test import testImage, testText
from data import load_data, clear_data, check_data

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','jfif'])
IMAGE_DIR = os.path.join(os.getcwd(), "static", "Images")
UPLOAD_DIR = os.path.join(os.getcwd(), "static", "uploads")

def initialize_weaviate_client():
    '''
    Intialize weaviate client based on arg or env var
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weaviate",
        default=os.getenv("WEAVIATE_API"),
        help="Weaviate REST endpoint.",
    )
    args = parser.parse_args()
    return weaviate.Client(args.weaviate)

client = initialize_weaviate_client()

def allowed_file(filename):
    '''
    Check if file is allowed
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_query(file):
    '''
    Send image query to testImage() and engineer results to display in Gradio
    '''
    # Get the full path of the temporary file
    temp_file_path = file.name
    
    # Define the destination file path
    full_path = os.path.join(UPLOAD_DIR, os.path.basename(temp_file_path))
    
    # Move the temporary file to the destination directory
    shutil.move(temp_file_path, full_path)
    
    dic = testImage({"image": f"{full_path}"}, client)
    image_paths = dic['objects']
    certainty = dic['scores']

    images = [f'{IMAGE_DIR}/{i}' for i in image_paths if any(ext in i for ext in ['.jfif','.jpg','.jpeg','.png'])]

    return images, certainty

def text_query(description):
    '''
    Send text query to testText() and engineer results to display in Gradio
    '''
    dic = testText({"concepts": [description]}, client)
    text_results = dic['objects']
    certainty = dic['scores']
    
    images = [f'{IMAGE_DIR}/{i}' for i in text_results if any(ext in i for ext in ['.jfif','.jpg','.jpeg','.png'])]
    #texts = [i for i in text_results if i not in images] #TODO: enable text to be returned as well

    return images, certainty

def set_query(username, token, query):
    '''
    load sage data to IMAGE_DIR and return 'Images Loaded'
    '''
    load_data(username, token, query, client, IMAGE_DIR)
    return "Images Loaded"

def rm_data():
    '''
    Clear IMAGE_DIR data and return 'Empty'
    '''
    clear_data(IMAGE_DIR)
    return "Empty"

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
        Upload data from Sage to vectorize and use with CLIP
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
        certainty = gr.Textbox(label="Certainty Scores")
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
    with iface_upload_image:

        # set title and description
        gr.Markdown(
        """
        # Image Query
        Upload an image to find similar images.
        """)

        #set inputs
        query = gr.File(label="Upload Image", file_types=['png', 'jpg', 'jpeg', 'gif','jfif'])

        #set buttons
        with gr.Row():
            sub_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        #set Outputs
        certainty = gr.Textbox(label="Certainty Scores")
        gr.Markdown(
        """
        Images Returned
        """)
        gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")

        #clear function
        def clear():
            return None

        #set event listeners
        sub_btn.click(fn=image_query, inputs=query, outputs=[gallery, certainty])
        clear_btn.click(fn=clear, outputs=query)

    iface = gr.TabbedInterface(
        [iface_load_data, iface_text_description, iface_upload_image],
        ["Load Data", "Text Query", "Image Query"]
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    load_interface()