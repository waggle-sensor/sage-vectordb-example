'''This file contains the code to generate the gradio app'''
#NOTE: This is will be replaced with our UI in k8s namespace beekeeper.
#   The UI will use the new queries added to our data API to do the same thing this file is doing.

import gradio as gr
import os
import weaviate
import argparse
import logging
import time
import plotly.graph_objects as go
from query import testText, getImage
from gradio import ChatMessage
import asyncio
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType

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

#set up args
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
parser.add_argument(
    "--ollama_host",
    default=os.getenv("OLLAMA_HOST","127.0.0.1"),
    help="Ollama host IP.",
)
parser.add_argument(
    "--ollama_port",
    default=os.getenv("OLLAMA_PORT","11434"),
    help="Ollama host IP.",
)
args = parser.parse_args()

def allowed_file(filename):
    '''
    Check if file is allowed
    '''
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_weaviate_client(args):
    '''
    Intialize weaviate client based on arg or env var
    '''

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

weaviate_client = initialize_weaviate_client(args)

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

def filter_map(df):
    '''
    This function generates a map based on the results from text_query
    '''
    #set up custom data
    uuid = df["uuid"].tolist()
    address = df["address"].tolist() 
    score = df["score"].tolist()
    rerank_score = df["rerank_score"].tolist()
    data = [(uuid[i], address[i], score[i], rerank_score[i]) for i in range(0, len(uuid))]

    # Create the plot
    fig = go.Figure(go.Scattermapbox(
        lat=df['location_lat'].tolist(),
        lon=df['location_lon'].tolist(),
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=6
        ),
        hoverinfo="text",
        hovertemplate='<b>uuid</b>: %{customdata[0]}<br><b>address</b>: %{customdata[1]}<br><b>score</b>: %{customdata[2]}<br><b>rerank_score</b>: %{customdata[3]}',
        customdata=data
    ))

    fig.update_layout(
        mapbox_style="open-street-map",
        hovermode='closest',
        mapbox=dict(
            bearing=0,
            center=go.layout.mapbox.Center(
                lat=39.00,
                lon=-98.00
            ),
            pitch=0,
            zoom=1
        ),
    )

    return fig

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
                images.append((image, f"{row['uuid']}"))

    #get location details
    location = df[['location_lat', 'location_lon', 'uuid', 'score', 'rerank_score', 'address']]
    map_fig = filter_map(location)

    #drop columns that I dont want to show
    meta = df.drop(columns=["link", "node", "location_lat", "location_lon"])

    # Return the images, DataFrame, and map
    return images, meta, map_fig

# -------------------------------
# This section contains the code that wraps the image search as a tool for an LLM agent 
# TODO: Move to another py file called agent.py
# -------------------------------

def image_search_tool(query: str) -> str:
    """
    This tool calls the image search functionality (text_query) and returns a textual summary
    of the results.
    """
    # Call the image search function with the provided query
    images, meta, map_fig = text_query(query)
    
    # Create a textual summary
    if len(images) == 0:
        return "No images found for the query."
    
    summary = f"Found {len(images)} images.\n"
    summary += "metadata:\n"
    # Convert metadata DataFrame to CSV
    summary += meta.to_csv(index=False)
    
    # (Optionally, you could save the map or gallery images and provide links.)
    return summary

# Create a LangChain Tool for image search
tools = [
    Tool(
        name="ImageSearch",
        func=image_search_tool,
        description=(
            "Use this tool to search for images based on a descriptive text query."
            "Input should be a plain text description of what images to find."
            "The tool uses a hybrid approach to find images based on a descriptive text query."
            "The hybrid approach incorporates vector search and a keyword (BM25F) search to find images."
            "This tool uses waeviate as the backend for vector storage and Image retrieval."
            "The keyword search in the tool will look in the fields called caption, camera, host, job, vsn, plugin, zone, project, and address."
            "The vector search in the tool will look in the fields called image and caption."
            "Weaviate will use Imagebind to generated vectors for images and texts."
        )
    )
]

# Define a custom prompt for better instruction
custom_prompt = PromptTemplate.from_template(
    """
    You are an AI agent that follows a ReAct-style format.
    You can use tools when needed.

    Format:
    Thought: [Describe what you are thinking]
    Action: [Select the tool to use]
    Action Input: [Provide input to the tool]

    Respond ONLY in this format.

    Question: {input}
    """
)

# Initialize the LLM (Ollama)
ollama_host= args.ollama_host
ollama_port= args.ollama_port
llm = OllamaLLM(model="llama3", base_url=f"http://{ollama_host}:{ollama_port}", temperature=0)

# Create the agent using a zero-shot chain that reacts to descriptions.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"prompt": custom_prompt},
    handle_parsing_errors=True)
agent_executor = AgentExecutor(agent=agent, tools=tools).with_config(
    {"run_name": "Agent"}
)

def llm_agent_interface(user_query: str) -> str:
    """
    This function receives a user query, lets the LLM agent decide how to use the tool,
    and returns the LLM’s response.
    """
    response = agent.run(user_query)
    return response

# Asynchronous generator function for interacting with the agent.
async def interact_with_image_search_agent(prompt, messages):
    # Append the user's message to the conversation.
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages

    #start action stream
    async for chunk in agent_executor.astream(
        {"input": prompt}
    ):
        if "steps" in chunk:
            for step in chunk["steps"]:
                messages.append(ChatMessage(role="assistant", content=step.action.log,
                                  metadata={"title": f"🛠️ Used tool {step.action.tool}"}))
                yield messages
        if "output" in chunk:
            messages.append(ChatMessage(role="assistant", content=chunk["output"]))
            yield messages

# -------------------------------
# End of agent section
# -------------------------------

# Gradio Interface Setup
def load_interface():
    '''
    Configure Gradio interface
    '''
    #set blocks
    iface_text_description = gr.Blocks()
    iface_upload_image = gr.Blocks()
    iface_agent = gr.Blocks()

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

        #Give examples
        queries=[["Show me images in Hawaii"], 
                 ["Rainy Chicago"], 
                 ["Snowy Mountains"], 
                 ["Show me clouds in the top camera"],
                 ["Cars in W049"],
                 ["W040"],
                 ["intersection in the right camera"]]
        examples = gr.Dataset(label="Example Queries", components=[query], samples=queries)

        #set buttons
        with gr.Row():
            sub_btn = gr.Button("Submit")
            clear_btn = gr.Button("Clear")

        #set Outputs
        gr.Markdown(
        """
        Images Returned
        """)
        col_widths = [
            "350px", #uuid
            "120px", #filename
            "1100px", #caption
            "180px", #score
            "1100px", #explain score
            "180px", #rerank score
            "90px", #vsn
            "150px", #camera
            "110px", #project
            "160px", #timestamp
            "250px", #host
            "250px", #job
            "500px", #plugin
            "220px", #task
            "100px", #zone
            "450px", #address
        ]
        gallery = gr.Gallery( label="Returned Images", columns=[3], object_fit="contain", height="auto")
        meta = gr.DataFrame(label="Metadata", show_fullscreen_button=True, show_copy_button=True, column_widths=col_widths)
        plot = gr.Plot(label="Image Locations")

        #clear function
        def clear():
            return "", [], gr.DataFrame(value=None), gr.Plot(value=None)
        
        #select example func
        def on_select(evt: gr.SelectData):
            return evt.value[0]

        #set event listeners
        sub_btn.click(fn=text_query, inputs=query, outputs=[gallery, meta, plot])
        clear_btn.click(fn=clear, outputs=[query, gallery, meta, plot])  # Clear all components
        examples.select(fn=on_select, outputs=query)
        gr.SelectData

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

    # image search llm agent tab
    with iface_agent:

        # set title and description
        gr.Markdown(
        """
        # Image Search AI Agent
        Chat with me to help you find images in Sage.
        """)
        
        # Chatbot component for conversation display.
        chatbot = gr.Chatbot(
            type="messages",
            label="Sage Agent",
            # avatar_images=(
            #     None,
            #     "https://example.com/agent_avatar.png"
            # ),
        )
        
        # Textbox for user input.
        user_input = gr.Textbox(lines=1, label="Chat Message")
        
        # When the user submits text, call the async chat function.
        user_input.submit(interact_with_image_search_agent, [user_input, chatbot], [chatbot])


    iface = gr.TabbedInterface(
        [iface_text_description, iface_agent],
        ["Text Query", "Sage Agent"]
        # [iface_text_description, iface_upload_image], Implement image_query() first
        # ["Text Query", "Image Query"] #TODO
    )
    
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)

def main():

    #start gradio app
    load_interface()

if __name__ == "__main__":
    main()
