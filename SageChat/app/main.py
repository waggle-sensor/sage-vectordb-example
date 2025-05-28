import os
import weaviate
import logging
import argparse
import time
import pprint
import requests
from urllib.parse import urljoin
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from query import testText, getImage
import HyperParameters as hp
import gradio as gr
import sage_data_client as sdc
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

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

# ==============================
# Define Node search tool.
# ==============================
@tool
def node_search_tool(vsn: str) -> str:
    """
    Call to do a search on devices called nodes. the nodes ID called vsn are in W[1-9] format.
    The response is the final answer.

    args:
        vsn: str: The vsn of the node to search for.
    returns:
        str: The details of the node includes the node's hardware, sensors, devices, capabilities, and other metadata.
    """
    try:
        MANIFEST_API = os.environ.get("MANIFEST_API", "https://auth.sagecontinuum.org/manifests/")
        response = requests.get(urljoin(MANIFEST_API, vsn.upper()))
        response.raise_for_status()  # Raise error for bad responses
    except requests.exceptions.HTTPError as e:
        logging.debug(f"Node search failed, HTTPError: {e}")
        return f"Error: HTTP error occurred: {e}"
    except requests.exceptions.RequestException as e:
        logging.debug(f"Node search failed, request exception: {e}")
        return f"Error: Request exception occurred: {e}"
    except Exception as e:
        logging.debug(f"Node search failed, unknown error: {e}")
        return f"Error: An unexpected error occurred: {e}"
    
    manifest = response.json()
    formatted = []
    
    # Basic Node Information
    formatted.append(f"**Node Manifest for {manifest.get('vsn', 'N/A')}**")
    formatted.append(f"- **Name:** {manifest.get('name', 'N/A')}")
    formatted.append(f"- **Phase:** {manifest.get('phase', 'N/A')}")
    formatted.append(f"- **Project:** {manifest.get('project', 'N/A')}")
    formatted.append(f"- **Address:** {manifest.get('address', 'N/A')}")
    
    # Computes
    formatted.append("\n**Computes:**")
    computes = manifest.get("computes", [])
    if computes:
        for comp in computes:
            hw = comp.get("hardware", {})
            formatted.append(f"  - **Name:** {comp.get('name', 'N/A')}, **Serial:** {comp.get('serial_no', 'N/A')}, **Zone:** {comp.get('zone', 'N/A')}")
            formatted.append(f"    - **Hardware:** {hw.get('hardware', 'N/A')} (Model: {hw.get('hw_model', 'N/A')}, Version: {hw.get('hw_version', 'N/A')})")
            formatted.append(f"    - **Manufacturer:** {hw.get('manufacturer', 'N/A')}")
            formatted.append(f"    - **Datasheet:** {hw.get('datasheet', 'N/A')}")
            caps = hw.get("capabilities", [])
            if caps:
                formatted.append(f"    - **Capabilities:** {', '.join(caps)}")
    else:
        formatted.append("  - No compute information available.")
    
    # Sensors
    formatted.append("\n**Sensors:**")
    sensors = manifest.get("sensors", [])
    if sensors:
        for sensor in sensors:
            sensor_hw = sensor.get("hardware", {})
            formatted.append(f"  - **Name:** {sensor.get('name', 'N/A')}, **Active:** {sensor.get('is_active', False)}")
            formatted.append(f"    - **Hardware:** {sensor_hw.get('hardware', 'N/A')} (Model: {sensor_hw.get('hw_model', 'N/A')})")
    else:
        formatted.append("  - No sensor information available.")
    
    # Resources
    formatted.append("\n**Resources:**")
    resources = manifest.get("resources", [])
    if resources:
        for res in resources:
            res_hw = res.get("hardware", {})
            formatted.append(f"  - **Name:** {res.get('name', 'N/A')}")
            formatted.append(f"    - **Hardware:** {res_hw.get('hardware', 'N/A')} (Model: {res_hw.get('hw_model', 'N/A')})")
            formatted.append(f"    - **Datasheet:** {res_hw.get('datasheet', 'N/A')}")
    else:
        formatted.append("  - No resource information available.")
    
    # LoRaWAN Connections
    formatted.append("\n**LoRaWAN Connections:**")
    lorawan = manifest.get("lorawanconnections", [])
    if lorawan:
        for conn in lorawan:
            connection_name = conn.get("connection_name", "N/A")
            created_at = conn.get("created_at", "N/A")
            last_seen_at = conn.get("last_seen_at", "N/A")
            margin = conn.get("margin", "N/A")
            uplink_interval = conn.get("expected_uplink_interval_sec", "N/A")
            connection_type = conn.get("connection_type", "N/A")
            formatted.append(f"  - **Connection Name:** {connection_name}")
            formatted.append(f"    - **Created At:** {created_at}")
            formatted.append(f"    - **Last Seen At:** {last_seen_at}")
            formatted.append(f"    - **Margin:** {margin}")
            formatted.append(f"    - **Expected Uplink Interval (sec):** {uplink_interval}")
            formatted.append(f"    - **Connection Type:** {connection_type}")
            # Format the LoRaWAN device details if available
            device = conn.get("lorawandevice", {})
            if device:
                dev_name = device.get("name", "N/A")
                is_active = device.get("is_active", False)
                battery = device.get("battery_level", "N/A")
                hw = device.get("hardware", {})
                hw_model = hw.get("hw_model", "N/A")
                hw_version = hw.get("hw_version", "N/A")
                manufacturer = hw.get("manufacturer", "N/A")
                datasheet = hw.get("datasheet", "N/A")
                formatted.append(f"    - **Device Name:** {dev_name}")
                formatted.append(f"      - **Active:** {is_active}, **Battery:** {battery}")
                formatted.append(f"      - **Hardware Model:** {hw_model}, **Version:** {hw_version}")
                formatted.append(f"      - **Manufacturer:** {manufacturer}")
                formatted.append(f"      - **Datasheet:** {datasheet}")
    else:
        formatted.append("  - No LoRaWAN connections")

    # Join all formatted lines into a single string.
    final_formatted = "\n".join(formatted)
    return final_formatted

# ==============================
# Define image search tool.
# ==============================
@tool
def image_search_tool(query: str) -> str:
    """
    Call to do an image search.
    always give the user the link.
    The response is the final answer.

    args:
        query: str: The image query to search for.
    returns:
        str: details of the image search including metadata and links to the images.
    """
    # Retrieve the dataframe from your text search function.
    df = testText(query, weaviate_client)
    
    # Extract image data
    images = []
    for _, row in df.iterrows():
        if any(row["filename"].endswith(ext) for ext in [".jfif", ".jpg", ".jpeg", ".png"]):
            image = getImage(row['link'])
            if image:
                images.append((image, f"{row['uuid']}"))
    
    # Create metadata dataframe (dropping unwanted columns)
    meta = df.drop(columns=["node"])
    
    # Build the summary
    if not images or len(images) == 0:
        return f"No images found for the query {query}, you may not have access to the images."
    
    summary = f"Found {len(images)} images with the query being {query}.\n\n"
    summary += "### Image Metadata:\n"
    if not meta.empty:
        try:
            # Try to convert the metadata dataframe to a Markdown table.
            formatted_meta = meta.to_markdown(index=False)
        except Exception as e:
            logging.debug(f"Markdown conversion failed: {e}. Falling back to CSV.")
            formatted_meta = meta.to_csv(index=False, sep="|")
        summary += formatted_meta
    else:
        summary += "No metadata available.\n"
    
    summary += "\n### Image Links:\n"
    for idx, (image, uuid) in enumerate(images):
        if "link" in meta.columns:
            summary += f"Image {uuid}: {meta.iloc[idx]['link']}\n"
        else:
            summary += f"Image {uuid}: No link available\n"
    
    return f"{summary}"

# ==============================
# Define measurment name search tool.
# ==============================
@tool
def get_measurement_name_tool(vsn: str, time: str) -> str:
    """
    Call to find measurement names based on time in a node. the nodes ID called vsn are in W[1-9] format.
    The time is in the format of -[number][unit] where unit is m for minutes, h for hours, d for days, and w for weeks.
    The response is the final answer.
    args:
        vsn: str: The vsn of the node to search for.
        time: str: The time to search for.
    returns:
        str: this list of measurement names being collected in the node.
    """
    df = sdc.query(
        start=time,
        filter={
            "vsn": vsn
        }
    )

    # Extract the 'name' column as a list
    names = df["name"].unique().tolist()

    # Generate a summary
    summary = f"# These are the measurements being collected in {vsn} in {time}.\n"
    if names:
        for name in names:
            summary += f"- {name}\n"
    else:
        summary += f"No measurements found for {vsn}"

    return f"{summary}"

# ==============================
# Define measurement value search tool.
# ==============================
@tool
def get_measurement_values_tool(vsn: str, measurement_name:str, time: str) -> str:
    """
    Call to get measurement values based on time and measurement name in a node. the nodes ID called vsn are in W[1-9] format.
    The time is in the format of -[number][unit] where unit is m for minutes, h for hours, d for days, and w for weeks.
    To join the measurement name use a | between the names.
    The response is the final answer.
    args:
        vsn: str: The vsn of the node to search for.
        measurement_name: str: The name of the measurement to search for.
        time: str: The time to search for.
    returns:
        str: the measurement values being collected in the node.
    """
    df = sdc.query(
        start=time,
        filter={
            "vsn": vsn,
            "name": measurement_name
        }
    )

    # Generate a summary
    summary = f"# The measurement values for {measurement_name} in {vsn} in {time}:\n"
    if not df.empty:
        summary += df.to_markdown(index=False)
    else:
        summary += f"No measurements values found for {measurement_name} in {vsn} in {time}"

    return f"{summary}"

# ==============================
# MCP Integration for ToolNode
# ==============================
# Retrieve tools from MCP server wrapping local tool APIs
client = MultiServerMCPClient({
    "sage_tools": {
        "url": os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp"),
        "transport": "streamable_http",
    }
})
# Asynchronously fetch the MCP tools
tools = asyncio.run(client.get_tools())
tool_node = ToolNode(tools)

# ==============================
# Set up the LLM(s).
# ==============================
# Initialize the LLM (Ollama) and bind tools with llm
ollama_host= args.ollama_host
ollama_port= args.ollama_port
model = ChatOllama(
    model=hp.model,
    base_url=f"http://{ollama_host}:{ollama_port}", 
    temperature=0, 
    verbose=True).bind_tools(tools)

# ==============================
# Define a system prompt that tells the agent who it is.
# ==============================
sys_msg = SystemMessage(hp.SYSTEM_PROMPT)

# ==============================
# Define the function that calls the LLM.
# ==============================
def call_model(state: MessagesState):
    return {"messages": [model.invoke([sys_msg] + state["messages"])]}
    
# ==============================
# Build the state graph.
# ==============================

# init workflow
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model) #model node
workflow.add_node("tools", tool_node) #tool node

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We add a conditional edge from `agent` to `tools`.
workflow.add_conditional_edges(
    "agent",
    tools_condition
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')


# ==============================
# Compile the graph with memory.
# ==============================

# Initialize memory to persist state between graph runs
memory = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=memory)

config = {"recursion_limit": hp.recursion_limit, "configurable": {"thread_id": 42}}

# ==============================
# Define the Gradio chat function.
# ==============================
def chat(message, history):
    messages= []
   # Create an initial ChatMessage that will hold the intermediate “thinking” text.
    thinking_msg = gr.ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "Thinking...", "status": "pending"}
    )
    messages.append(thinking_msg)
    yield messages

    # Start a new conversation with a single human message.
    input_messages = [HumanMessage(message)]
    output = app.invoke({"messages": input_messages}, config)

    # Mark the thinking process as done.
    thinking_msg.metadata["status"] = "done"

    done_msg = gr.ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "Done", "status": "done"}
    )
    messages.append(done_msg)

    # Return the content of the last message as the final answer.
    final_msg = gr.ChatMessage(
        role="assistant",
        content=output['messages'][-1].content
    )
    messages.append(final_msg)
    yield messages

async def stream_chat(message, history):
    messages= []

    # Create an initial ChatMessage that will hold the intermediate “thinking” text.
    thinking_msg = gr.ChatMessage(
        role="assistant",
        content="",
        metadata={"title": "Thinking...", "status": "pending"}
    )
    messages.append(thinking_msg)
    yield messages

    # Create an init Chatmessage that will hold the response
    final_msg = gr.ChatMessage(
        role="assistant",
        content=""
    )
    # Start a new conversation with a single human message.
    input_messages = [HumanMessage(message)]

    # Start the stream.
    #https://langchain-ai.github.io/langgraph/concepts/streaming/#streaming-llm-tokens-and-events-astream_events
    async for event in app.astream_events({"messages": input_messages}, config, version="v1"):
        logging.debug(pprint.pformat(event))
        data = event.get("data", {})

        # If there's a chunk in the event, update the thinking message.
        if "chunk" in data:
            chunk = data["chunk"]
            if hasattr(chunk, "content"):
                # Append the chunk's text to the thinking message.
                final_msg.content += chunk.content
                final_msg.metadata["last_event"] = event.get("event", "")

        # Optionally, if the event indicates a tool usage:
        if "langgraph_node" in event.get("metadata", {}):
            node_name = event["metadata"].get("langgraph_node")
            if node_name and "tool" in node_name.lower():
                output = data.get("output")
                if output and hasattr(output, "content"):
                    tool_msg = gr.ChatMessage(
                        role="assistant",
                        content=output.content,
                        metadata={"title": f"{node_name} output", "status": "done"}
                    )
                    messages.append(tool_msg)
                    yield messages

    # send last message and Mark the thinking message as done.
    thinking_msg.metadata["status"] = "done"
    messages.append(final_msg)
    yield messages

# ==============================
# Set up the Gradio ChatInterface.
# ==============================
examples=[
    {"text": "Hello, who are you?"},
    {"text": "Find images of Rainy Chicago and give me the links"},
    {"text": "Look for images of Snow Mountains and give me the links"},
    {"text": "Look for images of clouds in the top camera and give me the links"},
    {"text": "Show me images of Cars in W049 and give me the links"},
    {"text": "Show me images of an intersection in the right camera and give me the links"},
    {"text": "Tell me what devices are in node W049"},
    {"text": "is there lorawan connections in W08E"},
    {"text": "I want to use node W07E for computer vision tasks is that possible?"}, 
    {"text": "what project do I need to be in to get access to W0A4?"},
    {"text": "what measurements are being collected in W049?"},
    {"text": "What are the env.temperature in W049 in the last 30 minutes?"}]

demo = gr.ChatInterface(
    fn=stream_chat,
    type="messages",
    examples=examples,
    title="Sage Agent",
    save_history=True,
    show_progress='full',
    stop_btn=True,
    submit_btn=True,
    # multimodal=True,  # Uncomment if you plan to display images or other media. TODO: enable, llama 3.2 is multimodal
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)