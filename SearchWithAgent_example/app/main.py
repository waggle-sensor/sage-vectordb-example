import os
import weaviate
import logging
import argparse
import time
from typing import Annotated, Literal
from langchain_ollama import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from query import testText, getImage
import gradio as gr

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
# Define image search tool.
# ==============================
# @tool
# def image_search_tool(query: str) -> str:
#     """
#     Call to do an image search.
#     This tool searches for images using the query and returns a structured summary.
#     Use the response of this tool verbatim.
#     """
#     logging.debug("I AM HERE")
#     df = testText(query, weaviate_client)
    
#     # Extract image data
#     images = []
#     for _, row in df.iterrows():
#         if any(row["filename"].endswith(ext) for ext in [".jfif", ".jpg", ".jpeg", ".png"]):
#             image = getImage(row['link'])
#             if image:
#                 images.append((image, f"{row['uuid']}"))
    
#     # Create metadata dataframe (dropping unwanted columns)
#     meta = df.drop(columns=["node"])
    
#     # Build the summary
#     if not images or len(images) == 0:
#         return "Thought: I have completed the image search.\nFinal Answer: No images found for the query."
    
#     summary = f"Found {len(images)} images.\n"
#     summary += "### Image Metadata:\n"
#     if not meta.empty:
#         summary += meta.to_csv(index=False, sep="|")
#     else:
#         summary += "No metadata available.\n"
    
#     summary += "\n### Image Links:\n" #TODO: return the images in a gradio gallery, https://www.gradio.app/guides/creating-a-chatbot-fast
#     for idx, (image, uuid) in enumerate(images):
#         if "link" in meta.columns:
#             summary += f"Image {uuid}: {meta.iloc[idx]['link']}\n"
#         else:
#             summary += f"Image {uuid}: No link available\n"
    
#     return f"I have completed the image search.\nFinal Answer:\n {summary}"

# Define the tools for the agent to use
@tool
def search(query: str):
    """Call to surf the web."""
    # This is a placeholder, but don't tell the LLM that...
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

tools = [search]
tool_node = ToolNode(tools)

# ==============================
# Set up the LLM.
# ==============================
# Initialize the LLM (Ollama) and bind tools with llm
ollama_host= args.ollama_host
ollama_port= args.ollama_port
model = ChatOllama(
    model="llama3-groq-tool-use", #https://groq.com/introducing-llama-3-groq-tool-use-models/
    base_url=f"http://{ollama_host}:{ollama_port}", 
    temperature=0, 
    verbose=True).bind_tools(tools)

# ==============================
# Define a system prompt that tells the agent when to invoke image search.
# ==============================
SYSTEM_PROMPT = """
You are SAGE Image Search Agent, an intelligent assistant that can search through images from an application called SAGE.
When a user requests an image search, you must respond with a command in the following format:
<search query>

For example, if a user asks "Show me images of Hawaii", you should respond with:
Hawaii

If the query does not require image search, answer normally.

After the image search tool returns results, incorporate them into your final answer.
"""

# Create a chat prompt template using a system message and a placeholder for conversation history.
prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ==============================
# Define the function that calls the LLM.
# ==============================
# def call_model(state: MessagesState):
#     prompt = prompt_template.invoke(state)
#     response = model.invoke(prompt)
#     # We return a list, because this will get added to the existing list
#     return {"messages": [response]}

# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

# ==============================
# Define the Conditional function to 
# call Image search tool or not.
# ==============================

# Condition: If the LLM makes a tool call, then we route to the "tools" node
def should_call_image_search(state: MessagesState) -> Literal['tools', '__end__']:
    last_message = state["messages"][-1]
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return 'tools'
    else:
        return END # Otherwise, we stop (reply to the user)
    
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

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_call_image_search,
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

config = {"configurable": {"thread_id": 42}}

# ==============================
# Define the Gradio chat function.
# ==============================
def chat(message, history):
    # Start a new conversation with a single human message.
    input_messages = [HumanMessage(message)]
    output = app.invoke({"messages": input_messages}, config)
    # Return the content of the last message as the final answer.
    return output['messages'][-1].content

# ==============================
# Set up the Gradio ChatInterface.
# ==============================
demo = gr.ChatInterface(
    fn=chat,
    type="messages",
    examples=[{"text": "Show me images of Hawaii"}, {"text": "Find images of sunsets"}],
    title="Sage Image Search Agent",
    # multimodal=True,  # Uncomment if you plan to display images or other media.
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)