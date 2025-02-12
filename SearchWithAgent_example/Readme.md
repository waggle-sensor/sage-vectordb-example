# Sage Agent

This project demonstrates an AI agent with access to Sage tools

---
## Features:
- Chat based system to find details about a node and find images.
- **Image Search**: Search for images based on a text query. This is were [Hybrid Search](../HybridSearch_example/) is implemented
- **Node Search**: Retrieve details about a specific node.
- **Measurement Name Search**: Find measurement names based on time in a node.
- **Measurement Value Search**: Get measurement values based on time and measurement name in a node.
---

### Authentication
To set up your cred environment variables . You can either:
- Set the environment variable directly in the terminal like this:
  ```bash
  export SAGE_USER=__INSERT_HERE__
  export SAGE_TOKEN=__INSERT_HERE__
  ```
- Or, you can create a `.env` file in the root of your project with the following content:
  ```sh
  export SAGE_USER=__INSERT_HERE__
  export SAGE_TOKEN=__INSERT_HERE__
  ```
- Then, run:
  ```bash
  source .env
  ```

---

## Running the Example
>NOTE: I didn't use docker compose because it doesn't have the ability to access to GPU in lower versions, like in node-V033

### Prerequisites
To run this example, you'll need:
- **Docker** installed on your machine with GPU access
- **Cuda** v11.6
- NVIDIA Driver Release 510 or later

### Step-by-Step Setup

1. **Spin up your Weaviate instance**:
   - Navigate to the directory containing the `Makefile` file and run:
     ```bash
     make db
     ```

2. **Spin up the app**:
   - Navigate to the directory containing the `Makefile` file and run:
     ```bash
     make build && make up
     ```

3. **Access Gradio App**:
   - After your Weaviate instance is running, access the user interface at:
     ```
     http://localhost:7861/ #or the shareable link gradio outputted in terminal
     ```

4. **Image Access**:
   - Before running, make sure you have access to the image data from Sage. You will need to fetch the relevant image dataset to perform searches.

---

## Optional

- **Accessing the UI Remotely through port forwarding**:
   - If your AI agent instance is running on a remote machine, use SSH tunneling to access the UI:
     ```bash
     ssh <client> -L 7861:<EXTERNAL-IP>:7861
     ```
   - Example:
     ```bash
     ssh node-V033 -L 7861:10.31.81.1:7861
     ```

---

## References

- **LangGraph & LangChain Documentation**:  
   - [LangGraph](https://www.langchain.com/langgraph)
   - [LangChain](https://www.langchain.com/)
   - [Making a Chatbot](https://www.youtube.com/watch?v=f9HXp75DVNY)
   - [LangChain First Steps](https://github.com/casedone/langchain-first-step/blob/main/llm-as-coach.py)
   - [LangGraph Tutorial](https://www.youtube.com/watch?v=pDuNkb6VWaM)
   - [Fully local tool calling with Ollama](https://www.youtube.com/watch?v=Nfk99Fz8H9k)
   - [ChatOllama](https://python.langchain.com/docs/integrations/chat/ollama/)
   - [LangGraph Repo](https://github.com/langchain-ai/langgraph)
   - [LangChain Documentation](https://python.langchain.com/docs/introduction/)
   - [Base Model Pydantic](https://docs.pydantic.dev/latest/api/base_model/)
   - [Building Agents with LangChain's Agent Framework](https://www.getzep.com/ai-agents/langgraph-tutorial)
- **Gradio Documentation**:
    - [Building a UI for an LLM Agent](https://www.gradio.app/guides/agents-and-tool-usage)
    - [Gradio & LLM Agents](https://www.gradio.app/guides/gradio-and-llm-agents)
    - [Chat Interface](https://www.gradio.app/docs/gradio/chatinterface)
    - [How to Create a Chatbot with Gradio](https://www.gradio.app/guides/creating-a-chatbot-fast)
---
