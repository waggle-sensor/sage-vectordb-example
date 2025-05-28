from mcp.server.fastmcp import FastMCP
from main import image_search_tool, node_search_tool, get_measurement_name_tool, get_measurement_values_tool

# Initialize MCP server
mcp = FastMCP("SageTools")

@mcp.tool()
def image_search(query: str) -> str:
    """Wrapper for image_search_tool"""
    return image_search_tool(query)

@mcp.tool()
def node_search(vsn: str) -> str:
    """Wrapper for node_search_tool"""
    return node_search_tool(vsn)

@mcp.tool()
def get_measurement_names(vsn: str, time: str) -> str:
    """Wrapper for get_measurement_name_tool"""
    return get_measurement_name_tool(vsn, time)

@mcp.tool()
def get_measurement_values(vsn: str, measurement_name: str, time: str) -> str:
    """Wrapper for get_measurement_values_tool"""
    return get_measurement_values_tool(vsn, measurement_name, time)

if __name__ == "__main__":
    # Start MCP server on default host and port
    mcp.run(transport="streamable-http")
