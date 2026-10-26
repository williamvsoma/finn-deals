"""Finn.no MCP server exposing listing search tools to agents."""

from mcp.server.fastmcp import FastMCP

from finn_deals.mcp.tools import finn_get_image, finn_get_listing, finn_search

mcp = FastMCP(
    name="finn-deals",
    instructions=(
        "Finn.no listing search and lookup server. "
        "Use finn_search to run a Finn.no search and filter listings by price range and location. "
        "Use finn_get_listing to fetch full details about a single listing (description, images, "
        "location, condition/extras, category, seller). "
        "Use finn_get_image to fetch and view a listing image."
    ),
)


mcp.add_tool(
    fn=finn_search,
    name="finn_search",
    description=finn_search.__doc__,
)

mcp.add_tool(
    fn=finn_get_listing,
    name="finn_get_listing",
    description=finn_get_listing.__doc__,
)

mcp.add_tool(
    fn=finn_get_image,
    name="finn_get_image",
    description=finn_get_image.__doc__,
)


def main(transport: str = "stdio") -> None:
    """Run the Finn.no MCP server."""
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "sse":
        mcp.run(transport="sse")
    elif transport == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(f"Unknown MCP transport: {transport}")
