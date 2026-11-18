"""Entry point for running the Finn.no MCP server via ``python -m finn_deals.mcp``."""

import argparse

from finn_deals.mcp.server import main


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finn.no MCP server")
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport to use (default: stdio)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.transport)
