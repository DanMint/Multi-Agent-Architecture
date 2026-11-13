from fastmcp import FastMCP
from ddgs import DDGS
import asyncio
import trafilatura
from urllib.parse import urlparse
import httpx

"""
ResearchAgent MCP Server: Provides web search and webpage retrieval tools
for the ResearchAgent using the Model Context Protocol (MCP).

    - mcp: FastMCP server that exposes tools to connected agents.
    - DDGS: Performs web searches and retrieves search results.
    - trafilatura: Extracts readable text from HTML webpages.
    - httpx: Handles asynchronous HTTP requests for webpage retrieval.

    Methods:
        - web_search(query, max_results): Searches the web and returns
          webpage titles, URLs, and snippets (maximum 10 results).

        - perform_search(query, max_results): Executes the underlying
          DDGS search in a separate thread to avoid blocking asyncio.

        - fetch_webpage(url, max_chars): Retrieves and extracts readable
          webpage content while enforcing HTTPS, response size limits,
          and basic URL security checks.

        - main(): Starts the MCP HTTP server on 127.0.0.1:8001.
"""

mcp = FastMCP("Research Agent Tools")

# web search tool
@mcp.tool
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    max_results = max(1, min(max_results, 10))
    results = await asyncio.to_thread(
        perform_search,
        query,
        max_results
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", "")
        }
        for result in (results or [])
    ]

def perform_search(query:str, max_results: int):
    with DDGS(timeout=10) as ddgs:
        return ddgs.text(
            query,
            max_results=max_results
        )

@mcp.tool
async def web_search(query: str, max_results: int = 5) -> list[dict]:
    print("-----------USING WEB SEARCH TOOL-----------")
    max_results = max(1, min(max_results, 10))
    results = await asyncio.to_thread(
        perform_search,
        query,
        max_results
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("href", ""),
            "snippet": result.get("body", "")
        }
        for result in (results or [])
    ]

@mcp.tool
async def fetch_webpage(url: str, max_chars: int = 10000) -> dict:
    print("-----------FETCH WEPAGE CONTET-----------")
    """Extract the readable content of a public HTTPS webpage."""

    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return {"error": "A valid HTTPS URL is required."}

    # Prototype guard; production requires network-level SSRF protection.
    hostname = parsed.hostname.lower()
    if (
        hostname == "localhost"
        or hostname.endswith((".localhost", ".local", ".internal"))
        or ":" in hostname
        or hostname.replace(".", "").isdigit()
    ):
        return {"error": "Local or IP-based URLs are not allowed."}

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()

            if "text/html" not in response.headers.get("content-type", ""):
                return {"error": "The URL is not an HTML page."}

            chunks = []
            size = 0
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > 2_000_000:
                    return {"error": "Page exceeds size limit."}
                chunks.append(chunk)

    html = b"".join(chunks).decode("utf-8", errors="replace")
    content = await asyncio.to_thread(trafilatura.extract, html, url=url)

    return {
        "url": url,
        "content": (content or "")[:max(1, min(max_chars, 20000))]
    }

def main():
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001
    )

if __name__ == "__main__":
    main()