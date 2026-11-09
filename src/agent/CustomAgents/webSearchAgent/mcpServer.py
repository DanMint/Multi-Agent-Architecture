from fastmcp import FastMCP, Client
from ddgs import DDGS
import asyncio
import trafilatura
from urllib.parse import urlparse
import httpx
import re

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