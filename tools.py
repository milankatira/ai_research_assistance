from langchain.tools import tool
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from rich import print

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

@tool
def web_search(query: str) -> str:
    """Search the web for recent and reliable information on topic. Return Title , Urls and snippets of the results."""
    results=tavily_client.search(query,max_results=5)

    out=[]
    for r in results['results']:
        out.append(f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}...")
    return "\n--------------------------------\n".join(out)


print(web_search.invoke("What is the recent news of war?"))