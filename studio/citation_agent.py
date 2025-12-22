import os
from pathlib import Path
from typing import List, TypedDict, Literal, Optional
from datetime import datetime
from langgraph.graph import StateGraph, START, END
from trafilatura import fetch_url, extract
import json
from together import Together
from pydantic import BaseModel, Field
from IPython.display import Image, display

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

script_dir = Path.cwd().parent
load_dotenv('../studio/.env')

model = ChatOpenAI(model="gpt-4o", temperature=0) 
client = Together()

class Author(BaseModel):
    first_name: str = Field(description="The author's first name")
    middle_name: str = Field(description="The author's middle name (can be empty string if not available)")
    last_name: str = Field(description="The author's last name")

class CitationData(BaseModel):
    authors: List[Author] = Field(
        description="A list of authors for the article"
    )
    article_title: str = Field(
        description="The headline or title of the article/page"
    )
    site_name: str = Field(
        description="The name of the website or publication"
    )
    publisher: str = Field(
        description="The organization publishing the site (often same as site name)"
    )
    pub_year: int = Field(
        description="The year of publication as an integer"
    )
    pub_date_formatted: str = Field(
        description="The publication date formatted for the citation style (e.g., '2023, May 12')"
    )

class State(BaseModel):
    citation_style: Literal['apa_7', 'mla_9', 'chicago_17', 'ieee', 'harvard']
    url: str
    raw_website_output: Optional[str] = None
    citation_data: Optional[CitationData] = None
    finalized_citation: Optional[str] = None
    date_accessed: Optional[str] = datetime.now().strftime("%Y-%m-%d")

def get_website_data(state: State):
    downloaded = fetch_url(state.url)
    result = extract(downloaded, with_metadata=True)
    return {"raw_website_output": result}


def extract_citation_data(state: State):
    prompt_path = script_dir / "prompts" / "citation_extraction.txt"
    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    formatted_prompt = prompt.format(extracted_website_data=state.raw_website_output, json_format=str(json.dumps(CitationData.model_json_schema())))
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        model="Qwen/Qwen3-Next-80B-A3B-Instruct",
        response_format={
            "type": "json_schema",
            "schema": CitationData.model_json_schema(),
        },
    )
    output = json.loads(response.choices[0].message.content)
    return {"citation_data": output}

def finalize_citation(state: State):
    citation_styles_path = script_dir / "public" / "citation_styles.json"
    with open(citation_styles_path, 'r') as f:
        json_data = json.load(f)
    citation_styles = json_data["citation_styles"]
    citation_style_data = json.dumps(next((style for style in citation_styles if style["id"] == state.citation_style), None))

    prompt_path = script_dir / "prompts" / "finalize_citation.txt"

    with open(prompt_path, 'r') as f:
        prompt = f.read()
    
    formatted_prompt = prompt.format(
        citation_style_guide=citation_style_data,
        citation_data=state.citation_data,
        url=state.url,
        date_accessed=state.date_accessed,
    )

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": formatted_prompt},
        ],
        model="Qwen/Qwen3-Next-80B-A3B-Instruct")
    
    return {"finalized_citation": response.choices[0].message.content}
    
workflow = StateGraph(State)
workflow.add_node("get_websites_data", get_website_data)
workflow.add_node("extract_citation_data", extract_citation_data)
workflow.add_node("finalize_citation", finalize_citation)

workflow.add_edge(START, "get_websites_data")
workflow.add_edge("get_websites_data", "extract_citation_data")
workflow.add_edge("extract_citation_data", "finalize_citation")
workflow.add_edge("finalize_citation", END)


graph = workflow.compile()