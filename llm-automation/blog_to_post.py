from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
import requests
from bs4 import BeautifulSoup
import json
from dotenv import load_dotenv
import openai
import os
from pathlib import Path
import pandas as pd
from utils import Prompter
from datasets import Dataset

def save_data_to_hugging_face(master_dic, hf_username, dataset_name):
    """
    Saves data to Hugging Face

    Parameters
    ----------
    master_dic : list
        A list of dictionaries containing the id, summary and social media post
    hf_username : str
        The username of the Hugging Face account
    dataset_name : str
        The name of the dataset to be created in Hugging Face

    Returns
    -------
    None
    """
    id = [item['id'] for item in master_dic]
    summary =  [item['summary'] for item in master_dic]
    sm_post = [item['social_media_post'] for item in master_dic]

    df = pd.DataFrame({'id': id, 'summary': summary, 'social_media_post': sm_post})

    # Push to Hugging Face
    hf_dataset = Dataset.from_pandas(df)
    

    hf_dataset.push_to_hub(f"{hf_username}/{dataset_name}")

@tool
def extract_data_from_page(url):

    """
    Extracts data from a webpage
    
    Parameters
    ----------
    url : str
        The url of the webpage to be scraped
        
    Returns
    -------
    final : list
        A list of the content of the webpage"""

    # Send a GET request to the webpage
    response = requests.get(url)

    # Parse the webpage's content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find and keep the order of h1, h2, h3, p tags
    content = []
    # Look for the tags of interest in order of appearance
    for tag in soup.find_all(['h1', 'h2', 'h3', 'p', 'pre', 'div']):
        # Ignore spans with class "caption-text"
        if tag.name == 'div' and 'sidebar-primary-item' in tag.get('class', []):
            continue

        # Check if it's the div with class 'cell docutils container'
        if tag.name == 'div' and 'cell docutils container' in tag.get('class', []):
            content.append(f"{tag.get_text(strip=True)}")
        # Check if it's 'pre' tag within the 'div' with class 'cell docutils container'
        elif tag.name == 'pre' and tag.find_parent('div', {'class': 'cell docutils container'}):
            content.append(f"{tag.get_text(strip=True)}")
        elif tag.name in ['h1', 'h2', 'h3', 'p']:
            content.append(f"{tag.get_text(strip=True)}")

    # Join all the elements together
    text = "\n".join(content)

    # remove the first 9 lines and the last 6 lines
    final = "\n".join(text.split("\n")[9:-6])

    return final

@tool
def generate_post(summary):
    """
    Generates a social media post from a summary
    
    Parameters
    ----------
    summary : str
        The summary of the content of a webpage

    Returns
    -------
    post : str
        The social media post inviting users to read the content of the webpage
    """

    prompter = Prompter(api_key=openai.api_key, gpt_model="gpt-4", temperature=1)

    post = prompter.social_media_wizard(summary)

    return post




if __name__=="__main__":

    # Load environment variables
    load_dotenv(".env")

    # Set your API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    llm = ChatOpenAI(model='gpt-4', temperature=1)

    # read content of data
    path_to_posts = Path(os.getcwd()) / "llm-automation"/ "posts-data" 
    social_media_results = path_to_posts / "summary_and_social_media_post_demo.json"

    # This is an example for the FASTAPI documentation
    # Replace with your list of links 
    links = ["https://numpy.org/numpy-tutorials/content/tutorial-svd.html"]

    master_dic = []
    i = 0
    for url in links:
        i+=1

        try:
            final = extract_data_from_page(url)
            # check if final is empty
            if not final:
                continue

            tools = [extract_data_from_page, generate_post]
            agent = initialize_agent(tools, llm, agent="chat-zero-shot-react-description", verbose=True)

            # Run the agent to perform a summary of the page
            summary = agent.run(f"Summarize the content of the following blog post: {url}")

            # Run the agent to generate a social media post
            social_media_post = agent.run(f"Generate a social media post for the following blog post: {summary}")

            # Append the dictionaries to the master dictionary
            master_dic.append({'id': i, "link": url, "summary": summary, "social_media_post": social_media_post})

        except Exception as e:
            master_dic.append({'id': i, "link": url, "summary": summary, "social_media_post": "", "error": str(e)})
            continue

    
    # Format 1: Save dic to json file
    with open(social_media_results, "w", encoding='utf-8') as f:
        json.dump(master_dic, f, ensure_ascii=False, indent=4)

    # Format 2: Load to Hugging Face
    # Format data as a dataframe
    try:
        hf_username = "your-hugging-face-username"
        dataset_name = "your-dataset-name"

        save_data_to_hugging_face(master_dic, hf_username, dataset_name)
    except Exception as e:
        print(e)
        pass



