import pandas as pd
import openai
from langchain.agents import tool
import requests
from bs4 import BeautifulSoup
import openai
import pandas as pd
from datasets import Dataset

class Prompter:
    def __init__(self, api_key, gpt_model, temperature=0.2):
        if not api_key:
            raise Exception("Please provide the OpenAI API key")

        self.gpt_model = gpt_model
        self.temperature = temperature
    
    def prompt_model_return(self, messages: list):
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=self.temperature)
        return response["choices"][0]["message"]["content"]
    
    def social_media_wizard(self, post_summary:str):

        system_content = "You are an expert digital marketing with knowledge about SQL, Python and Jupyter Notebooks. \
                        You are charismatic and have a great personality. You are given a summary for a topic your task is to write an engaging social media post.\
                        You are knwoledgeable about the following topics: SQL, Python, Jupyter Notebooks, Digital Marketing, Social Media, and Data Science.\
                        You also know how to select the appropriate hashtags for a post and when to use emojis and what emojis to use."
        user_content = f"Please write a social media post for the following summary: {post_summary}"

        social_media_prompts = [
                                {"role" : "system", "content" : system_content},
                                {"role" : "user", "content" : user_content},
                                ]
        
        result = self.prompt_model_return(social_media_prompts)

        return result
    

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

    
