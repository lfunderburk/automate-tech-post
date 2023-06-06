from langchain.agents import initialize_agent
import json
from dotenv import load_dotenv
import openai
from langchain.chat_models import ChatOpenAI
import os
from pathlib import Path
from utils import extract_data_from_page, generate_post, save_data_to_hugging_face

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



