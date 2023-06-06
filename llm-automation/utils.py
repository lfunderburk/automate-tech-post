import pandas as pd
import openai

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
    
    
