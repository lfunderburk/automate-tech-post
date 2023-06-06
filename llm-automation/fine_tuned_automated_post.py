import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from IPython.display import display, Markdown

def make_inference(topic):
  batch = tokenizer(f"### INSTRUCTION\nBelow summary for a blog post, \
                    please write a social media post\
                    \n\n### Topic:\n{topic}\n### Social media post:\n", return_tensors='pt')

  with torch.cuda.amp.autocast():
    output_tokens = model.generate(**batch, max_new_tokens=200)

  display(Markdown((tokenizer.decode(output_tokens[0], skip_special_tokens=True))))

if __name__=="__main__":

  # Set up user name and model name
  hf_username = "lgfunderburk"
  model_name = 'numpy-social-media-post'
  peft_model_id = f"{hf_username}/{model_name}"

  # Apply PETF configuration, setup model and autotokenizer
  config = PeftConfig.from_pretrained(peft_model_id)
  model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=False, device_map='auto')
  tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
  
  # Load the Lora model
  model = PeftModel.from_pretrained(model, peft_model_id)

  # Summary to generate a social media post about
  topic = "The blog post demonstrates how to use JupySQL and DuckDB to query CSV files with SQL in a Jupyter notebook. \
          It covers installation, setup, querying, and converting queries to DataFrame. \
          Additionally, the post shows how to register SQLite user-defined functions (UDF), \
          connect to a SQLite database with spaces, switch connections between databases, and connect to existing engines. \
          It also provides tips for using JupySQL in Databricks, ignoring deprecation warnings, and hiding connection strings."
  

  # Generate social media post
  make_inference(topic)