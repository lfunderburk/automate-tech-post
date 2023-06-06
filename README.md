# Set up

Create a virtual environment

```
conda create --name postenv python==3.10
```

Activate

```
conda activate postenv
```

Clone repo and install dependencies

```
git clone https://github.com/lfunderburk/automate-tech-post.git
cd automate-tech-post/
pip install -r requirements.txt
```

Create a `.env` file where you can store your OpenAI API key. The set your key within the `.env` file as follows:

```
OPENAI_API_KEY = <your-keyy>
```

## Executing LangChain pipeline

You can execute the pipeline as follows:

```
python llm-automation/blog_to_post.py
```