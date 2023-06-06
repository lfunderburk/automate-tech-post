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

## Executing LangChain pipeline

Create a `.env` file where you can store your OpenAI API key. The set your key within the `.env` file as follows:

```
OPENAI_API_KEY = <your-keyy>
```

You can execute the pipeline as follows:

```
python llm-automation/blog_to_post.py
```

## Fine-tuning a model

If you would prefer not to use OpenAI API and fine tune a model instead, you can use the following colab notebook. 

### Assumptions:

Training the model requires GPUs and high RAM. If your local machine does not support this, you can use Colab Pro with the following specs:

![](colab-reqs.png)