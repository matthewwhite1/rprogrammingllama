# rprogrammingllama

This project allows you to locally run an R-programming chatbot. The chatbot accesses a Code Llama 7B model fine tuned on R programming examples.

## Running the Model

### Download this repository

```bash
git clone https://github.com/matthewwhite1/rprogrammingllama
```

### Run the chatbot

Load into the virtual environment

```bash
source .venv/Scripts/activate
```

Run the chatbot

```bash
python main.py
```

### Access the chatbot

Once your terminal says something like:

```bash
Loading checkpoint shards: 100%|##########| 2/2 [00:20<00:00, 10.46s/it]
```

You can access the model locally by entering the URL:

http://127.0.0.1:7860

## Data sources

The chatbot access a Code Llama 7B model found at:

https://huggingface.co/codellama

The R examples were downloaded from:

https://github.com/drndr/gencodesearchnet