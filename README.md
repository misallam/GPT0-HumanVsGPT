# GPT0 (Human Written Vs GPT Written Text)
The "GPT0" project introduces a cutting-edge application designed to differentiate between human-written and GPT-generated text. At its core, the application leverages a machine learning model built with PyTorch, trained on a dataset comprising over 260,000 labeled instances. This initiative aims to support text analysis and digital authenticity verification efforts, providing a valuable tool in the age of advanced text-generating AI.

## Video Overview
Watch this comprehensive video guide to get started and understand the capabilities of GPT0: [GPT0 Video Guide](https://drive.google.com/drive/folders/1hu5Vhj8ymjP1_H11UdcmOLqaCMBHxcNf?usp=sharing)

## Features
- Streamlit-based interactive interface for easy use.
- Utilizes the SentenceTransformer for accurate text embeddings.
- Simple neural network model trained on a vast dataset for high precision in distinguishing between human and AI text.

## Installation
1. Clone the repository to your local machine.
2. Download the required checkpoints and datasets from [this link](https://drive.google.com/drive/folders/1hu5Vhj8ymjP1_H11UdcmOLqaCMBHxcNf?usp=sharing) and place them into their respective folders within the repository.
3. Ensure you have Python 3.7 or newer installed.
4. Install the required dependencies by running `pip install -r requirements.txt` from the command line within the project directory.
5. To start the Streamlit application, execute `streamlit run gpt0.py` in your terminal.

## How It Works
- The `gpt0.py` script is the entry point for the Streamlit application. It utilizes the SentenceTransformer model for generating embeddings of the input text.
![GPT0 Model Primary GUI](https://github.com/misallam/GPT0-HumanVsGPT/gpt0.png)
- The machine learning model, detailed within `model.ipynb`, is trained to recognize patterns in text that differentiate human writing from text generated by models like GPT.
- Users can input text into the Streamlit interface, which is then processed by the model to classify as either human-written or GPT-generated.
