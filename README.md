# Offline FAQ Chatbot

This is a simple, offline FAQ chatbot that responds to predefined questions and answers using sentence similarity. It’s implemented in Python and uses the `sentence-transformers` library to find the best answer match based on user input.

## Features

- **Offline and Lightweight**: No server or internet connection is needed; it runs locally on your computer.
- **Predefined FAQ Responses**: Responds based on a predefined set of FAQs.
- **Similarity Matching**: Uses sentence embeddings to find the closest match to the user’s question.
- **Easy Customization**: You can easily add more questions and answers to improve the chatbot’s responses.

## Demo

![Chatbot Demo GIF](path_to_demo.gif)

## Requirements

- **Python 3.x**
- **sentence-transformers** library for generating sentence embeddings.

## Installation

1. Clone this repository or download the code files.

    ```bash
    git clone https://github.com/your-username/offline-faq-chatbot.git
    cd offline-faq-chatbot
    ```

2. Install the required dependencies:

    ```bash
    pip install sentence-transformers
    ```

## Usage

Run the chatbot by executing the following command:

```bash
python offline_chatbot.py
