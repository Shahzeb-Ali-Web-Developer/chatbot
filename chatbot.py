import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the FAQ data from the CSV file
faq_df = pd.read_csv("knowledgebase.csv")  # Make sure the CSV file is in the same directory or specify the path
questions = faq_df["question"].tolist()
answers = faq_df["answer"].tolist()

# Initialize the Sentence Transformer model for generating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all questions in the FAQ
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Function to get the most relevant answer based on user question
def get_answer(user_question):
    # Generate embedding for the user's question
    user_embedding = model.encode(user_question, convert_to_tensor=True)
    
    # Compute similarity scores between the user's question and the FAQ questions
    similarities = util.pytorch_cos_sim(user_embedding, question_embeddings)
    
    # Find the most similar question
    best_match_idx = similarities.argmax().item()
    
    # Return the corresponding answer
    return answers[best_match_idx]





# Main function to interact with the chatbot
def chat():
    print("Hello! I'm your chatbot. Ask me anything (type 'exit' to quit).")
    
    while True:
        # Take user input
        user_question = input("You: ")
        
        # Exit condition
        if user_question.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Get the chatbot's response
        answer = get_answer(user_question)
        
        # Display the answer
        print("Bot:", answer)

# Run the chatbot
if __name__ == "__main__":
    chat()
