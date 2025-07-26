import json
import string
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import tkinter as tk
import pyttsx3
import wikipedia
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
# Load intents
with open("intents copy/chatbot_training_data.json", "r") as f:
    intents = json.load(f)

# Tokenizer function (simple split and clean)
def tokenize(sentence):
    translator = str.maketrans('', '', string.punctuation)
    return sentence.translate(translator).lower().split()

# Bag of words
def bag_of_words(tokenized_sentence, words):
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

# Create training data
all_words = []
tags = []
x_train = []
y_train = []

for intent in intents['intents']:
    if 'tag' not in intent or 'patterns' not in intent:
        print(f"Skipping invalid intent: {intent}")
        continue
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        x_train.append(w)
        y_train.append(tag)


all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = [bag_of_words(x, all_words) for x in x_train]
Y_train = [tags.index(y) for y in y_train]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return self.l3(x)

model = NeuralNet(len(all_words), 8, len(tags))

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def augment_training_with_wiki(tag, topic, intents_data):
    summary = fetch_wikipedia_summary(topic)
    preprocessed = preprocess_text(summary)
    for sentence in preprocessed:
        # Join cleaned words back into a pattern string
        pattern = ' '.join(sentence)
        if not intent or not intent.get("name"):
            continue  # or log it for review
        # Add to your intents dataset under a new or existing tag
        for intent in intents_data['intents']:
            if intent['tag'] == tag:
                intent['patterns'].append(pattern)
                break

for epoch in range(1000):
    outputs = model(torch.from_numpy(X_train))
    loss = criterion(outputs, torch.tensor(Y_train))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training complete")

# Text-to-speech
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Get response from model
def get_response(sentence):
    sentence_tokens = tokenize(sentence)
    X = bag_of_words(sentence_tokens, all_words)
    X = torch.from_numpy(X).unsqueeze(0)

    with torch.no_grad():
        output = model(X)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    if confidence.item() < 0.75:
        return "I'm not sure how to respond to that."
    tag = tags[predicted]
    for intent in intents['intents']:
        if tag == intent['tag']:
            return random.choice(intent['responses'])

    return "I'm not sure how to respond to that."


def fetch_wikipedia_summary(query, sentences=5):
    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(query, sentences=sentences)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation Error: {e.options}"
    except wikipedia.exceptions.PageError:
        return "Page not found."
def preprocess_text(text):
    # Tokenize sentences
    sentences = sent_tokenize(text)
    cleaned = []

    for sent in sentences:
        words = word_tokenize(sent)
        # Remove punctuation and lowercase
        words = [w.lower() for w in words if w.isalpha()]
        # Remove stopwords
        words = [w for w in words if w not in stopwords.words('english')]
        cleaned.append(words)

    return cleaned

# GUI setup
root = tk.Tk()
root.title("Axon 02")

frame = tk.Frame(root)
frame.pack(padx=10, pady=10)

chat_log = tk.Text(frame, height=20, width=60, wrap=tk.WORD)
chat_log.pack()

entry = tk.Entry(frame, width=50)
entry.pack(pady=5)

button_frame = tk.Frame(frame)
button_frame.pack(pady=5)

send_button = tk.Button(button_frame, text="Send", command=lambda: on_send())
send_button.pack(side=tk.LEFT, padx=5)



# Flag to track button state
wiki_button_active = False

# Dummy search_wikipedia function
def change_button():
    print("Searching Wikipedia...")

# Function to toggle color and perform search
def toggle_wikipedia_button():
    global wiki_button_active
    if not wiki_button_active:
        wiki_button.config(fg="blue")
        search_wikipedia()
    else:
        wiki_button.config(fg="black")
    wiki_button_active = not wiki_button_active

# GUI setup
root.title("Axon 02")

button_frame = tk.Frame(root)
button_frame.pack(pady=10)

wiki_button = tk.Button(button_frame, text="Search Wikipedia", command=toggle_wikipedia_button)
wiki_button.pack(side=tk.RIGHT, padx=5)

# Button functions
def on_send():
    user_input = entry.get().strip()
    if not user_input:
        return
    entry.delete(0, tk.END)
    chat_log.insert(tk.END, f"You: {user_input}\n")
    response = get_response(user_input)
    chat_log.insert(tk.END, f"Axon 02: {response}\n")
    chat_log.see(tk.END)
    speak(response)

def search_wikipedia():
    query = entry.get().strip()
    if not query:
        return
    result = fetch_wikipedia_summary(query)
    chat_log.insert(tk.END, f"Axon 02 (wikipedia): {result}\n")
    chat_log.see(tk.END)
    speak(result)

# Main loop
root.mainloop()
