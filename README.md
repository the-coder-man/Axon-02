# 🤖 Axon 02 Chatbot

Axon 02 is a Python-based AI chatbot that uses machine learning and natural language processing (NLP) to simulate human conversation. It supports topics like books, health, fun facts, music, and can even answer questions using Wikipedia.

---

## 🛠️ Features

- Pre-trained with JSON-based intents
- Answers questions, tells jokes, gives advice, and more
- Wikipedia support via the `wikipedia` module
- Easy to customize and extend
- Command-line interface

---

## 📂 Files Included

- `axon 02.py` — The main chatbot Python script
- `chatbot_training_data.json` — Intent dataset for training the bot
- `chatbot.pth` - a pre-set model for Axon 02 to work off

---

## ⚙️ Installation

### Step 1: Clone This Repository

```bash
cd axon-chatbot
```

### Step 2: (Optional) Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### Step 3: Install Required Libraries

Create a `requirements.txt` with the following contents:

```text
torch
nltk
numpy
wikipedia
requests
pyttsx3
```

Then run:

```bash
pip install -r requirements.txt
```

Or install them manually:

```bash
pip install torch nltk numpy wikipedia pyttsx3 requests
```

---

## 🚀 Running the Chatbot

Start the bot by running:

```bash
python "axon 02.py"
```

Once it launches, type messages like:

```text
> Hello
> Tell me a joke
> What is artificial intelligence?
> Recommend a good book
> Ask me a riddle
```

---

## 🧐 Customizing the Chatbot

To add new intents, open `chatbot_training_data.json` and add new sections like:

```json
{
  "tag": "new_topic",
  "patterns": ["example question"],
  "responses": ["your response here"]
}
```

Then retrain or restart the chatbot to include the new patterns.

---

## 🌐 Wikipedia Integration

Ask:

```text
>  climate change
>  Ada Lovelace
```

The bot uses Wikipedia to summarize the answer for you.

Note: Ensure internet connectivity is available to use this feature.

---

## 🙋‍ Authors

Created by:

doctor retro G aka "the coder man" on Github

---

## 📄 License

MIT License. Use, modify, and distribute freely.
