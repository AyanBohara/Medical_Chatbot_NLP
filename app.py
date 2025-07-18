from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
from keras.models import load_model
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Load model and data
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.load(open('intents.json'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__, static_folder='.')

# --- Copy your preprocessing and response functions here ---
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = np.random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    if not ints or float(ints[0]['probability']) < 0.8:
        return "Sorry, I don't understand."
    else:
        res = getResponse(ints, intents)
        return res

# --- End of chatbot logic ---

@app.route('/')
def serve_index():
    return send_from_directory('.', 'Chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    bot_reply = chatbot_response(user_message)
    return jsonify({'reply': bot_reply})

if __name__ == '__main__':
    app.run(debug=True)
