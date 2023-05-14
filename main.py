import scipy
from flask import Flask, render_template, request
import pdfplumber
import pickle
import re
from test_utils import *
import PyPDF2
import spacy
app = Flask(__name__)

# Load the pre-trained model from the pickle file
from scipy.sparse import csr_matrix
import pickle

with open('vectorizer.pkl', 'rb') as f:
    model = pickle.load(f, encoding='latin1')

# convert deprecated csr matrix to the new format
if isinstance(model, scipy.sparse.csr_matrix):
    model = csr_matrix(model)


@app.route("/login")
def login():
    return render_template('login.html')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/tweet', methods=["GET","POST"])
def tweet():
    if request.method == "POST" and "username" in request.form:
        username = request.form['username']
        model_prediction, tweets = get_prediction_for_tweets(username)
        return render_template('prediction.html', username=username, predicted_type=model_prediction, tweets=tweets)


@app.route('/predict', methods=['POST'])
def predict():
    # Get the resume file from the form data
    resume = request.files['resume']

    # Read the resume as a PDF file
    with pdfplumber.open(resume) as pdf:
        # Extract the text content of the resume
        text = ""
        for page in pdf.pages:
            text += page.extract_text()

    # Make a prediction on the extracted text using the pre-trained model
    personality_trait = get_prediction(text)
    result=""
    IorE=""
    SorN=""
    TorF=""
    PorJ=""
    character=""
    if personality_trait == "ISTJ":
        result="Reserved and practical, they tend to be loyal, orderly, and traditional."
        character="The Inspector"
        IorE = "Introvert"
        SorN = "Sensing"
        TorF = "Thinking"
        PorJ = "Judging"
    elif personality_trait == "ISTP":
        result="Highly independent, they enjoy new experiences that provide first-hand learning."
        character = "The Crafter"
        IorE = "Introvert"
        SorN = "Sensing"
        TorF = "Thinking"
        PorJ = "Perceiving"
    elif personality_trait == "ISFJ":
        result="Warm-hearted and dedicated, they are always ready to protect the people they care about."
        character = "The Protector"
        IorE = "Introvert"
        SorN = "Sensing"
        TorF = "Feeling"
        PorJ = "Judging"
    elif personality_trait == "ISFP":
        result="Easy-going and flexible, they tend to be reserved and artistic."
        character = "The Artist"
        IorE = "Introvert"
        SorN = "Sensing"
        TorF = "Feeling"
        PorJ = "Perceiving"
    elif personality_trait == "INFJ":
        result="Creative and analytical, they are considered one of the rarest Myers-Briggs types."
        character = "The Advocate"
        IorE = "Introvert"
        SorN = "Intuition"
        TorF = "Feeling"
        PorJ = "Judging"
    elif personality_trait == "INFP":
        result="Idealistic with high values, they strive to make the world a better place."
        character = "The Mediator"
        IorE = "Introvert"
        SorN = "Intuition"
        TorF = "Feeling"
        PorJ = "Perceiving"
    elif personality_trait == "INTJ":
        result="High logical, they are both very creative and analytical."
        character = "The Architect"
        IorE = "Introvert"
        SorN = "Intuition"
        TorF = "Thinking"
        PorJ = "Judging"
    elif personality_trait == "INTP":
        result="Quiet and introverted, they are known for having a rich inner world."
        character = "The Thinker"
        IorE = "Introvert"
        SorN = "Intuition"
        TorF = "Thinking"
        PorJ = "Perceiving"
    elif personality_trait == "ESTP":
        result="Out-going and dramatic, they enjoy spending time with others and focusing on the here-and-now."
        character = "The Persuader"
        IorE = "Extrovert"
        SorN = "Sensing"
        TorF = "Thinking"
        PorJ = "Perceiving"
    elif personality_trait == "ESTJ":
        result="Assertive and rule-oriented, they have high principles and a tendency to take charge."
        character = "The Director"
        IorE = "Extrovert"
        SorN = "Sensing"
        TorF = "Thinking"
        PorJ = "Judging"
    elif personality_trait == "ESFP":
        result="Outgoing and spontaneous, they enjoy taking center stage."
        character = "The Performer"
        IorE = "Extrovert"
        SorN = "Sensing"
        TorF = "Feeling"
        PorJ = "Perceiving"
    elif personality_trait == "ESFJ":
        result="Soft-hearted and outgoing, they tend to believe the best about other people."
        character = "The Caregiver"
        IorE = "Extrovert"
        SorN = "Sensing"
        TorF = "Feeling"
        PorJ = "Judging"
    elif personality_trait == "ENFP":
        result="Charismatic and energetic, they enjoy situations where they can put their creativity to work."
        character = "The Champion"
        IorE = "Extrovert"
        SorN = "Intuition"
        TorF = "Feeling"
        PorJ = "Perceiving"
    elif personality_trait == "ENFJ":
        result="Loyal and sensitive, they are known for being understanding and generous."
        character = "The Giver"
        IorE = "Extrovert"
        SorN = "Intuition"
        TorF = "Feeling"
        PorJ = "Judging"
    elif personality_trait == "ENTP":
        result="Highly inventive, they love being surrounded by ideas and tend to start many projects (but may struggle to finish them)."
        character = "The Debater"
        IorE = "Extrovert"
        SorN = "Intuition"
        TorF = "Thinking"
        PorJ = "Perceiving"
    elif personality_trait == "ENTJ":
        result="Outspoken and confident, they are great at making plans and organizing projects."
        character = "The Commander"
        IorE = "Extrovert"
        SorN = "Intuition"
        TorF = "Thinking"
        PorJ = "Judging"


    # Render the result template with the predicted personality trait
    return render_template('result.html', personality_trait=personality_trait, result=result, character=character,IorE=IorE,SorN=SorN,TorF=TorF,PorJ=PorJ )


if __name__ == '__main__':
    app.run(debug=True)
