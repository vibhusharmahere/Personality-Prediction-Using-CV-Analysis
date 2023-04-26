import scipy
from flask import Flask, render_template, request
import pdfplumber
import pickle
from test_utils import *

app = Flask(__name__)

# Load the pre-trained model from the pickle file
from scipy.sparse import csr_matrix
import pickle

with open('vectorizer.pkl', 'rb') as f:
    model = pickle.load(f, encoding='latin1')

# convert deprecated csr matrix to the new format
if isinstance(model, scipy.sparse.csr_matrix):
    model = csr_matrix(model)



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
    if personality_trait == "ISTJ":
        result="ISTJ - The Inspector: Reserved and practical, they tend to be loyal, orderly, and traditional."
    elif personality_trait == "ISTP":
        result="ISTP - The Crafter: Highly independent, they enjoy new experiences that provide first-hand learning."
    elif personality_trait == "ISFJ":
        result="ISFJ - The Protector: Warm-hearted and dedicated, they are always ready to protect the people they care about."
    elif personality_trait == "ISFP":
        result="ISFP - The Artist: Easy-going and flexible, they tend to be reserved and artistic."
    elif personality_trait == "INFJ":
        result="INFJ - The Advocate: Creative and analytical, they are considered one of the rarest Myers-Briggs types."
    elif personality_trait == "INFP":
        result="INFP - The Mediator: Idealistic with high values, they strive to make the world a better place."
    elif personality_trait == "INTJ":
        result="INTJ - The Architect: High logical, they are both very creative and analytical."
    elif personality_trait == "INTP":
        result="INTP - The Thinker: Quiet and introverted, they are known for having a rich inner world."
    elif personality_trait == "ESTP":
        result="ESTP - The Persuader: Out-going and dramatic, they enjoy spending time with others and focusing on the here-and-now."
    elif personality_trait == "ESTJ":
        result="ESTJ - The Director: Assertive and rule-oriented, they have high principles and a tendency to take charge."
    elif personality_trait == "ESFP":
        result="ESFP - The Performer: Outgoing and spontaneous, they enjoy taking center stage."
    elif personality_trait == "ESFJ":
        result="ESFJ - The Caregiver: Soft-hearted and outgoing, they tend to believe the best about other people."
    elif personality_trait == "ENFP":
        result="ENFP - The Champion: Charismatic and energetic, they enjoy situations where they can put their creativity to work."
    elif personality_trait == "ENFJ":
        result="ENFJ - The Giver: Loyal and sensitive, they are known for being understanding and generous."
    elif personality_trait == "ENTP":
        result="ENTP - The Debater: Highly inventive, they love being surrounded by ideas and tend to start many projects (but may struggle to finish them)."
    elif personality_trait == "ENTJ":
        result="ENTJ - The Commander: Outspoken and confident, they are great at making plans and organizing projects."


    # Render the result template with the predicted personality trait
    return render_template('result.html', personality_trait=result)


if __name__ == '__main__':
    app.run(debug=True)
