import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import librosa

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the Keras model
model = load_model('Emotion_voice_detection_model.keras')

# Define a function to predict emotions from audio files
def predict_emotion(audio_file):
    # Load and preprocess the audio file
    audio, _ = librosa.load(audio_file, sr=22050, duration=3)
    audio = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=40).T

    # Reshape the input for the model
    audio = audio.reshape((1, audio.shape[0], audio.shape[1]))

    # Perform emotion prediction
    predictions = model.predict(audio)
    predicted_emotion = ['happy', 'sad', 'angry', 'neutral', 'ps', 'disgust', 'fear'][predictions.argmax()]

    return predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predicted_emotion = predict_emotion(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'emotion': predicted_emotion})

if __name__ == '__main__':
    app.run(debug=True)