from flask import Flask, request, render_template, jsonify, send_file
import os
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import tempfile
import json
from threading import Lock
from classroom_assistant_api import ClassroomAssistant
from flask_cors import CORS
import speech_recognition as sr 
import wave

app = Flask(__name__)
CORS(app)

# Initialize the assistant once
assistant = ClassroomAssistant()
# Lock for matplotlib to avoid race conditions
plt_lock = Lock()

# Save original matplotlib show function
original_show = plt.show

# Override matplotlib's show function to save to a buffer instead
def save_plot():
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close('all')
    return img_buf

# Create a temporary directory for files
TEMP_DIR = tempfile.mkdtemp()
os.makedirs(os.path.join(TEMP_DIR, 'uploads'), exist_ok=True)
os.makedirs(os.path.join(TEMP_DIR, 'outputs'), exist_ok=True)
@app.route('/')
def home():
    return render_template('index.html') 


@app.route('/process', methods=['POST'])
def process_query():
    # Get query from form or JSON
    if request.content_type == 'application/json':
        data = request.get_json()
        query = data.get('query', '')
    else:
        query = request.form.get('query', '')
    
    if not query:
        return jsonify({'status': 'error', 'message': 'No query provided'})
    
    # Process the query
    try:
        with plt_lock:
            # Override plt.show to capture output
            plt.show = lambda: None
            
            # Process the query
            response, is_visual = assistant.process_query(query)
            
            # If it's a visual response, get the image
            if is_visual:
                # Check if we have diagram.png from mermaid
                if os.path.exists("diagram.png"):
                    with open("diagram.png", "rb") as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    os.remove("diagram.png")  # Clean up
                else:
                    # Save any matplotlib figures
                    img_buf = save_plot()
                    img_data = base64.b64encode(img_buf.getvalue()).decode('utf-8')
                
                return jsonify({
                    'status': 'success',
                    'response': response,
                    'is_visual': True,
                    'image': img_data
                })
            
            # For text responses
            return jsonify({
                'status': 'success',
                'response': response,
                'is_visual': False
            })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        # Restore the original plt.show function
        plt.show = original_show

recognizer = sr.Recognizer()

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': 'No audio file'})

    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_file = request.files['audio']
            audio_file.save(tmp.name)
            
            # Convert to proper WAV format
            with wave.open(tmp.name, 'rb') as wav_file:
                if (wav_file.getnchannels() != 1 or 
                    wav_file.getsampwidth() != 2 or 
                    wav_file.getframerate() not in (8000, 16000)):
                    return jsonify({
                        'status': 'error',
                        'message': 'Invalid audio format. Use 16-bit PCM mono WAV'
                    })
                with sr.AudioFile(wav_file) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)
                    return jsonify({'status': 'success', 'text': text})
                    
    except sr.UnknownValueError:
        return jsonify({'status': 'error', 'message': 'Could not understand audio'})
    except sr.RequestError as e:
        return jsonify({'status': 'error', 'message': f'API unavailable: {str(e)}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    finally:
        if tmp and os.path.exists(tmp.name):
            try:
                os.unlink(tmp.name)
            except PermissionError:
                pass

@app.route('/check-microphone', methods=['GET'])
def check_microphone():
    return jsonify({'available': True})

if __name__ == '__main__':
     app.run(port=3000, debug=True)
     