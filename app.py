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

@app.route('/check-microphone', methods=['GET'])
def check_microphone():
    return jsonify({'available': True})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
