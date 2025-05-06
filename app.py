from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from classroom_assistant_api import ClassroomAssistant

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

assistant = ClassroomAssistant()

# Override plt.show to save plots instead of displaying
original_show = plt.show
plt.show = lambda: None

def save_plot():
    """Save matplotlib plot to a buffer"""
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', bbox_inches='tight')
    img_buf.seek(0)
    plt.close('all')
    return img_buf

@app.route('/process', methods=['POST'])
def process_query():
    try:
        data = request.get_json()
        query = data.get('query', '')

        if not query:
            return jsonify({
                'status': 'error',
                'message': 'No query provided',
                'text_response': 'Please ask a question'
            })

        # Process the query through the assistant
        text_response, is_visual = assistant.process_query(query)

        # Prepare response data
        response_data = {
            'status': 'success',
            'text_response': text_response,
            'is_visual': is_visual,
            'image': None
        }

        # Handle visual response if needed
        if is_visual:
            if os.path.exists("diagram.png"):
                with open("diagram.png", "rb") as f:
                    response_data['image'] = base64.b64encode(f.read()).decode('utf-8')
                os.remove("diagram.png")
            else:
                img_buf = save_plot()
                response_data['image'] = base64.b64encode(img_buf.getvalue()).decode('utf-8')

        return jsonify(response_data)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'text_response': f"Error processing request: {str(e)}"
        })

if __name__ == '__main__':
    app.run(port=3000, debug=True)
