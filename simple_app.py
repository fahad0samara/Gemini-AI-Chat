import os
import traceback
import argparse
import requests
import sqlite3
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
from io import BytesIO
import uuid
import mimetypes
from pathlib import Path
import markdown
import pdfkit
from PIL import Image
import subprocess
from flask_cors import CORS
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import pytesseract
from googletrans import Translator
from textblob import TextBlob
import qrcode

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the Gemini AI Chat app')
parser.add_argument('--port', type=int, default=3693, help='Port to run the server on')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Configure Gemini - hardcode the working API key
GEMINI_API_KEY = "AIzaSyBgXL28M8_JAuRuFN8SWkcL8OrQWYpVkf4"  # Hardcoded working key
print(f"Using API key: {GEMINI_API_KEY}")

# Gemini API endpoints
GEMINI_CHAT_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
GEMINI_VISION_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro-vision:generateContent"

app = Flask(__name__)
CORS(app)
app.config['DEBUG'] = True
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

class AIPersona:
    def __init__(self, name, description, style, temperature):
        self.name = name
        self.description = description
        self.style = style
        self.temperature = temperature

class ChatApp:
    def __init__(self, api_key):
        self.api_key = api_key
        self.genai = genai.GenerativeModel('gemini-pro')
        self.genai_vision = genai.GenerativeModel('gemini-pro-vision')
        self.setup_database()
        self.chat_history = {}
        self.setup_file_storage()
        self.setup_ai_personas()
        self.setup_nlp_tools()
        
    def setup_nlp_tools(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.translator = Translator()
        
    def setup_ai_personas(self):
        self.personas = {
            "teacher": AIPersona(
                "Professor AI",
                "A patient and knowledgeable teacher who explains complex concepts clearly",
                "educational and detailed",
                0.7
            ),
            "coder": AIPersona(
                "Code Master",
                "An expert programmer who helps with coding and debugging",
                "technical and precise",
                0.3
            ),
            "creative": AIPersona(
                "Creative Spirit",
                "An imaginative assistant for brainstorming and creative tasks",
                "imaginative and inspiring",
                0.9
            ),
            "analyst": AIPersona(
                "Data Sage",
                "A analytical thinker who helps with data analysis and insights",
                "analytical and methodical",
                0.4
            )
        }

    def setup_database(self):
        self.conn = sqlite3.connect('chat.db')
        self.cursor = self.conn.cursor()
        
        # Create tables if they don't exist
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            role TEXT,
            message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            has_code BOOLEAN DEFAULT FALSE,
            has_file BOOLEAN DEFAULT FALSE,
            file_path TEXT,
            reactions TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS code_executions (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            code TEXT,
            language TEXT,
            output TEXT,
            error TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )''')

        # Add new tables for enhanced features
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_analysis (
            id TEXT PRIMARY KEY,
            message_id TEXT,
            image_path TEXT,
            analysis_type TEXT,
            result TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (message_id) REFERENCES messages(id)
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_summaries (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            summary TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )''')

        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS collaborative_code (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            code TEXT,
            language TEXT,
            contributors TEXT,
            version INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id)
        )''')

        self.conn.commit()

    def setup_file_storage(self):
        self.upload_folder = Path("uploads")
        self.upload_folder.mkdir(exist_ok=True)
        self.exports_folder = Path("exports")
        self.exports_folder.mkdir(exist_ok=True)

    def save_message(self, session_id, role, message, has_code=False, has_file=False, file_path=None):
        message_id = str(uuid.uuid4())
        self.cursor.execute('''
        INSERT INTO messages (id, session_id, role, message, has_code, has_file, file_path, reactions)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, role, message, has_code, has_file, file_path, '{}'))
        
        self.cursor.execute('''
        UPDATE chat_sessions 
        SET last_updated = CURRENT_TIMESTAMP 
        WHERE id = ?
        ''', (session_id,))
        
        self.conn.commit()
        return message_id

    def handle_file_upload(self, file_data, filename):
        file_id = str(uuid.uuid4())
        file_ext = Path(filename).suffix
        safe_filename = f"{file_id}{file_ext}"
        file_path = self.upload_folder / safe_filename
        
        # Decode and save file
        file_content = base64.b64decode(file_data.split(',')[1])
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        return str(file_path)

    def execute_code(self, code, language):
        execution_id = str(uuid.uuid4())
        output = ""
        error = None

        try:
            if language.lower() == 'python':
                # Create a temporary file
                temp_file = f"temp_{execution_id}.py"
                with open(temp_file, 'w') as f:
                    f.write(code)
                
                # Run the code in a separate process with timeout
                result = subprocess.run(['python', temp_file], 
                                     capture_output=True, 
                                     text=True, 
                                     timeout=10)
                output = result.stdout
                error = result.stderr

                # Clean up
                Path(temp_file).unlink()
            else:
                error = f"Language {language} not supported yet"
        except Exception as e:
            error = str(e)

        return {
            'id': execution_id,
            'output': output,
            'error': error
        }

    def export_chat(self, session_id, format='pdf'):
        # Get chat history
        self.cursor.execute('''
        SELECT m.*, s.name as session_name
        FROM messages m
        JOIN chat_sessions s ON m.session_id = s.id
        WHERE m.session_id = ?
        ORDER BY m.timestamp
        ''', (session_id,))
        
        messages = self.cursor.fetchall()
        
        if format == 'pdf':
            # Convert chat to HTML
            html_content = self.generate_chat_html(messages)
            
            # Convert HTML to PDF
            pdf_path = self.exports_folder / f"chat_export_{session_id}.pdf"
            pdfkit.from_string(html_content, str(pdf_path))
            
            return str(pdf_path)
        else:
            # Export as JSON
            json_path = self.exports_folder / f"chat_export_{session_id}.json"
            with open(json_path, 'w') as f:
                json.dump(messages, f, indent=2)
            
            return str(json_path)

    def generate_chat_html(self, messages):
        html = '''
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
                .user { background: #e3f2fd; }
                .assistant { background: #f5f5f5; }
                .timestamp { font-size: 0.8em; color: #666; }
                pre { background: #f8f9fa; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
        '''
        
        for msg in messages:
            html += f'''
            <div class="message {msg['role']}">
                <div class="timestamp">{msg['timestamp']}</div>
                {markdown.markdown(msg['message'])}
            </div>
            '''
        
        html += '</body></html>'
        return html

    def update_reactions(self, message_id, reaction):
        self.cursor.execute('SELECT reactions FROM messages WHERE id = ?', (message_id,))
        current_reactions = json.loads(self.cursor.fetchone()[0])
        
        if reaction in current_reactions:
            current_reactions[reaction] += 1
        else:
            current_reactions[reaction] = 1
        
        self.cursor.execute('UPDATE messages SET reactions = ? WHERE id = ?',
                          (json.dumps(current_reactions), message_id))
        self.conn.commit()
        
        return current_reactions

    def analyze_image(self, image_path, analysis_type):
        try:
            img = Image.open(image_path)
            
            if analysis_type == 'text_detection':
                # Extract text from image
                text = pytesseract.image_to_string(img)
                return text
            
            elif analysis_type == 'object_detection':
                # Convert PIL image to OpenCV format
                cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                
                # Use OpenCV for basic object detection
                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                objects = []
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Filter small contours
                        x, y, w, h = cv2.boundingRect(contour)
                        objects.append({
                            'x': int(x),
                            'y': int(y),
                            'width': int(w),
                            'height': int(h)
                        })
                
                return objects
            
            elif analysis_type == 'enhance':
                # Basic image enhancement
                enhanced = img.convert('RGB')
                enhanced = ImageEnhance.Contrast(enhanced).enhance(1.5)
                enhanced = ImageEnhance.Brightness(enhanced).enhance(1.2)
                enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.3)
                
                # Save enhanced image
                enhanced_path = image_path.replace('.', '_enhanced.')
                enhanced.save(enhanced_path)
                return enhanced_path
            
            elif analysis_type == 'generate_qr':
                # Generate QR code for image sharing
                qr = qrcode.QRCode(version=1, box_size=10, border=5)
                qr.add_data(image_path)
                qr.make(fit=True)
                qr_img = qr.make_image(fill_color="black", back_color="white")
                
                qr_path = image_path.replace('.', '_qr.')
                qr_img.save(qr_path)
                return qr_path
        
        except Exception as e:
            return str(e)

    def summarize_chat(self, session_id, max_length=150):
        # Get recent messages
        self.cursor.execute('''
        SELECT message FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp DESC LIMIT 50
        ''', (session_id,))
        
        messages = self.cursor.fetchall()
        if not messages:
            return "No messages to summarize"
        
        # Combine messages into a single text
        text = " ".join([msg[0] for msg in messages])
        
        # Generate summary
        summary = self.summarizer(text, max_length=max_length, min_length=30)[0]['summary_text']
        
        # Save summary
        summary_id = str(uuid.uuid4())
        self.cursor.execute('''
        INSERT INTO chat_summaries (id, session_id, summary)
        VALUES (?, ?, ?)
        ''', (summary_id, session_id, summary))
        
        self.conn.commit()
        return summary

    def analyze_sentiment(self, text):
        result = self.sentiment_analyzer(text)[0]
        return {
            'label': result['label'],
            'score': float(result['score'])
        }

    def translate_text(self, text, target_lang):
        try:
            translation = self.translator.translate(text, dest=target_lang)
            return {
                'translated_text': translation.text,
                'source_lang': translation.src,
                'pronunciation': getattr(translation, 'pronunciation', None)
            }
        except Exception as e:
            return str(e)

    def save_collaborative_code(self, session_id, code, language, contributor):
        # Get current version
        self.cursor.execute('''
        SELECT MAX(version) FROM collaborative_code
        WHERE session_id = ?
        ''', (session_id,))
        
        current_version = self.cursor.fetchone()[0] or 0
        new_version = current_version + 1
        
        # Save new version
        code_id = str(uuid.uuid4())
        self.cursor.execute('''
        INSERT INTO collaborative_code (id, session_id, code, language, contributors, version)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (code_id, session_id, code, language, json.dumps([contributor]), new_version))
        
        self.conn.commit()
        return {
            'code_id': code_id,
            'version': new_version
        }

    def get_code_history(self, session_id):
        self.cursor.execute('''
        SELECT code, language, contributors, version, timestamp
        FROM collaborative_code
        WHERE session_id = ?
        ORDER BY version DESC
        ''', (session_id,))
        
        return [{
            'code': row[0],
            'language': row[1],
            'contributors': json.loads(row[2]),
            'version': row[3],
            'timestamp': row[4]
        } for row in self.cursor.fetchall()]

chat_app = ChatApp(GEMINI_API_KEY)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_session', methods=['POST'])
def create_session():
    try:
        name = request.json.get('name', 'New Chat')
        session_id = datetime.now().strftime('%Y%m%d%H%M%S')
        
        chat_app.cursor.execute('INSERT INTO chat_sessions (id, name) VALUES (?, ?)',
                 (session_id, name))
        chat_app.conn.commit()
        
        return jsonify({
            'session_id': session_id,
            'name': name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_sessions', methods=['GET'])
def get_sessions():
    try:
        chat_app.cursor.execute('SELECT id, name, created_at FROM chat_sessions ORDER BY created_at DESC')
        sessions = [{'id': row[0], 'name': row[1], 'created_at': row[2]} for row in chat_app.cursor.fetchall()]
        return jsonify({'sessions': sessions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rename_session', methods=['POST'])
def rename_session():
    try:
        session_id = request.json.get('session_id')
        new_name = request.json.get('name')
        
        chat_app.cursor.execute('UPDATE chat_sessions SET name = ? WHERE id = ?', (new_name, session_id))
        chat_app.conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_session', methods=['POST'])
def delete_session():
    try:
        session_id = request.json.get('session_id')
        
        chat_app.cursor.execute('DELETE FROM chat_sessions WHERE id = ?', (session_id,))
        chat_app.cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        chat_app.conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/send_message', methods=['POST'])
def send_message():
    try:
        message = request.json.get('message', '')
        session_id = request.json.get('session_id', 'default')
        print(f"Received message: {message}")
        
        if message.lower() == 'clear':
            chat_app.cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
            chat_app.conn.commit()
            print("Chat session cleared")
            return jsonify({
                'message': 'Conversation cleared!',
                'type': 'system'
            })
        
        # Save user message to database
        message_id = chat_app.save_message(
            session_id=session_id,
            role='user',
            message=message
        )
        
        print("Sending message to Gemini...")
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": message
                }]
            }]
        }
        
        print(f"\nRequest URL: {GEMINI_CHAT_URL}?key={GEMINI_API_KEY}")
        print(f"Request headers: {headers}")
        print(f"Request data: {data}\n")
        
        response = requests.post(
            f"{GEMINI_CHAT_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        print(f"\nResponse status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response text: {response.text}\n")
        
        response.raise_for_status()
        
        response_data = response.json()
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            response_text = response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            response_text = "I couldn't generate a response. Please try again."
        
        chat_app.save_message(
            session_id=session_id,
            role='assistant',
            message=response_text
        )
        
        print(f"Received response: {response_text}")
        
        return jsonify({
            'message': response_text,
            'type': 'assistant'
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_message = f"""
        Error: {str(e)}
        API Key: {GEMINI_API_KEY}
        Traceback: {error_trace}
        """
        print(error_message)
        return jsonify({
            'message': error_message,
            'type': 'error'
        })

@app.route('/generate_image', methods=['POST'])
def generate_image():
    try:
        prompt = request.json.get('prompt')
        session_id = request.json.get('session_id', 'default')
        
        # Save user prompt to database
        chat_app.save_message(
            session_id=session_id,
            role='user',
            message=f"🎨 Generate image: {prompt}"
        )
        
        # Call Gemini Vision API for image generation guidance
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [{
                "parts": [{
                    "text": f"Generate a detailed description for creating an image of: {prompt}. Focus on visual details, style, composition, and mood."
                }]
            }]
        }
        
        response = requests.post(
            f"{GEMINI_VISION_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data,
            timeout=30
        )
        
        response.raise_for_status()
        response_data = response.json()
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            image_description = response_data['candidates'][0]['content']['parts'][0]['text']
        else:
            image_description = "Could not generate image description."
        
        # Save the response to database
        chat_app.save_message(
            session_id=session_id,
            role='assistant',
            message=f"🎨 Image Description:\n\n{image_description}"
        )
        
        return jsonify({
            'message': image_description,
            'type': 'assistant'
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        error_message = f"Error generating image: {str(e)}\n{error_trace}"
        print(error_message)
        return jsonify({
            'message': error_message,
            'type': 'error'
        })

@app.route('/get_history', methods=['POST'])
def get_history():
    try:
        session_id = request.json.get('session_id', 'default')
        chat_app.cursor.execute('SELECT timestamp, role, message FROM messages WHERE session_id = ? ORDER BY timestamp', (session_id,))
        history = [{'timestamp': row[0], 'role': row[1], 'message': row[2]} for row in chat_app.cursor.fetchall()]
        return jsonify({'history': history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        data = request.json
        file_data = data['file']
        filename = data['filename']
        session_id = data['session_id']
        
        file_path = chat_app.handle_file_upload(file_data, filename)
        message_id = chat_app.save_message(
            session_id=session_id,
            role='user',
            message=f'Uploaded file: {filename}',
            has_file=True,
            file_path=file_path
        )
        
        return jsonify({
            'success': True,
            'message_id': message_id,
            'file_path': file_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/execute_code', methods=['POST'])
def execute_code():
    try:
        data = request.json
        code = data['code']
        language = data['language']
        
        result = chat_app.execute_code(code, language)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_chat', methods=['POST'])
def export_chat():
    try:
        data = request.json
        session_id = data['session_id']
        format = data.get('format', 'pdf')
        
        export_path = chat_app.export_chat(session_id, format)
        return jsonify({
            'success': True,
            'export_path': export_path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/react_to_message', methods=['POST'])
def react_to_message():
    try:
        data = request.json
        message_id = data['message_id']
        reaction = data['reaction']
        
        updated_reactions = chat_app.update_reactions(message_id, reaction)
        return jsonify({
            'success': True,
            'reactions': updated_reactions
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    try:
        data = request.json
        image_path = data['image_path']
        analysis_type = data['analysis_type']
        
        result = chat_app.analyze_image(image_path, analysis_type)
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/summarize_chat', methods=['POST'])
def summarize_chat():
    try:
        data = request.json
        session_id = data['session_id']
        max_length = data.get('max_length', 150)
        
        summary = chat_app.summarize_chat(session_id, max_length)
        return jsonify({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.json
        text = data['text']
        
        result = chat_app.analyze_sentiment(text)
        return jsonify({
            'success': True,
            'sentiment': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data['text']
        target_lang = data['target_lang']
        
        result = chat_app.translate_text(text, target_lang)
        return jsonify({
            'success': True,
            'translation': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_code', methods=['POST'])
def save_code():
    try:
        data = request.json
        session_id = data['session_id']
        code = data['code']
        language = data['language']
        contributor = data['contributor']
        
        result = chat_app.save_collaborative_code(session_id, code, language, contributor)
        return jsonify({
            'success': True,
            'code_id': result['code_id'],
            'version': result['version']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_code_history', methods=['POST'])
def get_code_history():
    try:
        data = request.json
        session_id = data['session_id']
        
        history = chat_app.get_code_history(session_id)
        return jsonify({
            'success': True,
            'history': history
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Starting server on port {args.port}")
    app.run(port=args.port, debug=True)
