import os
import re
import cohere
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import wolframalpha
import wikipedia
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import textwrap

# Configure Wikipedia
wikipedia.set_lang("en")
wikipedia.set_rate_limiting(True)

load_dotenv()

class ClassroomAssistant:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.co = cohere.Client(os.getenv('COHERE_API_KEY'))
        self.wolfram_client = self.init_wolfram()
        self.microphone_available = self.check_microphone()
        self.context_history = []
        
    def init_wolfram(self):
        """Initialize Wolfram client with error handling"""
        app_id = os.getenv('WOLFRAM_APP_ID')
        return wolframalpha.Client(app_id) if app_id else None
    
    def check_microphone(self):
        """Check microphone availability"""
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            return True
        except OSError:
            return False
    
    def get_input(self):
        """Get input with enhanced error handling"""
        if self.microphone_available:
            try:
                with sr.Microphone() as source:
                    print("\n🎤 Listening... (speak clearly)")
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                    text = self.recognizer.recognize_google(audio)
                    print(f"🗣️ Raw input: {text}")
                    return self.correct_spelling(text)
            except sr.WaitTimeoutError:
                print("⏳ Listening timed out. Please type your question.")
            except Exception as e:
                print(f"🔇 Voice input error: {str(e)}")
        
        text = input("\n📝 Type your question: ")
        return self.correct_spelling(text)
    
    def correct_spelling(self, text):
        """Advanced spelling correction using Cohere"""
        try:
            response = self.co.generate(
                model='command',
                prompt=f"Correct any spelling mistakes in this text, keeping the technical terms intact:\n{text}\nCorrected:",
                max_tokens=50,
                temperature=0.3
            )
            corrected = response.generations[0].text.strip()
            cleaned = re.sub(r'[^a-zA-Z0-9\s\-\+\*\/\^\(\)]', '', corrected)
            print(f"🔡 Corrected to: {cleaned}")
            return cleaned
        except Exception as e:
            print(f"⚠️ Spelling correction failed: {e}")
            return text
    
    def generate_graph_code(self, query):
        """Generate Python code for visualization using Cohere"""
        try:
            prompt = f"""Generate Python matplotlib code for: {query}
            Requirements:
            - Use plt.figure(figsize=(8,6))
            - Add proper labels and title
            - Include legend if needed
            - Return ONLY the code without any explanations
            - Use sample data if not specified
            - Ensure code is self-contained
            - Use plt.show() instead of return
            - Never use markdown formatting
            """
            
            response = self.co.generate(
                model='command',
                prompt=prompt,
                max_tokens=500,
                temperature=0.2
            )
            
            code = response.generations[0].text
            return self.sanitize_code(code)
        except Exception as e:
            print(f"🧑💻 Code generation failed: {e}")
            return None

    def sanitize_code(self, code):
        """Improved: Clean only real Python code from generation"""
        # Remove markdown formatting
        code = code.replace('```python', '').replace('```', '')

        # Keep only lines that look like real code
        cleaned_lines = []
        allowed_starts = ("import ", "from ", "plt.", "np.", "x", "y", "fig", "ax", "def ", "for ", "if ", "else", "elif ", "return ", "while ")

        for line in code.splitlines():
            line = line.strip()
            if any(line.startswith(start) for start in allowed_starts) or "=" in line or line.endswith(":") or line.endswith(")"):
                cleaned_lines.append(line)

        # Force plt.show() if missing
        if not any("plt.show()" in line for line in cleaned_lines):
            cleaned_lines.append('plt.show()')

        # Combine cleaned lines
        cleaned_code = "\n".join(cleaned_lines)
        return cleaned_code


    def execute_code(self, code):
        """Safely execute generated code"""
        try:
            exec_globals = {'plt': plt, 'np': np}
            exec(code, exec_globals)
            plt.tight_layout()
            plt.show()
            return True
        except Exception as e:
            print(f"🚨 Execution error: {e}")
            return False

    def format_code(self, code):
        """Format code for display"""
        return textwrap.indent(code, '    ')

    def process_query(self, question):
        """Enhanced processing pipeline with code generation"""
        if not question:
            return "Please ask a question.", None
        
        # Check for visualization requests
        if any(keyword in question.lower() for keyword in ['graph', 'chart', 'plot', 'diagram']):
            print("\n⚙️ Generating visualization code...")
            code = self.generate_graph_code(question)
            
            if code:
                print("\n🖥️ Generated Code:")
                print(self.format_code(code))
                
                print("\n📈 Rendering graph...")
                if self.execute_code(code):
                    return "Here's your visualization!", True
                return "Failed to generate graph", False
        
        # Try Wolfram Alpha for math/science
        if self.wolfram_client and any(word in question for word in ['calculate', 'solve', '+', '-', '*', '/', '^']):
            result = self.query_wolfram(question)
            if result:
                return result, False
        
        # Try Cohere for general answers
        try:
            response = self.co.chat(
                message=question,
                model='command',
                temperature=0.5,
                chat_history=self.context_history[-5:],
                prompt_truncation='AUTO'
            )
            self.context_history.append({"role": "USER", "message": question})
            self.context_history.append({"role": "CHATBOT", "message": response.text})
            return response.text, False
        except Exception as e:
            print(f"Cohere error: {e}")
            return "I couldn't process that request. Please try again.", False
    
    def run(self):
        """Enhanced interactive session"""
        print("\n🧠 Smart Classroom Assistant 🤖")
        print("------------------------------")
        print("Features:")
        print("- Voice/Text input with spelling correction")
        print("- Dynamic code generation for visualizations")
        print("- Math/Science problem solving")
        print("- Context-aware conversations")
        print("------------------------------")
        
        while True:
            try:
                question = self.get_input()
                
                if not question:
                    continue
                if question.lower() in ['exit', 'quit']:
                    break
                
                response, is_visual = self.process_query(question)
                print(f"\n📚 Response:")
                print(response)
                
                if is_visual:
                    print("\n🖼️ Check the visualization window!")
                
            except KeyboardInterrupt:
                print("\n👋 Session ended. Keep learning!")
                break

if __name__ == "__main__":
    assistant = ClassroomAssistant()
    assistant.run()