import os
import re
import cohere
import speech_recognition as sr
import matplotlib.pyplot as plt
import numpy as np
import wolframalpha
import wikipedia
from dotenv import load_dotenv
import textwrap
import requests
import json
import random
import pyttsx3
from PIL import Image
import io
from urllib.parse import quote
import subprocess  
from typing import Dict, Any, Tuple, Optional, List

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
        self.tts_engine = pyttsx3.init()
        self.graph_types = {
            'bar': self.generate_bar_config,
            'line': self.generate_line_config,
            'pie': self.generate_pie_config,
            'scatter': self.generate_scatter_config,
            'radar': self.generate_radar_config,
            'bubble': self.generate_bubble_config
        }

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
    
    def speak(self, text):
        """Convert text to speech (except for graphs)"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
    
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

    # ========== Enhanced Graph Generation ==========
    def generate_dynamic_data(self, topic):
        """Generate context-aware dataset based on user topic"""
        try:
            response = self.co.generate(
                model="command",
                prompt=f"""Generate 4-6 realistic numeric data points and labels for a visualization about: {topic}
                Format as JSON: {{"labels": ["label1", ...], "data": [value1, ...], "title": "chart title"}}""",
                max_tokens=200,
                temperature=0.7
            )
            result = json.loads(response.generations[0].text.strip())
            return (
                result.get("labels", ["Q1","Q2","Q3","Q4"]),
                result.get("data", [random.randint(10,100) for _ in range(4)]),
                result.get("title", f"{topic.capitalize()} Analysis")
            )
        except:
            labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            return labels, [random.randint(10,100) for _ in labels], f"{topic.capitalize()} Trends"

    def generate_bar_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": topic,
                    "data": data,
                    "backgroundColor": [f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.7)' for _ in data]
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                },
                "legend": {
                    "position": "bottom"
                }
            }
        }

    def generate_pie_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "pie",
            "data": {
                "labels": labels,
                "datasets": [{
                    "data": data,
                    "backgroundColor": [f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.7)' for _ in data]
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                }
            }
        }

    def generate_scatter_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "scatter",
            "data": {
                "datasets": [{
                    "label": topic,
                    "data": [{"x": x, "y": y} for x, y in zip(range(len(data)), data)],
                    "backgroundColor": f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.7)'
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                }
            }
        }

    def generate_radar_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "radar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": topic,
                    "data": data,
                    "borderColor": f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},1)',
                    "backgroundColor": f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},0.2)'
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                }
            }
        }

    def generate_bubble_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "bubble",
            "data": {
                "datasets": [{
                    "label": topic,
                    "data": [{
                        "x": x,
                        "y": y,
                        "r": random.randint(5,20)
                    } for x, y in zip(range(len(data)), data)]
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                }
            }
        }

    def generate_line_config(self, topic):
        labels, data, title = self.generate_dynamic_data(topic)
        return {
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": topic,
                    "data": data,
                    "borderColor": f'rgba({random.randint(0,255)},{random.randint(0,255)},{random.randint(0,255)},1)',
                    "borderWidth": 2,
                    "fill": False
                }]
            },
            "options": {
                "title": {
                    "display": True,
                    "text": title,
                    "fontSize": 16
                },
                "scales": {
                    "yAxes": [{
                        "ticks": {
                            "beginAtZero": True
                        }
                    }]
                }
            }
        }

    def display_graph_image(self, chart_config):
        """Display graph directly using QuickChart's image endpoint"""
        try:
            # Convert config to URL-safe string
            config_str = json.dumps(chart_config)
            encoded_config = quote(config_str, safe='')
            
            # Generate image URL
            image_url = f"https://quickchart.io/chart?c={encoded_config}&width=800&height=500&backgroundColor=white"
            
            # Fetch and display image
            response = requests.get(image_url)
            img = Image.open(io.BytesIO(response.content))
            
            plt.figure(figsize=(10, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            return True
        except Exception as e:
            print(f"🚨 Graph display error: {e}")
            return False

    def detect_graph_type(self, query):
        """Auto-detect chart type from query"""
        query = query.lower()
        type_mapping = {
            'bar': ['bar', 'column', 'histogram'],
            'line': ['line', 'trend', 'growth'],
            'pie': ['pie', 'percentage', 'share'],
            'scatter': ['scatter', 'distribution', 'correlation'],
            'radar': ['radar', 'spider', 'comparison'],
            'bubble': ['bubble', 'size', 'volume']
        }
        
        for graph_type, keywords in type_mapping.items():
            if any(keyword in query for keyword in keywords):
                return graph_type
        return 'bar'  # Default fallback

    def generate_api_graph(self, query):
        """Full dynamic graph generation pipeline"""
        graph_type = self.detect_graph_type(query)
        topic = re.sub(r'(graph|chart|plot|diagram|show|display)', '', query, flags=re.IGNORECASE).strip()
        
        if not topic or len(topic) < 3:
            topic = "Statistical Data"
        
        generator_func = self.graph_types.get(graph_type, self.generate_bar_config)
        chart_config = generator_func(topic)
        
        if self.display_graph_image(chart_config):
            return f"Displaying {graph_type} chart about: {topic}", True
        return "Failed to generate graph", False

    # ========== Diagram Generation Methods ==========
    def detect_visual_type(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Determine if request needs chart/diagram and return type"""
        query = query.lower()
        
        # Check for diagram keywords first (more specific matching)
        diagram_keywords = {
            'flowchart': ['flowchart', 'flow chart', 'process diagram'],
            'sequence': ['sequence diagram', 'interaction diagram'],
            'class': ['class diagram', 'uml class'],
            'state': ['state diagram', 'state machine'],
            'er': ['er diagram', 'entity relationship']
        }
        
        for diagram_type, keywords in diagram_keywords.items():
            if any(f' {keyword} ' in f' {query} ' for keyword in keywords):
                return 'diagram', diagram_type
        
        # Check for chart keywords if no diagram found
        chart_type = self.detect_graph_type(query)
        if chart_type:
            return 'chart', chart_type
        
        return None, None

    def generate_diagram(self, query: str, diagram_type: str) -> Tuple[Optional[str], bool]:
        """Generate and display diagram from query with enhanced prompts"""
        try:
            # More specific prompt based on diagram type
            diagram_prompts = {
                'flowchart': "Convert this flowchart request into Mermaid.js syntax showing the process steps and decisions",
                'sequence': "Create a sequence diagram showing the interactions between components",
                'class': "Generate a class diagram with proper classes, attributes and relationships",
                'state': "Create a state diagram showing the different states and transitions",
                'er': "Generate an ER diagram with entities, attributes and relationships"
            }
            
            prompt = diagram_prompts.get(diagram_type, "Convert this into Mermaid.js diagram syntax") + f":\n{query}\nReturn ONLY the code block with ```mermaid``` formatting"
            
            response = self.co.generate(
                model='command',
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            diagram_code = response.generations[0].text.strip()
            
            # Clean and validate the code
            if '```mermaid' in diagram_code:
                diagram_code = diagram_code.replace('```mermaid', '').replace('```', '').strip()
            
            # Add proper diagram type if missing
            if not diagram_code.startswith(('graph', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'erDiagram')):
                diagram_code = f"graph TD\n{diagram_code}"  # Default to flowchart
            
            # Save and render with larger size
            with open("temp.mmd", "w") as f:
                f.write(diagram_code)
            
            subprocess.run(
                ["mmdc", "-i", "temp.mmd", "-o", "diagram.png", 
                 "-t", "neutral", "--width", "1000", "--height", "800"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Display the diagram
            img = Image.open("diagram.png")
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            return f"Displaying {diagram_type} diagram for: {query}", True
            
        except Exception as e:
            print(f"⚠️ Diagram generation failed: {e}")
            return None, False

    # ========== Google Images Search Functionality ==========
    def get_google_images(self, query: str, num_images: int = 9) -> Tuple[Optional[str], bool]:
        """Fetch relevant Google Images for educational purposes"""
        try:
            # Use a Google Custom Search JSON API
            api_key = os.getenv('GOOGLE_API_KEY')
            cx = os.getenv('GOOGLE_CSE_ID')
            
            if not api_key or not cx:
                raise ValueError("Google API credentials not configured")
            
            # Prepare the query - adding educational terms
            search_query = f"{query} educational diagram site:edu OR site:wikipedia.org"
            
            # Make the API request
            url = f"https://www.googleapis.com/customsearch/v1?q={quote(search_query)}&key={api_key}&cx={cx}&searchType=image&num={num_images}"
            response = requests.get(url)
            response.raise_for_status()
            
            # Extract image URLs
            results = response.json()
            image_urls = [item['link'] for item in results.get('items', [])]
            
            if not image_urls:
                return None, False
                
            # Display images in a grid
            self.display_image_grid(image_urls)
            return f"Showing {len(image_urls)} educational images for: {query}", True
            
        except Exception as e:
            print(f"⚠️ Image search failed: {e}")
            return None, False

    def display_image_grid(self, image_urls: List[str], cols: int = 3) -> bool:
        """Display images in a grid layout"""
        try:
            num_images = min(len(image_urls), 9)  # Show max 9 images
            rows = (num_images + cols - 1) // cols
            
            plt.figure(figsize=(15, 5 * rows))
            plt.suptitle("Educational Image Results", fontsize=16)
            
            for i, url in enumerate(image_urls[:num_images]):
                try:
                    response = requests.get(url, timeout=5)
                    img = Image.open(io.BytesIO(response.content))
                    
                    plt.subplot(rows, cols, i+1)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(f"Image {i+1}", fontsize=8)
                    
                except Exception as e:
                    print(f"⚠️ Couldn't load image {i+1}: {e}")
                    continue
                    
            plt.tight_layout()
            plt.show()
            return True
        except Exception as e:
            print(f"⚠️ Image display failed: {e}")
            return False

    # ========== Original Matplotlib Generation ==========
    def generate_graph_code(self, query):
        """Generate Python code for visualization using Cohere"""
        try:
            prompt = f"""Generate Python matplotlib code for: {query}
            Requirements:
            - Use plt.figure(figsize=(10,6))
            - Add proper labels and title
            - Include legend if needed
            - Return ONLY the code without any explanations
            - Use sample data if not specified
            - Ensure code is self-contained
            - Use plt.show() at the end
            - Never use markdown formatting
            """
            
            response = self.co.generate(
                model='command',
                prompt=prompt,
                max_tokens=600,
                temperature=0.2
            )
            
            code = response.generations[0].text
            return self.sanitize_code(code)
        except Exception as e:
            print(f"🧑💻 Code generation failed: {e}")
            return None

    def sanitize_code(self, code):
        """Clean only real Python code from generation"""
        code = code.replace('```python', '').replace('```', '').strip()
        cleaned_lines = []
        allowed_starts = ("import ", "from ", "plt.", "np.", "x", "y", "fig", "ax", "def ", "for ", "if ", "else", "elif ", "return ", "while ")

        for line in code.splitlines():
            line = line.strip()
            if any(line.startswith(start) for start in allowed_starts) or "=" in line or line.endswith(":") or line.endswith(")"):
                cleaned_lines.append(line)

        if not any("plt.show()" in line for line in cleaned_lines):
            cleaned_lines.append('plt.show()')

        return "\n".join(cleaned_lines)

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

    # ========== Query Processing ==========
    def query_wolfram(self, question):
        """Query Wolfram Alpha"""
        try:
            res = self.wolfram_client.query(question)
            return next(res.results).text
        except:
            return None

    def process_query(self, question):
        """Enhanced processing pipeline with clear diagram/chart separation"""
        if not question:
            return "Please ask a question.", False
        
        # Check for image requests first
        if any(keyword in question.lower() for keyword in ['image','show images', 'google images', 'visual examples', 'reference images']):
            response, success = self.get_google_images(question)
            if success:
                return response, True
        
        # First detect visual type with improved detection
        visual_type, specific_type = self.detect_visual_type(question)
        
        # Handle diagram requests first (more precise matching)
        if visual_type == 'diagram' and specific_type:
            response, success = self.generate_diagram(question, specific_type)
            if success:
                return response, True
        
        # Handle chart requests (existing behavior)
        if any(keyword in question.lower() for keyword in ['graph', 'chart', 'plot']):
            # First try API-based generation
            api_response, api_success = self.generate_api_graph(question)
            if api_success:
                return api_response, True
            
            # Fallback to matplotlib generation
            print("\n⚙️ Falling back to matplotlib...")
            code = self.generate_graph_code(question)
            if code and self.execute_code(code):
                return "Here's your visualization!", True
        
        # For explicit diagram requests that weren't caught by detector
        elif 'diagram' in question.lower():
            # Try with default flowchart type
            response, success = self.generate_diagram(question, 'flowchart')
            if success:
                return response, True
        
        # Wolfram Alpha for math/science
        if self.wolfram_client and any(word in question for word in ['calculate', 'solve', '+', '-', '*', '/', '^']):
            result = self.query_wolfram(question)
            if result:
                print(f"\n Response: {result}")
                if self.microphone_available:
                    self.speak(result)
                return result, False
        
        # Cohere for general answers
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
            
            print(f"\n Response: {response.text}")
            if self.microphone_available:
                self.speak(response.text)
                
            return response.text, False
        except Exception as e:
            print(f"Cohere error: {e}")
            return "I couldn't process that request. Please try again.", False
    
    def run(self):
        """Main interactive session"""
        print("\n🧠 Smart Classroom Assistant 4.0 🤖")
        print("----------------------------------")
        print("Enhanced Features:")
        print("- Clear diagram/chart separation")
        print("- 5 diagram types: flowchart, sequence, class, state, ER")
        print("- 6 chart types with smart detection")
        print("- Google Images search capability")
        print("- Improved visual generation")
        print("----------------------------------")
        print("Try: 'Create a flowchart for login process'")
        print("Or: 'Draw a class diagram for banking system'")
        print("Or: 'Show me google images of cell structures'")
        print("Or: 'Display visual examples of neural networks'")
        
        while True:
            try:
                question = self.get_input()
                
                if not question:
                    continue
                if question.lower() in ['exit', 'quit', 'stop']:
                    break
                
                response, is_visual = self.process_query(question)
                print(f"\n📚 Response:")
                print(response)
                
                if is_visual:
                    print("\n🖼️ Visual displayed in window!")
                
            except KeyboardInterrupt:
                print("\n👋 Session ended. Keep learning!")
                break

if __name__ == "__main__":
    assistant = ClassroomAssistant()
    assistant.run()