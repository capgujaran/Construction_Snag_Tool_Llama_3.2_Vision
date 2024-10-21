import os
import base64
import gradio as gr
from PIL import Image, ImageOps
import io
import json
from groq import Groq
import logging
import cv2
import numpy as np
import traceback
from datetime import datetime
import tempfile

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("GROQ_API_KEY is not set in environment variables")
    raise ValueError("GROQ_API_KEY is not set")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def encode_image(image):
    try:
        if isinstance(image, str):  # If image is a file path
            with open(image, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        elif isinstance(image, Image.Image):  # If image is a PIL Image
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
        elif isinstance(image, np.ndarray):  # If image is a numpy array (from video)
            is_success, buffer = cv2.imencode(".png", image)
            if is_success:
                return base64.b64encode(buffer).decode('utf-8')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    except Exception as e:
        logger.error(f"Error encoding image: {str(e)}")
        raise

def resize_image(image, max_size=(800, 800)):
    """Resize image to avoid exceeding the API size limits."""
    try:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        raise
        
def extract_frames_from_video(video, frame_points=[0, 0.5, 1], max_size=(800, 800)):
    """Extract key frames from the video at specific time points."""
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count / fps

    frames = []
    for time_point in frame_points:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_point * duration * 1000)
        ret, frame = cap.read()
        if ret:
            resized_frame = cv2.resize(frame, max_size)
            frames.append(resized_frame)
    cap.release()
    return frames

def detect_snags(file):
    """Detect snags in a single file (image or video)"""
    try:
        file_type = file.name.split('.')[-1].lower()
        if file_type in ['jpg', 'jpeg', 'png', 'bmp']:
            return detect_snags_in_image(file)
        elif file_type in ['mp4', 'avi', 'mov', 'webm']:
            return detect_snags_in_video(file)
        else:
            return "Unsupported file type. Please upload an image or video file."
    except Exception as e:
        logger.error(f"Error detecting snags: {str(e)}")
        return f"Error detecting snags: {str(e)}"

def detect_snags_in_image(image_file):
    image = Image.open(image_file.name)
    resized_image = resize_image(image)
    image_data_url = f"data:image/png;base64,{encode_image(resized_image)}"
    
    instruction = ("You are an AI assistant specialized in detecting snags in construction sites. "
                   "Your task is to analyze the image and identify any construction defects, unfinished work, "
                   "or quality issues. List each snag, categorize it, and provide a brief description. "
                   "If no snags are detected, state that the area appears to be free of visible issues.")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{instruction}\n\nAnalyze this image for construction snags and provide a detailed report."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_url
                    }
                }
            ]
        }
    ]
    
    completion = client.chat.completions.create(
        model="llama-3.2-90b-vision-preview",
        messages=messages,
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        stream=False,
        stop=None
    )
    
    return completion.choices[0].message.content

def detect_snags_in_video(video_file):
    frames = extract_frames_from_video(video_file.name)
    results = []
    
    instruction = ("You are an AI assistant specialized in detecting snags in construction sites. "
                   "Your task is to analyze the video frame and identify any construction defects, unfinished work, "
                   "or quality issues. List each snag, categorize it, and provide a brief description. "
                   "If no snags are detected, state that the area appears to be free of visible issues.")

    for i, frame in enumerate(frames):
        image_data_url = f"data:image/png;base64,{encode_image(frame)}"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{instruction}\n\nAnalyze this frame from a video (Frame {i+1}/{len(frames)}) for construction snags and provide a detailed report."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ]
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            stream=False,
            stop=None
        )
        results.append(f"Frame {i+1} analysis:\n{completion.choices[0].message.content}\n\n")
    
    return "\n".join(results)

def chat_about_snags(message, chat_history):
    try:
        messages = [
            {"role": "system", "content": "You are an AI assistant specialized in analyzing construction site snags and answering questions about them. Use the information from the initial analysis to answer user queries."},
        ]
        
        for human, ai in chat_history:
            if human:
                messages.append({"role": "user", "content": human})
            if ai:
                messages.append({"role": "assistant", "content": ai})
        
        messages.append({"role": "user", "content": message})
        
        completion = client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None
        )
        
        response = completion.choices[0].message.content
        chat_history.append((message, response))
        
        return "", chat_history
    except Exception as e:
        logger.error(f"Error during chat: {str(e)}")
        return "", chat_history + [(message, f"Error: {str(e)}")]

def generate_snag_report(chat_history):
    """
    Generate a snag report from the chat history.
    """
    report = "Construction Site Snag Detection Report\n"
    report += "=" * 40 + "\n"
    report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    for i, (user, ai) in enumerate(chat_history, 1):
        if user:
            report += f"Query {i}:\n{user}\n\n"
        if ai:
            report += f"Analysis {i}:\n{ai}\n\n"
        report += "-" * 40 + "\n"

    return report

def download_snag_report(chat_history):
    """
    Generate and provide a download link for the snag report.
    """
    report = generate_snag_report(chat_history)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snag_detection_report_{timestamp}.txt"
    
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as temp_file:
        temp_file.write(report)
        temp_file_path = temp_file.name
    
    return temp_file_path

# Custom CSS for improved styling
custom_css = """
:root {
    --primary-color: #FF6B35;
    --secondary-color: #004E89;
    --background-color: #F0F4F8;
    --text-color: #333333;
    --border-color: #CCCCCC;
}

body {
    font-family: 'Arial', sans-serif;
    background-color: var(--background-color);
    color: var(--text-color);
}

.container {
    max-width: 1200px;
    margin: auto;
    padding: 2rem;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid var(--primary-color);
}

.header h1 {
    color: var(--secondary-color);
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.subheader {
    color: var(--text-color);
    font-size: 1.1rem;
    line-height: 1.4;
    margin-bottom: 1.5rem;
    text-align: center;
}

.file-upload-container {
    border: 2px dashed var(--primary-color);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 1rem;
    background-color: #FFF5E6;
    height: 120px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}

.analyze-button {
    background-color: var(--primary-color) !important;
    color: white !important;
    font-size: 1.1rem !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 5px !important;
    width: 100%;
    transition: background-color 0.3s ease;
}

.analyze-button:hover {
    background-color: #E85A2A !important;
}

.info-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
}

.info-box {
    flex: 1;
    background-color: #E6F3FF;
    border: 1px solid var(--secondary-color);
    border-radius: 5px;
    padding: 1rem;
    font-size: 0.9rem;
    height: 200px;
    overflow-y: auto;
}

.info-box h4 {
    color: var(--secondary-color);
    margin-top: 0;
    margin-bottom: 0.5rem;
}

.info-box ul, .info-box ol {
    margin: 0;
    padding-left: 1.5rem;
}

.tag {
    display: inline-block;
    background-color: var(--primary-color);
    color: white;
    padding: 0.25rem 0.5rem;
    border-radius: 3px;
    font-size: 0.8rem;
    margin-right: 0.5rem;
    margin-bottom: 0.5rem;
}

.section-title {
    color: var(--secondary-color);
    font-size: 1.5rem;
    margin-top: 2rem;
    margin-bottom: 1rem;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 0.5rem;
}

.chatbot {
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 1rem;
    height: 400px;
    overflow-y: auto;
    background-color: white;
}

.chat-input {
    border: 1px solid var(--border-color);
    border-radius: 5px;
    padding: 0.75rem;
    width: 100%;
    font-size: 1rem;
}

.clear-button, .download-button {
    background-color: var(--secondary-color) !important;
    color: white !important;
    font-size: 1rem !important;
    padding: 0.5rem 1rem !important;
    border-radius: 5px !important;
    transition: background-color 0.3s ease;
}

.clear-button:hover, .download-button:hover {
    background-color: #003D6E !important;
}

.download-report-container {
    height: 60px;
    display: flex;
    align-items: center;
}

.footer {
    margin-top: 2rem;
    padding-top: 1rem;
    border-top: 2px solid var(--primary-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.groq-badge {
    background-color: var(--secondary-color);
    color: white;
    padding: 8px 15px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 1rem;
    display: inline-block;
}

.model-info {
    color: var(--text-color);
    font-size: 0.9rem;
}
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as iface:
    gr.HTML(
        """
        <div class="container">
            <div class="header">
                <h1>🏗️ Construction Site Snag Detector</h1>
                <p class="subheader">Enhance quality control and project management with AI-powered snag detection. Upload images or videos of your construction site to identify defects, unfinished work, and quality issues.</p>
            </div>
        """
    )
    
    with gr.Row():
        gr.HTML('<h3 class="section-title">Upload Files</h3>')
    
    with gr.Row():
        file_input = gr.File(
            label="Upload Construction Site Images or Videos",
            file_count="multiple",
            type="filepath",
            elem_classes="file-upload-container"
        )
    
    with gr.Row():
        analyze_button = gr.Button("🔍 Detect Snags", elem_classes="analyze-button")
    
    with gr.Row(elem_classes="info-row"):
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="info-box">
                    <h4>Supported File Types:</h4>
                    <ul>
                        <li>Images: JPG, JPEG, PNG, BMP</li>
                        <li>Videos: MP4, AVI, MOV, WEBM</li>
                    </ul>
                </div>
                """
            )
        
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="info-box">
                    <h4>Common Snags:</h4>
                    <div>
                        <span class="tag">Cracks</span>
                        <span class="tag">Leaks</span>
                        <span class="tag">Uneven Surfaces</span>
                        <span class="tag">Incomplete Work</span>
                        <span class="tag">Poor Finishes</span>
                        <span class="tag">Misalignments</span>
                    </div>
                </div>
                """
            )
        
        with gr.Column(scale=1):
            gr.HTML(
                """
                <div class="info-box">
                    <h4>How to use:</h4>
                    <ol>
                        <li>Upload images or videos of your construction site</li>
                        <li>Click "Detect Snags" to analyze the files</li>
                        <li>Review the detected snags in the chat area</li>
                        <li>Ask follow-up questions about the snags or request more information</li>
                        <li>Download a comprehensive report for your records</li>
                    </ol>
                </div>
                """
            )
    
    gr.HTML('<h3 class="section-title">Snag Detection Results</h3>')
    chatbot = gr.Chatbot(
        label="Snag Detection Results and Expert Chat",
        elem_classes="chatbot",
        show_share_button=False,
        show_copy_button=False
    )
    
    with gr.Row():
        msg = gr.Textbox(
            label="Ask about detected snags or quality issues",
            placeholder="E.g., 'What are the most critical snags detected?'",
            show_label=False,
            elem_classes="chat-input"
        )

    with gr.Row():
        clear = gr.Button("🗑️ Clear Chat", elem_classes="clear-button")
        download_button = gr.Button("📥 Download Report", elem_classes="download-button")

    with gr.Row(elem_classes="download-report-container"):
        report_file = gr.File(label="Download Snag Detection Report")

    gr.HTML(
        """
        <div class="footer">
            <div class="groq-badge">Powered by Groq</div>
            <div class="model-info">Model: llama-3.2-90b-vision-preview</div>
        </div>
        """
    )

    def process_files(files):
        results = []
        for file in files:
            result = detect_snags(file)
            results.append((file.name, result))
        return results

    def update_chat(history, new_messages):
        history = history or []
        for title, content in new_messages:
            history.append((None, f"File: {title}\n\n{content}"))
        return history

    analyze_button.click(
        process_files,
        inputs=[file_input],
        outputs=[chatbot],
        postprocess=lambda x: update_chat(chatbot.value, x)
    )

    msg.submit(chat_about_snags, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    download_button.click(
        download_snag_report,
        inputs=[chatbot],
        outputs=[report_file]
    )

# Launch the app
if __name__ == "__main__":
    try:
        iface.launch(debug=True)
    except Exception as e:
        logger.error(f"Error when trying to launch the interface: {str(e)}")
        logger.error(traceback.format_exc())
        print("Failed to launch the Gradio interface. Please check the logs for more information.")
