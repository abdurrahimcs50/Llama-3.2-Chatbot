import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import time
import base64
import io
from flask import Flask
from PIL import Image
import zipfile
import PyPDF2

# Import your existing functions
from helper import extract_text_from_image, load_pdf_documents, split_text_img_documents, split_text_documents, create_vector_store, load_and_search_vector_store, create_rag_chain

from decouple import config
import os
import tempfile
from werkzeug.utils import secure_filename
import base64

GROQ_API_KEY = config('GROQ_API_KEY')
print(GROQ_API_KEY)

# Store the vector store path for simplicity
VECTOR_STORE_DB_NAME = "My_Test_App_Data"

# Initialize Flask server
server = Flask(__name__)

# Configure file upload folder
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
server.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Dash app with Flask server
app = dash.Dash(__name__, server=server, suppress_callback_exceptions=True, url_base_pathname='/', external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "ChatBotApp"

# Custom CSS styles for better UI
styles = """
<style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f7fc;
        margin: 0;
    }
    .header {
        background-color: #4CAF50;
        color: white;
        padding: 20px;
        text-align: center;
        font-size: 2em;
    }
    .container {
        display: flex;
        padding: 20px;
        justify-content: space-between;
    }
    .sidebar {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 30%;
    }
    .upload-area {
        border: 2px dashed #4CAF50;
        padding: 20px;
        text-align: center;
        border-radius: 8px;
        background-color: #f9f9f9;
        cursor: pointer;
    }
    .upload-area:hover {
        background-color: #e7f7e7;
    }
    .upload-text {
        margin-top: 10px;
        font-size: 14px;
        color: #555;
    }
    .chat-container {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        width: 65%;
        flex-direction: column;
        justify-content: space-between;
    }
    .conversation {
        overflow-y: auto;
        max-height: 400px;
        margin-bottom: 20px;
        padding: 10px;
        border-radius: 8px;
        background-color: #f9f9f9;
        border: 1px solid #ccc;
    }

    .user-message {
        background-color: #e0f7fa;
        align-self: flex-start;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .bot-message {
        background-color: #e8f5e9;
        align-self: flex-end;
    }
    .input-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .input-container input {
        width: 85%;
        padding: 12px;
        border-radius: 20px;
        border: 1px solid #ccc;
    }
    .input-container button {
        padding: 12px 20px;
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 20px;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .input-container button:hover {
        background-color: #45a049;
    }
    .upload-status, .uploaded-files-list {
        font-size: 14px;
        color: #555;
        margin-top: 15px;
    }
</style>
"""

app.index_string = styles + app.index_string

# Layout of the Dash app
app.layout = html.Div(
    [
        # Top header
        html.Header(
            className="header",
            children="ChatBotApp"
        ),
        
        # Main content layout with two sidebars
        html.Div(
            [
                # Left Sidebar: File Upload
                html.Div(
                    [
                        html.H2("Upload Files", style={"textAlign": "center"}),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(
                                [
                                    html.H4("Drag and Drop or Click to Upload", style={"marginBottom": "5px"}),
                                    html.P("Accepts .zip, .pdf, .jpg, .png files", style={"fontSize": "14px", "color": "#555"})
                                ],
                                className="upload-area"
                            ),
                            multiple=True,
                            accept=".zip,.pdf,image/*",
                        ),
                        html.Div(id="upload-status", className="upload-status"),
                        html.Div(id="uploaded-files-list", className="uploaded-files-list"),
                    ],
                    className="sidebar",  # Left Sidebar class
                ),
                
                # Right Sidebar: Chat Interface
                html.Div(
                    [
                        # Chat Interface Section
                        html.Div(
                            id="conversation",
                            children=[
                                html.Div("You: Hi, There!", className="user-message", style={
                                    "background-color": "#e0f7fa",
                                    "align-self": "flex-start",
                                    "margin-bottom": "10px",
                                    "margin-top": "10px",
                                    "padding": "10px",
                                    "border-radius": "8px",
                                    "background-color": "#e0f7fa",
                                    "border": "1px solid #ccc"
                                    
                                    }),
                                html.Div("Bot: Hello! How can I assist you today?", className="bot-message", style={
                                    "background-color": "#e8f5e9",
                                    "align-self": "flex-end",
                                    "padding": "10px",
                                    "border-radius": "8px",
                                    "background-color": "#e8f5e9",
                                    "border": "1px solid #ccc"
                                    }),
                            ],
                            className="conversation",
                        ),
                        dcc.Loading(
                            id="loading-indicator",
                            type="circle",
                            children=[
                                html.Div(
                                    className="input-container",
                                    children=[
                                        dcc.Input(
                                            id="user_input",
                                            type="text",
                                            placeholder="Type your message here...",
                                        ),
                                        html.Button(
                                            "Send",
                                            id="send_button",
                                            n_clicks=0,
                                        ),
                                    ]
                                ),
                            ]
                        ),
                    ],
                    className="chat-container",  # Right Sidebar class
                ),
            ],
            className="container",  # Main container for both sidebars
        ),
    ],
)

# File Processing: Handle file uploads

@app.callback(
    [Output("upload-status", "children"),
     Output("uploaded-files-list", "children")],
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
)
def handle_file_upload(contents, filenames):
    if contents is not None:
        messages = []
        for content, name in zip(contents, filenames):
            try:
                # Save uploaded file
                filename = secure_filename(name)
                file_path = os.path.join(server.config['UPLOAD_FOLDER'], filename)

                # Decode base64 file content
                content_type, content_string = content.split(',')
                content_decoded = base64.b64decode(content_string)
                with open(file_path, "wb") as f:
                    f.write(content_decoded)
                    
                # Debug: Check the file size and path before processing
                if os.path.exists(file_path):
                    messages.append(f"ZIP file {filename} found at {file_path} with size {os.path.getsize(file_path)} bytes.")
                else:
                    messages.append(f"ZIP file {filename} not found at {file_path}.")

                # File-specific processing
                if filename.endswith(".pdf"):
                    print(file_path, "This is File Path")
                    print('#'*20)
                    # load_pdf_documents,
                    
                    load_pdf = load_pdf_documents(file_path)
                    print(load_pdf, "This is Load PDF")
                    print('#'*20)
                    # with open(file_path, "rb") as f:
                    #     pdf_reader = PyPDF2.PdfReader(f)
                    #     pdf_text = " ".join([page.extract_text() for page in pdf_reader.pages])
                    messages.append(f"Extracted text from {filename}.")
                
                elif filename.endswith((".png", ".jpg", ".jpeg")):
                    image = Image.open(file_path)
                    extracted_text = extract_text_from_image(image)
                    messages.append(f"Extracted text from image {filename}.")
                
                elif filename.endswith(".zip"):
                    try:
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.testzip()  # Test the ZIP file for any errors before extracting
                            extracted_folder_path = os.path.join(server.config['UPLOAD_FOLDER'], "extracted")
                            os.makedirs(extracted_folder_path, exist_ok=True)
                            zip_ref.extractall(extracted_folder_path)

                            for root, dirs, files in os.walk(extracted_folder_path):
                                for file in files:
                                    if file == ".DS_Store":
                                        continue

                                    file_path = os.path.join(root, file)
                                    try:
                                        if file.endswith(".pdf"):
                                            pdf_text = load_pdf_documents(file_path)
                                            print(pdf_text, "This is PDF Text")
                                            print('#'*20)
                                            messages.append(f"Extracted text from PDF: {file_path}")
                                        elif file.endswith((".png", ".jpg", ".jpeg")):
                                            image = Image.open(file_path)
                                            extracted_text = extract_text_from_image(image)
                                            messages.append(f"Extracted text from image: {file_path}")
                                        else:
                                            messages.append(f"File type {file.split('.')[-1].upper()} not processed: {file_path}")
                                    except Exception as e:
                                        messages.append(f"Error while processing file {file_path}: {str(e)}")
                                        
                    except zipfile.BadZipFile:
                        messages.append(f"Failed to unzip {filename}: The file is not a valid ZIP archive.")
                    except Exception as e:
                        messages.append(f"Error while processing {filename}: {str(e)}")

                else:
                    messages.append(f"Uploaded {filename}, but it is not a ZIP file.")
            except Exception as e:
                messages.append(f"Failed to upload {name}: {str(e)}")
                
        return "File processing complete!", html.Ul([html.Li(msg) for msg in messages])

    return "Please upload files."


# Callback to handle chat updates
# @app.callback(
#     Output("conversation", "children"),
#     [Input("send_button", "n_clicks")],
#     [State("user_input", "value"),
#      State("conversation", "children")],
# )
# def update_chat(n_clicks, user_input, conversation):
#     if n_clicks > 0 and user_input:
#         bot_response = f"Bot: You said: {user_input}"
#         # bot_response = create_rag_chain(GROQ_API_KEY, user_input, VECTOR_STORE_DB_NAME)
#         # print(bot_response)
#         conversation.append(html.Div(user_input, className="user-message"))
#         conversation.append(html.Div(bot_response, className="bot-message"))
#         return conversation
#     return conversation

@app.callback(
    Output("conversation", "children"),
    Input("send_button", "n_clicks"),
    State("user_input", "value"),
    State("conversation", "children"),
)
def update_chat(n_clicks, user_input, conversation):
    if n_clicks > 0 and user_input:
        try:
            bot_response = create_rag_chain(GROQ_API_KEY, user_input, VECTOR_STORE_DB_NAME)
            
            # add styles on bot and user input
            # bot_response = f"Bot: {bot_response}"
            # user_input = f"You: {user_input}"
            conversation.append(html.Div(f"You: {user_input}", className="user-message", style={
                "background-color": "#e0f7fa",
                "align-self": "flex-start",
                "margin-bottom": "10px",
                "margin-top": "10px",
                "padding": "10px",
                "border-radius": "8px",
                "background-color": "#e0f7fa",
                "border": "1px solid #ccc"
            }))
            conversation.append(html.Div(f"Bot: {bot_response}", className="bot-message", style={
                "background-color": "#e8f5e9",
                "align-self": "flex-end",
                "padding": "10px",
                "border-radius": "8px",
                "background-color": "#e8f5e9",
                "border": "1px solid #ccc"
            }))
            print("X"*50)
            print(type(bot_response), "This is Type of Bot Response")
            print(bot_response, "This is Bot Response")
            print(conversation, "This is Conversation")
            print(user_input, "This is User Input")
            print("X"*50)
            return conversation
        except Exception as e:
            print(f"Error generating bot response: {e}")
            bot_response = "Sorry, I couldn't process your request."

        # updated_conversation = conversation + [
        #     html.Div(f"You: {user_input}", className="user-message"),
        #     html.Div(f"Bot: {bot_response}", className="bot-message"),
        # ]
        
        
        # return updated_conversation
    return conversation





if __name__ == "__main__":
    server.run(debug=True)
