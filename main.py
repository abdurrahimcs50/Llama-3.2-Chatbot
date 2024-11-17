from flask import Flask, request
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import os
import tempfile
from werkzeug.utils import secure_filename
import uuid

from PIL import Image, UnidentifiedImageError

# Import your existing functions
from helper import extract_text_from_image, load_pdf_documents, split_text_img_documents, split_text_documents, create_vector_store, load_and_search_vector_store, create_rag_chain

from decouple import config

GROQ_API_KEY = config('GROQ_API_KEY')
print(GROQ_API_KEY)
# Initialize Flask app
app = Flask(__name__)

# Configure file upload folder
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash layout with two columns
dash_app.layout = html.Div([ 
    dbc.Row([ 
        # Left Column for File Upload and Processing
        dbc.Col( 
            html.Div([ 
                html.H4("Upload Files and Process Data"),
                dcc.Upload( 
                    id='file-upload',
                    children=html.Button('Upload Files'),
                    multiple=True,
                    style={'margin-bottom': '10px'}
                ),
                # Wrap file upload output with loading spinner
                dcc.Loading(
                    id="loading-spinner",
                    type="circle",  # You can choose the type of spinner you like
                    children=html.Div([
                        html.Button('Process Data', id='process-data-btn', n_clicks=0, style={'margin-top': '10px'})
                    ]),
                    style={'margin-top': '20px'}
                    
                ),
                html.Div(id='upload-output')
            ]), 
            width=4, style={'padding': '20px'}
        ), 
        # Right Column for Chat Interface
        dbc.Col( 
            html.Div([ 
                html.H4("Chat with Uploaded Data"),
                dcc.Input( 
                    id='user-query', 
                    type='text', 
                    placeholder='Enter your query...', 
                    style={'width': '80%', 'margin-right': '10px'} 
                ), 
                html.Button('Submit', id='submit-query'), 
                # Wrap chat response area with loading spinner
                dcc.Loading(
                    id="loading-query",
                    type="circle",  # You can choose the type of spinner you like
                    children=html.Div(id='chat-response')
                ),
            ]), 
            width=8, style={'padding': '20px'}
        ) 
    ]), 
    # Popup/Toast for Success Message
    dbc.Toast(
        id='success-toast',
        header="Success",
        is_open=False,
        duration=4000,
        icon="success",
        dismissable=True,
        style={"position": "fixed", "top": 10, "right": 10, "width": 350}
    )
])

# Store the vector store path for simplicity
VECTOR_STORE_DB_NAME = "My_Test_App_Data"


# Callback for handling file upload
@dash_app.callback(
    Output('upload-output', 'children'),
    [Input('file-upload', 'contents')],
    [State('file-upload', 'filename')]
)
def handle_file_upload(contents, filenames):
    if contents is not None:
        messages = []
        for content, name in zip(contents, filenames):
            try:
                # Save uploaded file
                filename = secure_filename(name)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Decode base64 file content (used for Dash file uploads)
                content_type, content_string = content.split(',')
                
                import base64
                content_decoded = base64.b64decode(content_string)
                with open(file_path, "wb") as f:
                    f.write(content_decoded)

                messages.append(f"Uploaded {filename}")
            except Exception as e:
                messages.append(f"Failed to upload {name}: {str(e)}")

        return html.Div([html.P(msg) for msg in messages])

    return "Please upload files."


# Callback for processing data and saving to vector DB
@dash_app.callback(
    Output('success-toast', 'is_open'),
    Output('loading-spinner', 'children'),  # This will trigger the spinner during processing
    [Input('process-data-btn', 'n_clicks')],
    [State('file-upload', 'filename')]
)
def process_data(n_clicks, filenames):
    
    # Show spinner while processing
    if n_clicks > 0 and filenames:
        # Show spinner while processing
        spinner_content = html.Div("Processing... Please wait.")  # Update the spinner message
        for name in filenames:
            try:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(name))

                # Process file based on type
                if name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Try to extract text from the image
                    text = extract_text_from_image(file_path)
                    from langchain_core.documents import Document
                    my_doc = Document(page_content=text, metadata={"source": name})
                    print(type(my_doc))
                    # Split and store documents in vector DB
                    try:
                        splits = split_text_img_documents(my_doc)
                        print(splits, len(splits))
                        vector_store = create_vector_store(splits)
                        print(vector_store)
                        if 'Local db created':
                            return True  # Display success message
                        else:
                            return False
                    except Exception as e:
                        return f"Error storing documents: {str(e)}"
                        
                elif name.lower().endswith('.pdf'):
                    documents = []
                    pdf_docs = load_pdf_documents(file_path)
                    documents.extend(pdf_docs)
                    # Split and store documents in vector DB
                    try:
                        splits = split_text_documents(documents)
                        print(splits, len(splits))
                        vector_store = create_vector_store(splits)
                        print(vector_store)
                        if 'Local db created':
                            return True  # Display success message
                        else:
                            return False
                    except Exception as e:
                        return f"Error storing documents: {str(e)}"

            except Exception as e:
                return f"Error processing {name}: {str(e)}"

        # print(documents)

    return False


# Callback for handling chat interaction
@dash_app.callback(
    Output('chat-response', 'children'),
    [Input('submit-query', 'n_clicks')],
    [State('user-query', 'value')]
)
def chat_with_uploaded_data(n_clicks, user_input):
    if n_clicks and user_input:
        print(f"User Query: {user_input}")
        if os.path.exists(VECTOR_STORE_DB_NAME):
            try:
                response = create_rag_chain(GROQ_API_KEY, user_input, VECTOR_STORE_DB_NAME)
                print(response)
                # response = rag_chain.invoke({"input": user_input})
                return html.Div(f"Response: {response['answer']}")
            except Exception as e:
                return f"Error querying vector store: {str(e)}"

    return "Enter a query to chat."


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
