from flask import Flask, render_template, request, redirect, url_for
import firebase_admin
from firebase_admin import credentials, storage
import os
from werkzeug.utils import secure_filename


app = Flask(__name__)

cred = credentials.Certificate('crop-weed-5d97d-firebase-adminsdk-b8la8-ecf7d09f7f.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'crop-weed-5d97d.appspot.com'
})
bucket = storage.bucket()

# Set the allowed file extensions for uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Define a function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route to render the homepage
@app.route('/')
def home():
    # Get a list of all the files in the Firebase Storage bucket
    files = []
    blobs = bucket.list_blobs()
    for blob in blobs:
        files.append(blob.public_url)
    return render_template('index.html', files=files)

# Define a route to handle file uploads
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            return redirect(request.url)
        # Secure the filename and upload the file to Firebase Storage
        filename = secure_filename(file.filename)
        blob = bucket.blob(filename)
        blob.upload_from_string(file.read(), content_type=file.content_type)
        return redirect(url_for('home'))
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)