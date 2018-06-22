from flask import Flask
app = Flask(__name__)
from flaskapp import views
UPLOAD_FOLDER = 'flaskapp/static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'dawnis is cool'
app.config['SESSION_TYPE'] = 'filesystem'
