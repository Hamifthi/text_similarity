from flask import Flask
from flask_mongoalchemy import MongoAlchemy

app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'library'
db = MongoAlchemy(app)

class content(db.Document):
    author = db.StringField()
    text = db.ListField(db.StringField())
    tensor = db.ListField(db.ListField(db.FloatField()))