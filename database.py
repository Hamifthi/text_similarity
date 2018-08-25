from flask import Flask
from flask_mongoengine import MongoEngine
from mongoengine import *
import datetime

app = Flask(__name__)
app.config['MONGODB_DB'] = 'text_similarity'
app.config['MONGODB_HOST'] = 'mongodb://localhost/text_similarity'
app.config['MONGODB_PORT'] = 27017
db = MongoEngine(app)

class Title(Document):
    title = StringField(required = True, unique = True)

class Text_content(Document):
    title = ReferenceField(Title)
    text = ListField(StringField())
    time = DateTimeField(default =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

class Tensor_content(Document):
    title = ReferenceField(Title)
    tensor = ListField(ListField(FloatField()))
    time = DateTimeField(default =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))