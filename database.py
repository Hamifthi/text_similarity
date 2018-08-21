from flask import Flask
from flask_mongoalchemy import MongoAlchemy
from mongoengine import *
import datetime

app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'text_similarity'
db = MongoAlchemy(app)

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