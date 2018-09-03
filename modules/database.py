from flask import Flask
from flask_mongoengine import MongoEngine
from mongoengine import *
import datetime

app = Flask(__name__)
app.config['MONGODB_DB'] = 'text_similarity'
app.config['MONGODB_HOST'] = 'mongodb://localhost/text_similarity'
app.config['MONGODB_PORT'] = 27017
db = MongoEngine(app)

class Content(Document):
    title = StringField(required = True, unique = True)
    array_of_ids = ListField(ReferenceField('Sentence'))

class Sentence(Document):
    content_reference = ReferenceField(Content)
    text = StringField()

class Sentence_Tensor(Document):
    sentence_reference = ReferenceField(Sentence)
    tensor = ListField(FloatField())

class Question(Document):
    text = StringField(required = True, unique = True)
    question_tensor = ListField(ListField(FloatField()))
    time = DateTimeField(default =  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result = ListField(ListField(StringField()))

if __name__ == '__main__':
    main()