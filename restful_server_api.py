from flask import Flask
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, reqparse
from flask_mongoalchemy import MongoAlchemy

app = Flask(__name__)
app.config['MONGOALCHEMY_DATABASE'] = 'library'
api = Api(app)
db = MongoAlchemy(app)

class content(db.Document):
    author = db.StringField()
    text = db.StringField()

@api.route('/create_contents', methods=["POST"])
class text_similarity(Resource):

    api.response(201, 'content successfully created.')
    def post(self):
        author = request.json['author']
        text = request.json['content']
        example = content(author = author, text = text)
        example.save()
        return 201

if __name__ == '__main__':
    app.run(debug = True)
    print(dict_of_contents)
