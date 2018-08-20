from flask import Flask
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, reqparse
from flask_mongoalchemy import MongoAlchemy
# import text_similarity_module
import database

app = Flask(__name__)
api = Api(app)
# embed_object = text_similarity_module.loading_module('https://tfhub.dev/google/universal-sentence-encoder/2')

@api.route('/create_contents', methods=["POST"])
class text_similarity(Resource):

    api.response(201, 'content successfully created.')
    def post(self):
        author = request.json['author']
        text = request.json['content']
        tensor = request.json['tensor']
        # tensor = text_similarity_module.run_embedding(embed_object, text)
        content = database.content(author = author, text = text, tensor = tensor)
        content.save()
        return 'your content saved successfully and text tensor calculated'

if __name__ == '__main__':
    app.run(debug = True)
