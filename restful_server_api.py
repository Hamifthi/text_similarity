from flask import Flask
from flask import Flask, request, jsonify
from flask_restplus import Api, Resource, reqparse
from flask_mongoalchemy import MongoAlchemy
from mongoengine import connect
import text_similarity_module
import database


app = Flask(__name__)
api = Api(app)
connect('text_similarity')
embed_object, graph = text_similarity_module.loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47') 

@api.route('/create_contents', methods=["POST"])
class text_similarity(Resource):

    @api.response(201, 'content successfully created.')
    def post(self):
        title = database.Title(title = request.json['title'])
        title.save()
        text = database.Text_content(title = title, text = request.json['text'])
        text.save()
        tensor_object = text_similarity_module.run_embedding(embed_object, graph, request.json['text'])
        tensor = database.Tensor_content(title = title, tensor = tensor_object)
        tensor.save()
        return None, 201

if __name__ == '__main__':
    app.run(debug = True)
