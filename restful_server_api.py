from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
import text_similarity_module
import database


app = Flask(__name__)
db = MongoEngine(app)
graph, embed_object, similarity_input_placeholder, encoding_tensor, session = text_similarity_module.loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')

@app.route('/create_contents', methods=["POST"])
def create_contents():
    title = database.Title(title = request.json['title'])
    title.save()
    text = database.Text_content(title = title, text = request.json['text'])
    text.save()
    tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)
    tensor = database.Tensor_content(title = title, tensor = tensor_object)
    tensor.save()
    return ('content successfully created', 201)

@app.route('/update_contents/<title>', methods=["POST"])
def update_contents(title):
    id = database.Title.objects(title = title)[0].id
    database.Text_content.objects(title = id).update(set__text = request.json['text'])
    tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)
    database.Tensor_content.objects(title = id).update(set__tensor = tensor_object)
    return ("content successfully updated", 200)

@app.route('/delete_contents/<title>', methods=["DELETE"])
def delete_contents(title):
    id = database.Title.objects(title = title)[0].id
    database.Title.objects(title = title).delete()
    database.Text_content.objects(title = id).delete()
    database.Tensor_content.objects(title = id).delete()
    return ("content successfully deleted", 202)

if __name__ == '__main__':
    app.run(debug = True)