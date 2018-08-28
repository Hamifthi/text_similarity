from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
from pymongo import MongoClient
import text_similarity_module
import database
import numpy as np



app = Flask(__name__)
db = MongoEngine(app)

# graph, embed_object, similarity_input_placeholder, encoding_tensor, session = text_similarity_module.loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')
module_calculate = text_similarity_module.calculating_similarity_tensor()

connection = MongoClient('localhost', 27017)
db_object = connection['text_similarity']

if 'All_contents' not in db_object.collection_names():
    database.All_contents.objects().update(set__titles = [], upsert = True)

@app.route('/create_contents', methods=["POST"])
def create_contents():
    title = database.Title(title = request.json['title'])
    title.save()
    database.All_contents.objects().update(push__titles = request.json['title'])
    database.Text_content(title = title, text = request.json['text']).save()
    # tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
    #                                                     embed_object, similarity_input_placeholder,
    #                                                     encoding_tensor, session)
    tensor_object = text_similarity_module.produce_fake_tensorobject(len(request.json['text']))
    array_of_tensors = np.vstack((np.array(database.All_contents.objects().get().tensors).reshape(-1, 512), tensor_object))
    database.All_contents.objects().update(set__tensors = array_of_tensors)
    database.Tensor_content(title = title, tensor = tensor_object).save()
    return ('content successfully created', 201)

@app.route('/update_contents/<title>', methods=["POST"])
def update_contents(title):
    id = database.Title.objects(title = title)[0].id
    database.Text_content.objects(title = id).update(set__text = request.json['text'])
    # tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
    #                                                     embed_object, similarity_input_placeholder,
    #                                                     encoding_tensor, session)
    tensor_object = text_similarity_module.produce_fake_tensorobject(len(request.json['text']))
    database.Tensor_content.objects(title = id).update(set__tensor = tensor_object)
    return ("content successfully updated", 200)

@app.route('/delete_contents/<title>', methods=["DELETE"])
def delete_contents(title):
    id = database.Title.objects(title = title)[0].id
    database.Title.objects(title = title).delete()
    database.Text_content.objects(title = id).delete()
    database.Tensor_content.objects(title = id).delete()
    return ("content successfully deleted", 202)

@app.route('/ask', methods=["POST"])
def ask():
    question = request.json['question']
    # question_tensor = text_similarity_module.run_embedding(question, graph,
    #                                                     embed_object, similarity_input_placeholder,
    #                                                     encoding_tensor, session)
    question_tensor = text_similarity_module.produce_fake_tensorobject(1)
    database.Question(question = question, question_tensor = question_tensor).save()
    content_tensor = database.All_contents.objects().get().tensors
    all_content_score = module_calculate(question_tensor, text_tensor)
    return ('content successfully created', 201)

if __name__ == '__main__':
    app.run(debug = True)