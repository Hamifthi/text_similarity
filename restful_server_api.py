from flask import Flask, request, jsonify
from flask_mongoengine import MongoEngine
from pymongo import MongoClient
import sys
sys.path.append('E:/Hamed/Projects/Python/Text Similarity/modules')
import text_similarity_module
import database
import numpy as np



app = Flask(__name__)
db = MongoEngine(app)

# graph, embed_object, similarity_input_placeholder, encoding_tensor, session = text_similarity_module.loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')

connection = MongoClient('localhost', 27017)
db_object = connection['text_similarity']

@app.route('/create_contents', methods=["POST"])
def create_contents():
    # first create a content with a title and an empty array of sentence ids
    content = database.Content(title = request.json['title']).save()
    # create individual sentences with a text and a reference to the content it belogs
    for sentence in request.json['text']:
        text = database.Sentence(content_referecnce = content, text = sentence).save()
    # get all of sentences's ids that belogs to that specific content and save it in the array ids for that content
    ids = database.Sentence.objects(content_referecnce = content).distinct('_id')
    database.Content.objects(title = request.json['title']).update(push_all__array_of_ids = ids)
    '''tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)'''
    # once calculate the sentences tensor of specific content and then save the tensors one by one with the id of it's sentences
    tensor_object = text_similarity_module.produce_fake_tensorobject(len(request.json['text']))
    for number in range(len(tensor_object)):
        sentence_tensor = database.Sentence_Tensor(sentence_referecnce = ids[number], tensor = tensor_object[number]).save()
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
    all_content_score = module_calculate(question_tensor, content_tensor)
    print(all_content_score)
    return ('content successfully created', 201)

if __name__ == '__main__':
    app.run(debug = True)