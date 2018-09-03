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

graph, embed_object, similarity_input_placeholder, encoding_tensor, session = text_similarity_module.loading_module('E:/Hamed/Projects/Python/Text Similarity/Tensorflow Hub Module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')

connection = MongoClient('localhost', 27017)
db_object = connection['text_similarity']

@app.route('/create_contents', methods=["POST"])
def create_contents():
    # first create a content with a title and an empty array of sentence ids
    content = database.Content(title = request.json['title']).save()
    # create individual sentences with a text and a reference to the content it belogs
    for sentence in request.json['text']:
        text = database.Sentence(content_reference = content, text = sentence).save()
    # get all of sentences's ids that belogs to that specific content and save it in the array ids for that content
    ids = database.Sentence.objects(content_reference = content).distinct('_id')
    database.Content.objects(title = request.json['title']).update(push_all__array_of_ids = ids)
    # once calculate the sentences tensor of specific content and then save the tensors one by one with the id of it's sentences
    tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)
    for number in range(len(tensor_object)):
        sentence_tensor = database.Sentence_Tensor(sentence_reference = ids[number], tensor = tensor_object[number]).save()
    return ('content successfully created', 201)

@app.route('/update_contents', methods=["POST"])
def update_contents():
    # first find the content and save old ids then delete old ids from content collection in database also delete old sentences content refrences
    content = database.Content.objects(title = request.json['title']).get()
    old_ids = [object.id for object in content.array_of_ids]
    database.Content.objects(title = request.json['title']).update(unset__array_of_ids = 0)
    database.Sentence.objects(content_reference = content).update(unset__content_reference = 0)
    # recreate new sentences with reference to content and tensors with reference to new sentences also save new sentences's ids in array of ids
    for sentence in request.json['text']:
        text = database.Sentence(content_reference = content, text = sentence).save()
    ids = database.Sentence.objects(content_reference = content).distinct('_id')
    database.Content.objects(title = request.json['title']).update(push_all__array_of_ids = ids)
    tensor_object = text_similarity_module.run_embedding(request.json['text'], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)
    for number in range(len(tensor_object)):
        sentence_tensor = database.Sentence_Tensor(sentence_reference = ids[number], tensor = tensor_object[number]).save()
    # delete old sentences that don't have content reference also delete old tensors with the ids of old sentences
    database.Sentence.objects(content_reference = None).delete()
    for id in old_ids:
        database.Sentence_Tensor.objects(sentence_reference = id).delete()
    return ("content successfully updated", 200)

@app.route('/delete_contents/<title>', methods=["DELETE"])
def delete_contents(title):
    # first find content
    content = database.Content.objects(title = title).get()
    # get all senteces objects that belongs to founded content
    sentences = database.Sentence.objects(content_reference = content).all()
    # deletes all tensor objects belongs to sentences
    for sentence_object in sentences:
        database.Sentence_Tensor.objects(sentence_reference = sentence_object.id).delete()
    # delete all sentences
    database.Sentence.objects(content_reference = content).delete()
    # delete content
    database.Content.objects(title = title).delete()
    return ("content successfully deleted", 202)

@app.route('/ask', methods=["POST"])
def ask():
    # create a variable called question to work simpler with it then calculate question tensor and save it on database
    question = request.json['question']
    question_tensor = text_similarity_module.run_embedding([question], graph,
                                                        embed_object, similarity_input_placeholder,
                                                        encoding_tensor, session)
    # check the question is not present in database
    try:
        database.Question(text = question, question_tensor = question_tensor).save()
    except:
        return ('question already asked before', 409)
    # get all tensor objects from database for calculating similarity and also sentences reference for specify which score is for which sentence
    tensor_objects = database.Sentence_Tensor.objects().all()
    content_tensor = np.array([tensor_object.tensor for tensor_object in tensor_objects])
    sentences_id = np.array([tensor_object.sentence_reference.id for tensor_object in tensor_objects]).reshape(-1, 1)
    # similarity calculation part
    all_content_score = text_similarity_module.calculating_similarity_tensor(question_tensor, content_tensor).reshape(-1, 1)
    # stacking scores with ids
    all_content_score_and_reference = np.hstack([all_content_score, sentences_id])
    # order the scores in decreasing order
    all_content_score_and_reference = all_content_score_and_reference[all_content_score_and_reference[:, 0].argsort()][-10:]
    # putting all sentences in one list for calculating jaccard similarity
    all_sentences = [database.Sentence.objects(id = sentence_id).get().text for sentence_id in all_content_score_and_reference[:, 1]]
    # calculating jaccard similarity
    jaccard = text_similarity_module.find_Jaccard_similarity(question, all_sentences)
    # print(all_sentences)
    sum_module_jaccard = np.add(all_content_score_and_reference[:, 0], jaccard[:, 0].astype('float')).reshape(-1, 1)
    all_content_score_and_reference = np.hstack([sum_module_jaccard, np.array(all_content_score_and_reference[:, 1]).reshape(-1, 1)])
    # create a list of results that are return as responce
    responce = []
    for i in range(len(all_content_score_and_reference)):
        all_together = [all_content_score_and_reference[i, 0]]
        sentence_reference = database.Sentence.objects(id = all_content_score_and_reference[i, 1]).get()
        all_together.append(sentence_reference.text)
        all_together.append(database.Content.objects(id = sentence_reference.content_reference.id).get().title)
        responce.append(all_together)
    
    # reverse the list to show results in descending order
    responce = list(reversed(responce))
    database.Question.objects(text = question).update(push_all__result = responce)
    return (jsonify(responce), 200)

if __name__ == '__main__':
    app.run(debug = True)