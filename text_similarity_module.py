import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os

# function for loading diffrenet module
def loading_module(path, module_url = None):
    # Import the Universal Sentence Encoder's TF Hub module
    if module_url != None:
        embed_object = hub.Module(module_url)
    else:
        embed_object = hub.Module(hub.load_module_spec(path))
    return embed_object

# function for runinng embedding module on text
def run_embedding(embed_object, text):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed_object(text))

    return message_embeddings

# function for calculating similarity between question and text
def calculating_similarity_tensor(module_url, question, text):
    question_tensor = tf.Variable(tf.convert_to_tensor(run_embedding(loading_module(module_url), question)))
    text_tensor = tf.Variable(tf.convert_to_tensor(run_embedding(loading_module(module_url), text)))
    multiply_tensor = tf.matmul(question_tensor, text_tensor, transpose_b = True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        answer = sess.run(multiply_tensor)
    return answer

# function for find sentence in text that answer question that has been asked
def find_the_most_similar_sentence(similarity_tensor, question, text):
    print('similarity score for the most similar sentence is {}'.format(np.max(similarity_tensor)))
    return np.hstack([question, text[np.argmax(similarity_tensor)]]).reshape(-1, 1)

# function for printing information about similarity tensor and printing a stack of similar sentences from text to question
def information_about_similar_sentences(similarity_tensor, question, text, threshold = 0.7, print_sorted = False):
    sorted_similarity_array = np.array([list(row) for row in sorted(zip(similarity_tensor[0], text), reverse = True)])
    for row in range(1, len(sorted_similarity_array)):
        try:
            if sorted_similarity_array[row][0] == sorted_similarity_array[row - 1][0]:
                sorted_similarity_array = np.delete(sorted_similarity_array, row, axis = 0)       
        except IndexError:
            pass
    if print_sorted:
        print(pd.DataFrame(sorted_similarity_array[1]))
    sorted_similarity_tensor = np.split(sorted_similarity_array, 2, axis = 1)[0].flatten().astype('float')
    sentences = np.array([sorted_similarity_array[i] for i in np.where(sorted_similarity_tensor > threshold)[0]])
    sentences = np.insert(sentences, 0, values = np.array([None, question[0]]).reshape(1, 2), axis=0)
    presentation_dataframe = pd.DataFrame(sentences, columns = ['similarity score', 'sentence'])
    presentation_dataframe = presentation_dataframe[['sentence', 'similarity score']]
    return presentation_dataframe

# function for calculating jaccard similarity between two sentences
def get_Jaccard_similarity(question, sentence):
    if type(question) != str:
        question = question[0]
    if type(sentence) != str:
        sentence = sentence[0]
    question_splitted = set(question.split())
    sentence_splitted = set(sentence.split())
    intersection_question_sentence = question_splitted.intersection(sentence_splitted)
    return round(len(intersection_question_sentence) / (len(question_splitted) + len(sentence_splitted) - len(intersection_question_sentence)), 3)


# function for returning filtering dissimilar sentences to question
def find_Jaccard_similarity(question, text):
    jaccard_similarity_score = np.array([get_Jaccard_similarity(question, sentence) for sentence in text])
    result = np.array([list(row) for row in sorted(zip(jaccard_similarity_score, text), reverse = True)])
    return result, jaccard_similarity_score.reshape(1, 180)

# function for sum both similarity scores
def final_result(similarity_tensor, jaccard_similarity_tensor, text):
    summation = similarity_tensor + jaccard_similarity_tensor
    final_result = np.array([list(row) for row in sorted(zip(summation[0], text), reverse = True)])
    return final_result

if __name__ == '__main__':
    pass