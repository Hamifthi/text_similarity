import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import time

def loading_module(path = '', module = 'https://tfhub.dev/google/universal-sentence-encoder/2'):
    graph = tf.Graph()
    with graph.as_default():
        if os.path.exists(path):
            embed_object = hub.Module(path)
        else:
            embed_object = hub.Module(module)
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        encoding_tensor = embed_object(similarity_input_placeholder)
        session = tf.train.SingularMonitoredSession()
    return graph, embed_object, similarity_input_placeholder, encoding_tensor, session

def run_embedding(text, graph, embed_object, similarity_input_placeholder, encoding_tensor, session):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    with graph.as_default():
        message_embeddings = session.run(encoding_tensor, feed_dict = {similarity_input_placeholder:text})
    return message_embeddings

def calculating_similarity_tensor(question_tensor, content_tensor):
    question_placeholder = tf.placeholder(tf.float32, shape = (1, 512))
    content_placeholder = tf.placeholder(tf.float32, shape = (None, 512))
    multiply_tensor = tf.matmul(question_placeholder, content_placeholder, transpose_b = True)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      score = sess.run(multiply_tensor, feed_dict = {question_placeholder: question_tensor, content_placeholder: content_tensor})
    return score

def produce_fake_tensorobject(number_of_sentences):
    return np.random.uniform(-1, 1, (number_of_sentences, 512))