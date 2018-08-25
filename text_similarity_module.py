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
        session = tf.train.MonitoredSession()
    return graph, embed_object, similarity_input_placeholder, encoding_tensor, session

def run_embedding(text, graph, embed_object, similarity_input_placeholder, encoding_tensor, session):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    with graph.as_default():
        message_embeddings = session.run(encoding_tensor, feed_dict = {similarity_input_placeholder:text})
    return message_embeddings