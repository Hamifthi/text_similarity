import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import os
import time
import timeit

# # function for loading diffrenet module
# def loading_module(path = None, module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'):
#     # Import the Universal Sentence Encoder's TF Hub module
#     g = tf.Graph()
#     with g.as_default():
#         if path == None:
#             embed_object = hub.Module(module_url)
#         else:
#             embed_object = hub.Module(hub.load_module_spec(path))
#     sess = tf.InteractiveSession(graph = g)
#     sess.run([tf.global_variables_initializer(), tf.tables_initializer()])

#     return embed_object, g, sess

# # function for runinng embedding module on text
# def run_embedding(embed_object, graph, sess, text):
#     # Reduce logging output.
#     tf.logging.set_verbosity(tf.logging.ERROR)
#     with graph.as_default():
#         similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
#         encoding_tensor = embed_object(similarity_input_placeholder)
#         message_embeddings = sess.run(encoding_tensor, feed_dict = {similarity_input_placeholder:text})

#     return message_embeddings

# embed_object, graph, sess = loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')
# start_time = time.time()
# run_embedding(embed_object, graph, sess, ['sth to be transformed', 'sth else to be transformed',
#                                           'another thing to be transformed', 'yes it is not different'])
# end_time = time.time()
# print(end_time - start_time)

# function for loading diffrenet module
def loading_module(path = None, module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'):
    # Import the Universal Sentence Encoder's TF Hub module
    if path == None:
        embed_object = hub.Module(module_url)
    else:
        embed_object = hub.Module(hub.load_module_spec(path))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
        saver.save(sess, 'E:/Hamed/Projects/Python/Text Similarity/graph/example')

    return embed_object

# function for runinng embedding module on text
def run_embedding(embed_object, text):
    # Reduce logging output.
    tf.logging.set_verbosity(tf.logging.ERROR)
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('E:/Hamed/Projects/Python/Text Similarity/graph/example.meta')
        saver.restore(sess, 'E:/Hamed/Projects/Python/Text Similarity/graph/example')
        sess.run(tf.tables_initializer())
        similarity_input_placeholder = tf.placeholder(tf.string, shape=(None))
        encoding_tensor = embed_object(similarity_input_placeholder)
        message_embeddings = sess.run(encoding_tensor, feed_dict = {similarity_input_placeholder:text})

    return message_embeddings

embed_object = loading_module('E:/Hamed/Projects/Python/Text Similarity/module/tfhub_modules/1fb57c3ffe1a38479233ee9853ddd7a8ac8a8c47')
start_time = time.time()
run_embedding(embed_object, ['sth'])
end_time = time.time()
print(end_time - start_time)