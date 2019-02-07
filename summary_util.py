import numpy as np
import tensorflow as tf


def init(logs_path):
    tf.summary.merge_all()
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    return writer


def create_avg_summary(arr, name):
    avg = np.mean(arr)
    return create_summary(avg, name)


def create_summary(value, name):
    summary = tf.Summary()
    summary.value.add(tag=name, simple_value=value)
    return summary


def write_summaries(writer, episode, summaries):
    for summary in summaries:
        writer.add_summary(summary, episode)
