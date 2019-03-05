import os
from collections import defaultdict

from common import *
from common import can_be_verb
from read_write import read_data, read_parse_file
from gap_types import find_segment_root
from fix_tree import fix_tree, find_possible_right_subtree_heads
from match_gaps import to_subtree_heads, char_to_word_labels

from udapi.core.node import Node
from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from common_neural import load_embedder, prepare_data
from network import DataGenerator, build, gapping_recall, gapping_support, gapping_total
from network import MyProgbarLogger, make_callback_metrics
from network import load_processor_from_json, GapProcessor, score_gapping


USE_TAGS, USE_DEPREL = False, True
# INPUT_INDEXES, OUTPUT_INDEXES = 0, 3
# ADDITIONAL_INPUT_INDEXES = None
# USE_NONGAP_VERBS = True
# SPLIT = False

INPUT_INDEXES, OUTPUT_INDEXES = 0, [1, 2]
ADDITIONAL_INPUT_INDEXES = [[4], [5]]
USE_NONGAP_VERBS = False
SPLIT = True
EPOCHS = 5

train_file, train_parse_file = "data/train.csv", "results/example_1.out"
dev_file, dev_parse_file = "data/dev.csv", "results/example_dev.out"

LSTM_UNITS, HEADS = 128, 2
POSITIVE_WEIGHT, USE_ATTENTION = 1.0, False
rnns_number = 2
DENSE_UNITS = 256
build_params = {"lstm_units": LSTM_UNITS, "heads": HEADS, "positive_weight": POSITIVE_WEIGHT,
                "use_attention": USE_ATTENTION, "rnns_number": rnns_number,
                "dense_units": DENSE_UNITS, "position_indexes_to_combine": [1]}

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
kbt.set_session(tf.Session(config=config))


if __name__ == "__main__":
    gap_processor = GapProcessor(use_tags=USE_TAGS, use_deprel=USE_DEPREL, input_indexes=INPUT_INDEXES,
                                 additional_input_indexes=ADDITIONAL_INPUT_INDEXES,
                                 output_indexes=OUTPUT_INDEXES, build_params=build_params,
                                 use_nongap_verbs=USE_NONGAP_VERBS, split_multiple_answers=SPLIT,
                                 epochs=EPOCHS)
    train_data = gap_processor.process_data(train_file, train_parse_file, to_train=True)
    dev_data = gap_processor.process_data(dev_file, dev_parse_file)
    gap_processor.build()
    model_file = "models/remnant_location_128_deprel.json"
    weights_file = "models/remnant_location_128_deprel.hdf5"
    gap_processor.to_json(model_file, weights_file)
    gap_processor.train(train_data, dev_data)
    gap_processor.to_json(model_file, weights_file, save_weights=True)
    # gap_processor = load_processor_from_json(model_file)
    # answer = gap_processor.predict(dev_data)
    # prec, rec, f1 = score_gapping(dev_data["gap_output_data"], answer, verbose=1)