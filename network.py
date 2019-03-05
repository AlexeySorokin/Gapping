import os
import inspect
import numpy as np
import ujson as json

import tensorflow as tf
import keras
import keras.layers as kl
import keras.backend as kb
import keras.optimizers as ko
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ProgbarLogger

from neural_tagging.common import to_one_hot
from neural_tagging.vocabulary import Vocabulary, vocabulary_from_json
from common_neural import _prepare_data, prepare_data, load_embedder


class DataGenerator:

    def __init__(self, embedder, data, labels=None, position_inputs_number=1,
                 additional_data=None, additional_shapes=None,
                 buckets_number=10, batch_size=8, epochs=None, yield_indexes=False,
                 shuffle=True, seed=184):
        self.data = data
        self.labels = labels
        self.position_inputs_number = position_inputs_number
        self.embedder = embedder
        self.additional_data = additional_data or []
        self.additional_shapes = additional_shapes or []
        self.buckets_number = buckets_number
        self.batch_size = batch_size
        self.epochs = epochs
        self.yield_indexes = yield_indexes
        self.shuffle = shuffle
        self.seed = seed
        self.initialize()

    @property
    def steps_per_epoch(self):
        return len(self.batch_starts)

    def initialize(self):
        lengths = [len(x) for x in self.data]
        indexes = np.argsort(lengths)
        self.buckets = []
        self.batch_starts = []
        start = 0
        for i in range(self.buckets_number):
            end = (len(self.data) * (i + 1)) // self.buckets_number
            bucket_indexes = indexes[start:end]
            start = end
            self.buckets.append(bucket_indexes)
            self.batch_starts.extend(((i, j) for j in range(0, len(bucket_indexes), self.batch_size)))
        self._remove_empty_batches()
        self.step = 0
        self.epoch = 0
        if self.shuffle:
            np.random.seed(self.seed)
        return

    def _remove_empty_batches(self):
        answer = []
        for bucket_index, start in self.batch_starts:
            end = min(start + self.batch_size, len(self.buckets[bucket_index]))
            batch_indexes = list(self.buckets[bucket_index][start:end])
            B = sum(len(self.data[index][1][0]) for index in batch_indexes)
            if B > 0:
                answer.append((bucket_index, start))
        self.batch_starts = answer

    def __iter__(self):
        return self

    def make_batch(self, indexes, L):
        B = sum(len(self.data[index][1][0]) for index in indexes)
        answer = np.zeros(shape=(B, L, self.embedder.dim), dtype=np.float)
        positions = [np.zeros(shape=(B,), dtype=int) for _ in range(self.position_inputs_number)]
        embeddings = self.embedder([self.data[index][0] for index in indexes])
        start = 0
        for i, index in enumerate(indexes):
            end = start + len(self.data[index][1][0])
            answer[start:end, :len(self.data[index][0])] = embeddings[i]
            for r in range(self.position_inputs_number):
                positions[r][start:end] = self.data[index][1][r]
            start = end
        return [answer] + positions

    def make_labels_batch(self, indexes, L):
        B = sum(len(self.labels[index]) for index in indexes)
        answer = np.zeros(shape=(B, L), dtype=int)
        start = 0
        for i, index in enumerate(indexes):
            end = start + len(self.labels[index])
            for j, elem in enumerate(self.labels[index]):
                elem = [x for x in elem if x is not None]
                try:
                    answer[start + j, elem] = 1
                except:
                    print(elem)
                    raise ValueError("")
            start = end
        return answer

    def _make_additional_batch(self, data, indexes, L, shape):
        shape = list(shape)
        B = sum(len(self.data[index][1][0]) for index in indexes)
        shape[0] = B
        if len(shape) > 1 and shape[1] is None:
            shape[1] = L
        shape_diff = len(shape) - len(np.shape(data[0]))
        if shape_diff not in [1, 2]:
            raise ValueError("Wrong dimension")
        answer_shape = shape if shape_diff == 1 else shape[:-1]
        answer = np.zeros(shape=answer_shape, dtype=int)
        start = 0
        for i, index in enumerate(indexes):
            end = start + len(self.data[index][1][0])
            curr_data = np.array(data[index])
            if curr_data.ndim > 0:
                answer[start:end, :curr_data.shape[0]] = curr_data
            else:
                answer[start:end] = curr_data
        if shape_diff == 2:
            answer = to_one_hot(answer, shape[-1])
        return answer

    def __next__(self):
        if self.step == 0:
            if self.epoch == self.epochs:
                raise StopIteration()
            if self.shuffle:
                np.random.shuffle(self.batch_starts)
        batch_index, start = self.batch_starts[self.step]
        end = min(start + self.batch_size, len(self.buckets[batch_index]))
        batch_indexes = list(self.buckets[batch_index][start:end])
        max_length = max(len(self.data[i][0]) for i in batch_indexes)
        batch = self.make_batch(batch_indexes, max_length)
        additional_batch = [self._make_additional_batch(data, batch_indexes, max_length, shape)
                            for data, shape in zip(self.additional_data, self.additional_shapes)]
        batch += additional_batch
        if self.labels is not None:
            labels = self.make_labels_batch(batch_indexes, max_length)
        #             additional_labels = [self._make_additional_batch(data, batch_indexes, max_length, shape)
        #                                  for data, shape in zip(self.additional_labels, self.additional_labels_shapes)]
        self.step += 1
        if self.step == self.steps_per_epoch:
            self.step = 0
            self.epoch += 1
        to_yield = [batch]
        if self.labels is not None:
            to_yield.append(labels)
        if self.yield_indexes:
            batch_indexes = [(index, r, positions) for index in batch_indexes
                             for r, positions in enumerate(zip(*self.data[index][1]))]
            to_yield.append(batch_indexes)
        return tuple(to_yield)


def gapping_loss(y_true, y_pred):
    first_log = -kb.log(kb.clip(y_pred, kb.epsilon(), 1.0-kb.epsilon()))
    second_log = -kb.log(kb.clip(1.0-y_pred, kb.epsilon(), 1.0-kb.epsilon()))
    loss = kb.max(y_true * first_log, axis=-1) + kb.max((1.0 - y_true) * second_log, axis=-1)
    # loss = kb.max(loss, axis=-1)
    return loss


class GappingLoss:

    def __init__(self, positive_weight=1.0):
        self.positive_weight = positive_weight
        self.__name__ = "gapping_loss"

    def __call__(self, y_true, y_pred):
        first_log = -kb.log(kb.clip(y_pred, kb.epsilon(), 1.0 - kb.epsilon()))
        second_log = -kb.log(kb.clip(1.0 - y_pred, kb.epsilon(), 1.0 - kb.epsilon()))
        loss = self.positive_weight * kb.max(y_true * first_log, axis=-1) + kb.max((1.0 - y_true) * second_log, axis=-1)
        # loss = kb.max(loss, axis=-1)
        return loss

def gapping_metric(y_true, y_pred):
    y_pred = kb.cast(y_pred > 0.5, dtype=np.float32)
    joint_sum, total_sum = kb.sum(y_true * y_pred, axis=-1), kb.sum(kb.maximum(y_true, y_pred), axis=-1)
    answer = kb.switch(kb.equal(total_sum, 0), kb.ones_like(total_sum), joint_sum / total_sum)
    return answer

def gapping_recall(y_true, y_pred):
    return kb.sum(y_true * kb.cast(y_pred > 0.5, dtype=np.float32))

def gapping_support(y_true, y_pred):
    return kb.sum(y_true)

def gapping_total(y_true, y_pred):
    return kb.sum(kb.maximum(y_true, kb.cast(y_pred > 0.5, dtype=np.float32)))

# metrics

def hit_recall(logs):
    return logs["gapping_recall"] / max(logs["gapping_support"], 1)

def hit_accuracy(logs):
    return logs["gapping_recall"] / max(logs["gapping_total"], 1)

# callbacks

class MyProgbarLogger(ProgbarLogger):

    def __init__(self, verbose=1, metrics_to_add=None, metrics_to_remove=None):
        self.verbose = verbose
        self.metrics_to_add = metrics_to_add or []
        self.metrics_to_remove = metrics_to_remove or []
        super().__init__(count_mode="steps")

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        if self.use_steps:
            self.seen += 1
        else:
            self.seen += batch_size

        for k in self.params['metrics']:
            if k in logs and k not in self.metrics_to_remove:
                self.log_values.append((k, logs[k]))
        for output_name, input_names, func in self.metrics_to_add:
            func_inputs = [logs[name] for name in input_names]
            value = func(*func_inputs)
            self.log_values.append((output_name, value))

        # Skip progbar update for the last batch;
        # will be handled by on_epoch_end.
        if self.verbose and self.seen < self.target:
            self.progbar.update(self.seen, self.log_values)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name in self.params['metrics']:
            if name in logs:
                log_name = name
                value = logs[name]
                if name[:4] == "val_":
                    name = name[4:]
                if name not in self.metrics_to_remove:
                    self.log_values.append((log_name, value))
        for output_name, input_names, func in self.metrics_to_add:
            func_inputs = [logs[name] for name in input_names]
            value = func(*func_inputs)
            self.log_values.append((output_name, value))
            if self.validation_data is not None:
                func_inputs = [logs["val_" + name] for name in input_names]
                value = func(*func_inputs)
                self.log_values.append(("val_" + output_name, value))

        if self.verbose:
            self.progbar.update(self.seen, self.log_values)

def make_callback_metrics(metrics="default"):
    if metrics == "default":
        metrics = ["hit_recall", "hit_accuracy"]
    answer = []
    if "hit_recall" in metrics:
        answer.append(("hit_recall", ["gapping_recall", "gapping_support"], lambda x, y: x / max(y, 1e-7)))
    if "hit_accuracy" in metrics:
        answer.append(("hit_accuracy", ["gapping_recall", "gapping_total"], lambda x, y: x / max(y, 1e-7)))
    return answer

# network

def positions_func(a, indexes):
    first_indexes = kb.arange(kb.shape(a)[0])
    indexes = tf.stack([first_indexes, indexes[:,0]], axis=-1)
    return tf.gather_nd(a, indexes)




class StateAttention(kl.Layer):

    def __init__(self, heads=1, **kwargs):
        self.heads = heads
        super(StateAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 3
        super(StateAttention, self).build(input_shape)

    def call(self, inputs, **kwargs):
        first, second = inputs
        products = kb.expand_dims(first, axis=-2) * second  # B * L * D
        # output_shape = tuple(kb.shape(second)[:-1]) + (heads, second.shape[-1] // heads)
        second_shape = kb.shape(second)
        output_shape = [second_shape[i] for i in range(kb.ndim(second) - 1)] + [self.heads, second.shape[-1] // self.heads]
        products = kb.reshape(products, shape=output_shape)
        scores = kb.sum(products, axis=-1)
        return scores

    def compute_output_shape(self, input_shape):
        return input_shape[1][:-1] + (self.heads,)


def build(embeddings_dim, lstm_units, rnns_number=2,
          position_inputs_number=1, position_input_states=None,
          position_indexes_to_combine=None,
          use_tags=False, tags_dim=None, use_deprel=False, deprel_dim=None,
          use_attention=True, dense_units=None,
          heads=1, metrics=None,
          positive_weight=1.0):
    metrics = metrics or []
    if position_input_states is None:
        if rnns_number > 1:
            position_input_states = [1] * position_inputs_number
        else:
            position_input_states = list(range(1, 1+position_inputs_number))
    if position_indexes_to_combine is None:
        position_indexes_to_combine = [0]
    embeddings = kl.Input(shape=(None, embeddings_dim), dtype=np.float32)
    inputs = [embeddings]
    for r in range(position_inputs_number):
        positions = kl.Input(shape=(1,), dtype=np.int32, name="positions_{}".format(r+1))
        inputs.append(positions)
    if use_tags:
        tags = kl.Input(shape=(None, tags_dim), dtype=np.float32)
        inputs.append(tags)
        embeddings = kl.Concatenate()([embeddings, tags])
    if use_deprel:
        deprels = kl.Input(shape=(None, deprel_dim), dtype=np.float32)
        inputs.append(deprels)
        embeddings = kl.Concatenate()([embeddings, deprels])
    if rnns_number == 1:
        rnn_states = kl.Bidirectional(kl.GRU(lstm_units, return_sequences=True))(embeddings)
        states = [kl.Dense(dense_units)(rnn_states) for _ in range(1+position_inputs_number)]
        state_units = dense_units
    else:
        states = [kl.Bidirectional(kl.GRU(lstm_units, return_sequences=True,
                                          name = "rnn_{}".format(i+1)))(embeddings)
                  for i in range(rnns_number)]
        state_units = 2 * lstm_units
    position_states = []
    for r, (curr_state_index, curr_positions) in enumerate(
            zip(position_input_states, inputs[1:1+position_inputs_number])):
        curr_states = states[curr_state_index]
        curr_position_states = kl.Lambda(
            positions_func, arguments={"indexes": curr_positions},
            output_shape=(state_units,),
            name="position_states_{}".format(r))(curr_states)
        position_states.append(curr_position_states)
    output_states = states[0]
    if use_attention:
        mean_position_states = kl.Average()(position_states)
        scores = StateAttention(heads=heads)([mean_position_states, output_states])
        probs = kl.Dense(1, activation="sigmoid")(scores)
    else:
        position_states = [kl.Lambda(kb.expand_dims, arguments={"axis": -2},
                                     output_shape=(lambda x: x[:-1] + (1, x[-1])))(elem)
                           for elem in position_states]
        tiling_shape = [1, kb.shape(embeddings)[-2], 1]
        position_states = [kl.Lambda(kb.tile, arguments={"n": tiling_shape},
                                     output_shape=(lambda x: x[:-2] + (None, x[-1])))(elem)
                           for elem in position_states]
        to_concatenate = [output_states]
        for r, state in enumerate(position_states):
            to_concatenate.append(state)
            if r in position_indexes_to_combine:
                to_concatenate.extend([kl.Subtract()([state, output_states]),
                                       kl.Multiply()([state, output_states])])
        states = kl.Concatenate()(to_concatenate)
        probs = kl.Dense(1, activation="sigmoid")(states)
    probs = kl.Lambda(lambda x:x[...,0], output_shape=(lambda x:x[:-1]))(probs)
    model = Model(inputs, [probs])
    print("positive weight:", positive_weight)
    model.compile(optimizer=ko.Adam(clipnorm=5.0), loss=GappingLoss(positive_weight),
                  metrics=[gapping_metric] + metrics)
    print(model.summary())
    return model


def load_processor_from_json(infile):
    with open(infile, "r", encoding="utf8") as fin:
        info = json.load(fin)
    other_info = dict()
    for key in ["tag_vocabulary", "deprel_vocabulary", "weights_file"]:
        other_info[key] = info.pop(key)
    processor = GapProcessor(**info)
    for key in ["tag_vocabulary", "deprel_vocabulary"]:
        if other_info[key] is not None:
            vocab = vocabulary_from_json(other_info[key], use_features=(key == "tag_vocabulary"))
            setattr(processor, key, vocab)
    processor.build()
    weights_file = os.path.join(os.path.dirname(infile), other_info["weights_file"])
    processor.model_.load_weights(weights_file)
    return processor


class GapProcessor:

    def __init__(self, embedder_mode="elmo", use_tags=False, use_deprel=False,
                 input_indexes=0, additional_input_indexes=None,
                 output_indexes=3, use_nongap_verbs=True, split_multiple_answers=False,
                 embedder_params=None, build_params=None,
                 epochs=1, max_patience=1, score_best_only=False):
        self.embedder_mode = embedder_mode
        self.use_tags = use_tags
        self.use_deprel = use_deprel
        self.input_indexes = input_indexes
        self.additional_input_indexes = additional_input_indexes
        self.output_indexes = output_indexes
        self.score_best_only = any(x in [1,2] for x in output_indexes)
        self.use_nongap_verbs = use_nongap_verbs
        self.split_multiple_answers = split_multiple_answers
        self.tag_vocabulary = None
        self.deprel_vocabulary = None
        self.embedder_params = embedder_params or {"elmo_output_names": ["lstm_outputs1"]}
        self.build_params = build_params or dict()
        self.epochs = epochs
        self.max_patience = max_patience
        self.score_best_only = score_best_only
        self._initialize()
        self.embedder = load_embedder(mode=self.embedder_mode, **self.embedder_params)

    def _initialize(self):
        if isinstance(self.input_indexes, int):
            self.input_indexes = [self.input_indexes]
        if isinstance(self.output_indexes, int):
            self.output_indexes = [self.output_indexes]

    def to_json(self, outfile, weights_file, save_weights=False):
        info = dict()
        for attr in ["embedder_mode", "use_tags", "use_deprel", "input_indexes",
                     "output_indexes", "additional_input_indexes",
                     "embedder_params", "build_params", "epochs", "max_patience"]:
            info[attr] = getattr(self, attr)
        for attr in ["tag_vocabulary", "deprel_vocabulary"]:
            val = getattr(self, attr)
            if val is not None:
                val = val.jsonize()
            info[attr] = val
        weights_file = os.path.relpath(weights_file, os.path.dirname(outfile))
        print(weights_file)
        info["weights_file"] = weights_file
        weights_file = os.path.join(os.path.dirname(outfile), weights_file)
        print(weights_file)
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)
        if save_weights:
            self.model_.save_weights(weights_file)

    def process_data(self, *args, to_train=False, max_sents=-1,
                     from_files=True, from_subtree_labels=False,
                     allow_none_outputs=False):
        params_keys = ["use_tags", "tag_vocabulary", "use_deprel", "deprel_vocabulary",
                       "input_indexes", "additional_input_indexes", "output_indexes",
                       "use_nongap_verbs", "split_multiple_answers"]
        params = {key: getattr(self, key) for key in params_keys}
        if from_files:
            train_file, parse_file = args
            data = prepare_data(train_file, parse_file, max_sents=max_sents,
                                allow_none_outputs=allow_none_outputs, **params)
        else:
            source_sents, answers, sents = args
            data = _prepare_data(source_sents, sents, answers,
                                 from_subtree_labels=from_subtree_labels,
                                 allow_none_outputs=allow_none_outputs,
                                 **params)
        if to_train:
            self.tag_vocabulary = data.get("tag_vocabulary")
            self.deprel_vocabulary = data.get("deprel_vocabulary")
        return data

    def _make_additional_shapes(self):
        self.additional_shapes = []
        if self.use_tags:
            self.additional_shapes.append((None, None, self.tag_vocabulary.symbol_vector_size_))
        if self.use_deprel:
            self.additional_shapes.append((None, None, self.deprel_vocabulary.symbols_number_))
        return self

    @property
    def positions_inputs_number(self):
        input_length = len(self.input_indexes)
        if self.additional_input_indexes is not None:
            input_length += len(self.additional_input_indexes[0])
        return input_length

    def build(self):
        build_params = self.build_params.copy()
        build_params["position_inputs_number"] = self.positions_inputs_number
        build_params["metrics"] = [gapping_recall, gapping_support, gapping_total]
        build_params.update({"use_tags": self.use_tags, "use_deprel": self.use_deprel})
        if self.use_tags:
            build_params["tags_dim"] = self.tag_vocabulary.symbol_vector_size_
        if self.use_deprel:
            build_params["deprel_dim"] = self.deprel_vocabulary.symbols_number_
        self.model_ = build(self.embedder.dim, **build_params)
        self._make_additional_shapes()
        return self

    def _make_gen(self, data, output_labels=True, epochs=None, yield_indexes=False, shuffle=False):
        gap_data = data["gap_output_data"] if output_labels else None
        additional_data = []
        if self.use_tags:
            additional_data.append(data["tag_label_sents"])
        if self.use_deprel:
            additional_data.append(data["deprel_label_sents"])
        generator = DataGenerator(self.embedder, data["data"], gap_data,
                                  position_inputs_number=self.positions_inputs_number,
                                  additional_data=additional_data,
                                  additional_shapes=self.additional_shapes,
                                  epochs=epochs, yield_indexes=yield_indexes, shuffle=shuffle)
        return generator

    def _make_callbacks(self):
        additional_metrics = [gapping_recall, gapping_support, gapping_total]
        metrics_to_remove = [func.__name__ for func in additional_metrics]
        metrics_to_add = make_callback_metrics()
        callbacks = [MyProgbarLogger(metrics_to_add=metrics_to_add, metrics_to_remove=metrics_to_remove)]
        return callbacks

    def train(self, train_data, dev_data):
        train_gen = self._make_gen(train_data, shuffle=True)
        best_f1, patience, best_weights = 0.0, 0, None
        callbacks = self._make_callbacks()
        for i in range(self.epochs):
            self.model_.fit_generator(train_gen, train_gen.steps_per_epoch,
                                      epochs=1, callbacks=callbacks, verbose=0)
            answer = self.predict(dev_data)
            prec, rec, f1 = score_gapping(dev_data["gap_output_data"], answer, verbose=1)
            if f1 > best_f1:
                best_f1, patience = f1, 0
                best_weights = self.model_.get_weights()
            else:
                patience += 1
                if patience > self.max_patience:
                    break
        self.model_.set_weights(best_weights)
        return self

    def predict(self, data):
        test_gen = self._make_gen(data, output_labels=False, yield_indexes=True, epochs=1)
        pred_gap_positions = [[[] for _ in elem[1][0]] for elem in data["data"]]
        for r, (x, indexes) in enumerate(test_gen):
            curr_answer = self.model_.predict(x)
            for (i, j, s), elem in zip(indexes, curr_answer):
                predicted = list(np.where(elem > 0.5)[0])
                if len(predicted) > 0 and self.score_best_only:
                    predicted = [np.argmax(elem)]
                pred_gap_positions[i][j] = predicted
        return pred_gap_positions

def score_gapping(corr_answer, answer, verbose=1):
    TP, FP, FN = 0, 0, 0
    for elem in zip(corr_answer, answer):
        for corr, pred in zip(*elem):
            common = [x for x in pred if x in corr]
            TP += len(common)
            FP += len(pred) - len(common)
            FN += len(corr) - len(common)
    prec, rec, f1 = TP / (TP + FP), TP / (TP + FN), TP / (TP + 0.5 * (FN + FP))
    if verbose:
        print(TP, FP, FN)
        print("Precision {:.2f}, Recall {:.2f}, F1 {:.2f}".format(100 * prec, 100 * rec, 100 * f1))
    return prec, rec, f1

def to_json(outfile, weights_file, **kwargs):
    kwargs["weights_file"] = weights_file
    new_metrics = []
    for func in kwargs["metrics"]:
        new_metrics.append(func.__name__)
    kwargs["metrics"] = new_metrics
    with open(outfile, "w", encoding="utf8") as fout:
        json.dump(kwargs, fout)

def from_json(infile, verbose=0):
    with open(infile, "r", encoding="utf8") as fin:
        params = json.load(fin)
    params["metrics"] = [eval(key) for key in params["metrics"]]
    weights_file = params.pop("weights_file")
    model = build(**params, verbose=verbose)
    model.load_weights(weights_file)
    return model