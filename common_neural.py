import numpy as np

from neural_tagging.extract_tags_from_UD import make_full_UD_tag
from neural_tagging.common import to_one_hot
from neural_tagging.vocabulary import FeatureVocabulary, Vocabulary

from read_write import read_data, read_parse_file
from match_gaps import to_subtree_heads, char_to_word_labels
from common import can_be_verb

from deeppavlov import build_model, configs
from deeppavlov.core.common.params import from_params
from deeppavlov.core.commands.utils import parse_config
from deeppavlov.models.embedders.fasttext_embedder import FasttextEmbedder


def make_verb_indexes(sents):
    answer = []
    for sent in sents:
        curr_answer = [i for i, node in enumerate(sent.descendants) if can_be_verb(node, allow_inf=False)]
        answer.append(curr_answer)
    return answer


def make_gap_data(verb_indexes, phrase_data, input_indexes=None,
                  additional_input_indexes=None, output_indexes=None,
                  use_nongap_verbs=False, split_multiple_answers=False,
                  allow_none_outputs=False):
    if input_indexes is None:
        input_indexes = [0]
    if output_indexes is None:
        output_indexes = [3]
    if additional_input_indexes is None:
        additional_input_indexes = [[] for _ in output_indexes]
    elif len(additional_input_indexes) != len(output_indexes):
        raise ValueError("Wrong length of additional indexes")
    input_indexes = [input_indexes + elem for elem in additional_input_indexes]
    if any(x != 0 for elem in input_indexes for x in elem) and use_nongap_verbs:
        raise ValueError("use_nongap_verbs must be False for any index != 0")
    if any(x >= 3 for elem in input_indexes for x in elem) and not split_multiple_answers:
        raise ValueError("")
    inputs, outputs = [], []
    for r, (curr_verb_indexes, curr_phrase_data) in enumerate(zip(verb_indexes, phrase_data)):
        gap_data_by_verbs = dict()
        if curr_phrase_data is not None:
            curr_verb_data, curr_gap_data = curr_phrase_data[0], curr_phrase_data[1:]
            verb_index = curr_verb_data[0]
            if verb_index not in gap_data_by_verbs:
                gap_data_by_verbs[verb_index] = list(curr_verb_data) + [[], [], []]
            for elem in curr_gap_data:
                for j, x in enumerate(elem):
                    gap_data_by_verbs[verb_index][3+j].append(x)
        if use_nongap_verbs:
            for index in curr_verb_indexes:
                if index not in gap_data_by_verbs:
                    gap_data_by_verbs[index] = [index, None, None, [], [], []]
        gap_data = sorted(gap_data_by_verbs.values())
        if split_multiple_answers:
            gap_data = [elem[:3] + list(second) for elem in gap_data for second in zip(*elem[3:])]
        curr_inputs, curr_outputs = [], []
        for elem in gap_data:
            for curr_input_indexes, curr_output_index in zip(input_indexes, output_indexes):
                curr_input = [elem[i] for i in curr_input_indexes]
                for i, x in enumerate(curr_input):
                    if isinstance(x, list):
                        curr_input[i] = x[0]
                curr_output = elem[curr_output_index]
                if None not in curr_input:
                    curr_output = curr_output if isinstance(curr_output, list) else [curr_output]
                    if (None not in curr_output or allow_none_outputs):
                        curr_inputs.append(curr_input)
                        curr_outputs.append(curr_output)
        if len(curr_inputs) > 0:
            curr_inputs = list(map(list, zip(*curr_inputs)))
        else:
            curr_inputs = [[] for _ in input_indexes[0]]
        # has_none =(None in curr_inputs or None in curr_outputs or
        #            any(isinstance(x, list) and None in x for x in curr_outputs))
        # if has_none:
        #     curr_inputs = [[] for _ in input_indexes[0]]
        #     curr_outputs = []
        inputs.append(curr_inputs)
        outputs.append(curr_outputs)
    return inputs, outputs


def _prepare_data(source, sents, answers=None, from_subtree_labels=False,
                  input_indexes=None, additional_input_indexes=None, output_indexes=None,
              use_nongap_verbs=False, split_multiple_answers=False,
              use_tags=False, tag_vocabulary=None,
              use_deprel=False, deprel_vocabulary=None,
                  allow_none_outputs=False):
    word_sents = [[elem.form for elem in sent.descendants] for sent in sents]
    verb_indexes = make_verb_indexes(sents)
    # data = list(zip(word_sents, verb_indexes))

    answer = {"source": source, "answers": answers, "sents": sents,
              "word_sents": word_sents, "verb_indexes": verb_indexes}
    word_labels, subtree_labels = None, None
    if not from_subtree_labels:
        if answers is not None:
            word_labels = char_to_word_labels(source, word_sents, answers)
            subtree_labels = to_subtree_heads(sents, word_labels)
    else:
        subtree_labels = answers
    gap_input_data, gap_output_data = make_gap_data(
        verb_indexes, subtree_labels, input_indexes=input_indexes,
        additional_input_indexes=additional_input_indexes,
        output_indexes=output_indexes, use_nongap_verbs=use_nongap_verbs,
        split_multiple_answers=split_multiple_answers,
        allow_none_outputs=allow_none_outputs)
    data = list(zip(word_sents, gap_input_data))
    answer.update({"word_labels": word_labels, "subtree_labels": subtree_labels,
                   "gap_output_data": gap_output_data, "data": data})
    if use_tags:
        tag_sents = [[make_full_UD_tag(elem.upos, dict(elem.feats), mode="dict")
                      for elem in sent.descendants] for sent in sents]
        if tag_vocabulary is None:
            tag_vocabulary = FeatureVocabulary().train(tag_sents)
            answer["tag_vocabulary"] = tag_vocabulary
        tag_label_sents = [[tag_vocabulary.to_vector(tag) for tag in sent] for sent in tag_sents]
        answer.update({"tag_sents": tag_sents, "tag_label_sents": tag_label_sents})
    if use_deprel:
        deprel_sents = [[elem.deprel for elem in sent.descendants] for sent in sents]
        if deprel_vocabulary is None:
            deprel_vocabulary = Vocabulary().train(deprel_sents)
            answer["deprel_vocabulary"] = deprel_vocabulary
        deprel_label_sents = [[deprel_vocabulary.toidx(elem) for elem in sent] for sent in deprel_sents]
        answer.update({"deprel_sents": deprel_sents, "deprel_label_sents": deprel_label_sents})
    return answer

def prepare_data(train_file, parse_file, max_sents=-1, **kwargs):
    (source, answers) = read_data(train_file, max_sents=max_sents)
    sents = read_parse_file(parse_file, max_sents=max_sents, parse=False)
    return _prepare_data(source, sents, answers, **kwargs)



FASTTEXT_LOAD_PATH = "/home/alexeysorokin/data/Data/DeepPavlov Embeddings/ft_native_300_ru_wiki_lenta_lower_case.bin"
EMBEDDINGS_DIM = 300

def load_fasttext(load_path=FASTTEXT_LOAD_PATH, dim=EMBEDDINGS_DIM):
    embedder = FasttextEmbedder(load_path=load_path, dim=dim)
    return embedder

def load_elmo(elmo_output_names=("word_emb",)):
    config = parse_config(getattr(configs.elmo_embedder, "elmo_ru-news"))
    elmo_config = config["chainer"]["pipe"][-1]
    elmo_config['elmo_output_names'] = elmo_output_names
    embedder = from_params(elmo_config)
    return embedder

def load_embedder(mode="fasttext", **kwargs):
    func = load_elmo if mode == "elmo" else load_fasttext
    embedder = func(**kwargs)
    return embedder

