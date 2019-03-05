from common import *

from io import StringIO
from udapi.block.read.conllu import Conllu
from udapi.core.document import Document
from udapi.block.write.textmodetrees import TextModeTrees


def read_data(infile, max_sents=-1):
    sents, answers = [], []
    with open(infile, "r", encoding="utf8") as fin:
        fin.readline()
        for i, line in enumerate(fin):
            line = line.strip()
            if line == "":
                continue
            splitted = line.split("\t", maxsplit=2)
            if len(splitted) >= 2:
                sent, label, other = splitted[0], splitted[1], splitted[2:]
            else:
                sent, label, other = splitted[0], None, None
            sents.append(sent)
            if label == "1":
                other = other[0].split("\t")
                groups = [elem.split() for elem in other[3:]]
                groups = [other[:3]] + list(map(list, zip(*groups)))
                # groups = [tuple(tuple(map(int, x.split(":"))) for x in elem if x != "") for elem in groups]
                groups = [tuple(
                     tuple(map(int, x.split(":"))) if x != "" else (-1, -1)
                    for x in elem) for elem in groups]
                for j, elem in enumerate(groups):
                    while len(groups[j]) < len(groups[0]):
                        groups[j] += ((-1, -1),)
                answers.append(groups)
            else:
                answers.append(None)
            if len(sents) == max_sents:
                break
    return sents, answers


def parse_ud_output(s: str):
    answer, curr_sent_data = [], []
    for line in s.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        elif line == "":
            if len(curr_sent_data) > 0:
                answer.append(curr_sent_data)
            curr_sent_data = []
            continue
        splitted = line.split("\t")
        if splitted[0].isdigit():
            curr_sent_data.append(splitted)
    if len(curr_sent_data) > 0:
        answer.append(curr_sent_data)
    return answer


def read_parse_file(infile, max_sents=-1, parse="split"):
    answer, curr_sent = [], []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                if len(curr_sent) > 0:
                    if parse != "split":
                        curr_sent = "\n".join(curr_sent)
                        if parse is not None:
                            curr_sent = Conllu(filehandle=StringIO(curr_sent)).read_tree()
                            curr_sent.sent_id = str(len(answer) + 1)
                    answer.append(curr_sent)
                curr_sent = []
                if len(answer) == max_sents:
                    break
                continue
            elif parse != "split":
                curr_sent.append(line)
                continue
            elif line.startswith("#"):
                continue
            splitted = line.split("\t")
            if not splitted[0].isdigit():
                continue
            for i in [0, HEAD_COLUMN]:
                splitted[i] = int(splitted[i])
            curr_sent.append(splitted)
        if len(curr_sent) > 0:
            if not parse:
                curr_sent = "\n".join(curr_sent)
            answer.append(curr_sent)
    return answer


def make_string_repr(node):
    return "{}\t{}\t{}\t{}".format(node.ord, node.form, node.deprel, tag_str(node))

def make_phrase_repr(sent, start, end):
    if isinstance(sent, Node):
        sent = [elem.form for elem in sent.descendants]
    if start == end:
        return ""
    elif end == start + 1:
        return "{}-{}".format(sent[start].ord, sent[start].form)
    elif end == start + 2:
        return "{}-{} {}-{}".format(sent[start].ord, sent[start].form, sent[start+1].ord, sent[start+1].form)
    else:
        return "{}-{} .. {}-{}".format(sent[start].ord, sent[start].form, sent[end-1].ord, sent[end-1].form)

def output_results(outfile, answer, gaps, corr_answer, sents,
                   parsed_sents, indexes=None, output_trees=False):
    if indexes is None:
        indexes = list(range(len(corr_answer)))
    with open(outfile, "w", encoding="utf8") as fout:
        for i in indexes:
            writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
            fout.write("{}\t{}\n{}\n".format(i, sents[i], "=" * 40))
            for j, (head, first, second) in enumerate(gaps[i]):
                if j == 0:
                    fout.write("{}\t{}\t{}\n".format(head.ord, head.form, head.deprel) + "-" * 40 + "\n")
                for elem in first:
                    fout.write(make_string_repr(elem) + "\n")
                fout.write("-" * 40 + "\n")
                for elem in second:
                    fout.write(make_string_repr(elem) + "\n")
                fout.write(("-" * 40 if j < len(gaps[i]) - 1 else "=" * 40) + "\n")
            if output_trees:
                writer.before_process_document(Document())
                writer.process_tree(parsed_sents[i])
                writer.after_process_document(Document())
                fout.write("=" * 40 + "\n")
            if len(answer[i]) > 0:
                for elem in answer[i]:
                    fout.write("\t".join([make_phrase_repr(parsed_sents[i], *x) for x in elem]) + "\n")
            if corr_answer[i] != answer[i] and corr_answer[i] is not None:

                fout.write("=" * 15 + " CORRECT " + "=" * 15 + "\n")
                for elem in corr_answer[i]:
                    fout.write("\t".join([make_phrase_repr(parsed_sents[i], *x) for x in elem]) + "\n")
                fout.write("=" * 40 + "\n")
            elif len(answer[i]) > 0:
                fout.write("=" * 40 + "\n")
            fout.write("\n")

def output_matching_results(outfile, answer, gaps, corr_answer, sents,
                            parsed_sents, indexes=None, output_trees=False):
    if indexes is None:
        indexes = list(range(len(corr_answer)))
    with open(outfile, "w", encoding="utf8") as fout:
        for i in indexes:
            writer = TextModeTrees(filehandle=fout, attributes="form,upos,deprel,ord,feats")
            fout.write("{}\t{}\n{}\n".format(i, sents[i], "=" * 40))
            for j, (head, first, second) in enumerate(gaps[i]):
                if j == 0:
                    fout.write("{}\t{}\t{}\n".format(head.ord, head.form, head.deprel) + "-" * 40 + "\n")
                for elem in first:
                    fout.write(make_string_repr(elem) + "\n")
                fout.write("-" * 40 + "\n")
                for elem in second:
                    fout.write(make_string_repr(elem) + "\n")
                fout.write(("-" * 40 if j < len(gaps[i]) - 1 else "=" * 40) + "\n")
            if output_trees:
                writer.before_process_document(Document())
                writer.process_tree(parsed_sents[i])
                writer.after_process_document(Document())
                fout.write("=" * 40 + "\n")
            if len(answer[i]) > 0:
                for elem in answer[i]:
                    fout.write("\t".join([make_phrase_repr(parsed_sents[i], *x) for x in elem]) + "\n")
            if corr_answer[i] != answer[i] and corr_answer[i] is not None:

                fout.write("=" * 15 + " CORRECT " + "=" * 15 + "\n")
                for elem in corr_answer[i]:
                    fout.write("\t".join([make_phrase_repr(parsed_sents[i], *x) for x in elem]) + "\n")
                fout.write("=" * 40 + "\n")
            elif len(answer[i]) > 0:
                fout.write("=" * 40 + "\n")
            fout.write("\n")


def read_gap_file(infile, L):
    answer = [([], []) for _ in range(L)]
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            if line == "":
                continue
            i, verb, gap = map(int, line.split())
            if i >= L:
                continue
            answer[i][0].append(verb)
            answer[i][1].append(gap)
    return answer