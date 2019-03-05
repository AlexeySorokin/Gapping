import sys

from deeppavlov import build_model, configs
from deeppavlov.models.morpho_tagger.common import call_model

from ufal_udpipe import Model as udModel, Pipeline

from read_write import read_data, parse_ud_output

ud_model_path = "models/russian-syntagrus-ud-2.3-181115.udpipe"
train_path = "data/test.csv"
outfile = "results/test.out"
tokenize, parse = False, True
tokenized_outfile = None


HYPHENS = "-—–"
QUOTES = "«“”„»``''"

def cannot_be_before_hyphen(x):
    return not (x.isalpha() or x.isdigit() or x in HYPHENS)       

def fix_quotes(x):
    answer = []
    lines = x.split("\n")
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        splitted = line.split("\t")
        if splitted[1] in QUOTES:
            splitted[1] = splitted[2] = '"'
            splitted[3] = "PUNCT"
            splitted[5] = "_"
        answer.append("\t".join(splitted))
    answer = "\n".join(answer)
    return answer
    
def sanitize(sent):
    sent = "".join(a if a not in QUOTES else '"' for a in sent)
    answer = ""
    indexes = [0] + [i for i, a in enumerate(sent) if a in HYPHENS] + [len(sent)]
    start = 0
    for i, hyphen_index in enumerate(indexes[1:-1], 1):
        answer += sent[start:hyphen_index]
        if hyphen_index > 0 and hyphen_index < len(sent) - 1 and cannot_be_before_hyphen(sent[hyphen_index+1]) and sent[hyphen_index-1].isalpha():
            answer += " " + sent[hyphen_index]
            if sent[hyphen_index+1] != " ":
                answer += " "
        elif hyphen_index > 0 and hyphen_index < len(sent) - 1 and cannot_be_before_hyphen(sent[hyphen_index-1]) and sent[hyphen_index+1].isalpha():
            if sent[hyphen_index-1] != " ":
                answer += " "
            answer += sent[hyphen_index] + " "
        else:
            answer += sent[hyphen_index]
        start = hyphen_index + 1
    answer += sent[start:]
    return answer

if __name__ == "__main__":
    model = build_model(configs.morpho_tagger.UD2_0.morpho_ru_syntagrus_pymorphy_lemmatize, download=True)
    ud_model = udModel.load(ud_model_path)
    sents, answers = read_data(train_path)
    symbols = sorted(set(a for sent in sents for a in sent))
    # with open("dump/symbols.out", "w", encoding="utf8") as fout:
        # fout.write("\n".join(symbols))
        # sys.exit()
    sents = [sanitize(sent) for sent in sents]
    # print("Tagging...")
    # tokenized_data = tokenizer.process("\n\n".join(sents))
    # data = parse_ud_output(tokenized_data)
    # source = [[elem[1] for elem in sent] for sent in data]
    # tagged_data = call_model(model, sents, batch_size=64)
    if tokenize:
        tokenized_data, for_tagging = "", []
        tokenizer = Pipeline(ud_model, "tokenize", Pipeline.NONE, Pipeline.NONE, "conllu")
        for start in range(0, len(sents), 40):
            if start % 400 == 0:
                print("{} sents processed".format(start))
            end = min(start + 40, len(sents))
            curr_output = tokenizer.process("\n\n".join(sents[start:end]))
            tokenized_data += curr_output + "\n"
            curr_output = parse_ud_output(curr_output)
            for_tagging.extend([[elem[1] for elem in sent] for sent in curr_output])
        if tokenized_outfile is not None:
            with open(tokenized_outfile, "w", encoding="utf8") as fout:
                fout.write(tokenized_data)
    else:
        for_tagging = sents
    if parse:
        print(len(for_tagging))
        print("Tagging...")
        tagged_data = call_model(model, for_tagging, batch_size=64)
        tagged_data = [fix_quotes(elem) for elem in tagged_data]
        print("Tagging completed...")
        parser = Pipeline(ud_model, "conllu", Pipeline.NONE, Pipeline.DEFAULT, "conllu")
        with open(outfile, "w", encoding="utf8") as fout:
            for start in range(0, len(tagged_data), 16):
                if start % 400 == 0:
                    print("{} sents processed".format(start))
                end = min(start + 16, len(tagged_data))
                parsed_data = parser.process("\n\n".join(tagged_data[start:end]))
                fout.write(parsed_data)
