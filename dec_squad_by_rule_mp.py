# %%
import json
import traceback
from concurrent.futures._base import as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from copy import deepcopy
import codecs
from functools import partial

from mosestokenizer import MosesDetokenizer
from conllu import parse
from tqdm import tqdm

from rule import Question, AnswerSpan

from allennlp.predictors.predictor import Predictor
import allennlp_models.structured_prediction

import tempfile

MASK = "[MASK]"

def get_formatted_conllu(dp_pred, pos_pred):
    to_write = ""
    for i in range(len(dp_pred['pos'])):
        cur_result = [str(i + 1), dp_pred['words'][i], '_', dp_pred['pos'][i], pos_pred["pos_tags"][i], '_', str(dp_pred['predicted_heads'][i]), dp_pred['predicted_dependencies'][i], '_', '_']
        to_write += '\t'.join(cur_result) + '\n'
    return to_write


def dec_qa(detokenizer, dependency_predictor, pos_predictor, question, answer):
    # turn to conllu
    question_conllu = dependency_predictor.predict(sentence=question)
    answer_conllu = dependency_predictor.predict(sentence=answer)

    question_pos = pos_predictor.predict(sentence=question)
    answer_pos = pos_predictor.predict(sentence=answer)

    with open("cur_file.conllu", "w") as conllu_f:
        question_formatted = get_formatted_conllu(question_conllu, question_pos)
        answer_formatted = get_formatted_conllu(answer_conllu, answer_pos)
        conllu_f.write(question_formatted + '\n')
        conllu_f.write(answer_formatted + '\n')

    with codecs.open('cur_file.conllu', 'r', encoding='utf-8') as f:
        conllu_file = parse(f.read())

    # Creating dict
    ids = range(int(len(conllu_file) / 2))
    examples = {}
    count = 0
    for i, s in enumerate(conllu_file):
        if i % 2 == 0:
            examples[ids[count]] = s
        else:
            examples[str(ids[count]) + '_answer'] = s
            count += 1

    def qa2d(idx):
        q = Question(deepcopy(examples[idx].tokens))
        if not q.isvalid:
            print("Question {} is not valid.".format(idx))
            return ''
        a = AnswerSpan(deepcopy(examples[str(idx) + '_answer'].tokens))
        if not a.isvalid:
            print("Answer span {} is not valid.".format(idx))
            return ''
        q.insert_answer_default(a)
        return detokenizer(q.format_declr())

    def print_sentence(idx):
        return detokenizer([examples[idx].tokens[i]['form'] for i in range(len(examples[idx].tokens))])

    total = int(len(examples.keys()) / 2)

    out = qa2d(0)
    if out == '':
        print("*" * 4)
        print(f"for question '{question}' and answer '{answer}' we got an empty string")
        print("*" * 4)
    return out


def submit_jobs(entries, max_workers):
    pool = ThreadPoolExecutor(max_workers)
    futures2title = {pool.submit(process_entry, entry): entry["title"] for entry in entries}
    return futures2title


def produce_new_entries(futures2title):
    data = []
    for i, future in enumerate(as_completed(futures2title), start=1):
        title = futures2title[future]
        try:
            processed_entry = future.result()
            if processed_entry is not None:
                data.append(processed_entry)
            else:
                raise ValueError
            print(f"future for title '{title}' finished as expected - [{i}/{len(futures2title)}]")
        except Exception as exc:
            print(f"future of '{title}' generated an exception: {exc}")
            print(traceback.format_exc())
    return data


def process_entry(entry):
    detokenizer = MosesDetokenizer('en')
    dependency_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz")
    pos_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    title = entry["title"]
    paragraphs = []
    for i, paragraph in enumerate(entry["paragraphs"]):
        context_text = paragraph["context"]
        qas = []
        for qa in paragraph["qas"]:
            qas_id = qa["id"]
            question_text = qa["question"]
            answer = qa["answers"][0]
            answer_text = answer["text"]
            # TODO some are ignored Houston , Texas vs Houston, Texas
            dec = dec_qa(detokenizer, dependency_predictor, pos_predictor, question_text, answer_text)
            if answer_text not in dec or answer_text not in context_text:
                continue
            cloze_question = dec.replace(answer_text, MASK)
            if i == 3:
                print("=" * 10)
                print(question_text)
                print(answer_text)
                print(cloze_question)
            qa = {"id": qas_id, "question": cloze_question, "is_impossible": False, "answers": qa["answers"]}
            qas.append(qa)
        if len(qas) > 0:
            paragraphs.append({"context": context_text, "qas": qas})
    if len(paragraphs) > 0:
        entry = {"title": title, "paragraphs": paragraphs}
        return entry
    return None


if __name__ == '__main__':
    squad_dev_path = "squad/dev-v1.1.json"
    out_path = "squad/mp-declarative-dev-v1.1.json"
    max_workers = 8



    data = json.load(open(squad_dev_path))["data"]

    futures2title = submit_jobs(data, max_workers)
    processed_data = produce_new_entries(futures2title)
    json.dump({"data": processed_data}, open(out_path, 'w'))


