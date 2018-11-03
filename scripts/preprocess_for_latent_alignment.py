import gzip
import json
import os

from tqdm import tqdm

from allennlp.data.dataset_readers.semantic_parsing.wikitables import util


DATA_PATH = "/u/murtyjay/WikiTableQuestions"
DPD_PATH = "/u/murtyjay/dpd_output/"


def process_file(file_path: str, out_path: str, is_labeled = False):
    examples = []
    gold_examples = []
    with open(file_path, "r") as data_file:
        for line in tqdm(data_file.readlines()):
            line = line.strip("\n")
            if not line:
                continue
            if is_labeled:
                try:
                    parsed_info = util.parse_example_line_with_labels(line)
                except:
                    continue
                sempre_form_gold = " ".join(parsed_info["target_lf"])
                sempre_form_gold = sempre_form_gold.replace("( ", "(").replace(" )", ")")
            else:
                parsed_info = util.parse_example_line(line)
            question = parsed_info["question"]
            dpd_output_filename = os.path.join(DPD_PATH, parsed_info["id"] + '.gz')
            try:
                dpd_file = gzip.open(dpd_output_filename)
                if is_labeled:
                    sempre_forms = [sempre_form_gold] + [dpd_line.strip().decode('utf-8') for dpd_line in dpd_file]
                else:
                    sempre_forms = [dpd_line.strip().decode('utf-8') for dpd_line in dpd_file]
            except FileNotFoundError:
                continue
            if is_labeled:
                gold_examples.append((question, sempre_form_gold))
            examples.append((question, sempre_forms))
    with open(out_path, "w") as out_file:
        json.dump(examples, out_file, indent=2)

    if is_labeled:
        with open(out_path + "gold", "w") as out_file:
            json.dump(gold_examples, out_file, indent=2) 

if __name__ == '__main__':
    process_file(f"{DATA_PATH}/data/random-split-1-train.examples", "train_all_full_dpd.json")
    #process_file(f"{DATA_PATH}/data/eval300.examples", "dev_small.json", is_labeled = True)
