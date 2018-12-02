#! /usr/bin/env python


# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))


from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util 
from allennlp.state_machines import BeamSearch

from weak_supervision.data.dataset_readers import WikiTablesVariableFreeDatasetReader
from weak_supervision.state_machines import GreedyEpsilonBeamSearch

def make_data(input_examples_file: str,
              tables_directory: str,
              archived_model_file: str,
              output_dir: str,
              num_logical_forms: int,
              override_file: str = "",
              lang: str = "mapo",
              beam_search_eps: float = 1.0,
              num_steps: int = -1) -> None:

    if lang == "mapo":
        reader = WikiTablesVariableFreeDatasetReader(tables_directory=tables_directory,
                                                     keep_if_no_logical_forms = True)
    else:
        reader = WikiTablesDatasetReader(tables_directory=tables_directory,
                                     keep_if_no_dpd=True,
                                     output_agendas=False)
    dataset = reader.read(input_examples_file)
    if override_file:
        archive = load_archive(archived_model_file, overrides = open(override_file).read())
    else:
        archive = load_archive(archived_model_file)
    model = archive.model
    model.eval()

    model._beam_search = BeamSearch(beam_size=100) #GreedyEpsilonBeamSearch(beam_size = 100, epsilon=beam_search_eps)

    if num_steps != -1:
        model._max_decoding_steps = num_steps
    
    lines = open(input_examples_file).readlines()

    cnt = 0.0
    for example_line, instance in zip(lines, dataset):
        outputs = model.forward_on_instance(instance)
        parsed_info = util.parse_example_line(example_line)
        example_id = parsed_info["id"]
        total = len(outputs['logical_form'])

        correct_logical_forms = outputs['correct_logical_form'][:num_logical_forms]
        num_found = len(correct_logical_forms)
        print(f"{num_found} / {total}  found for {example_id}")
        curr_cnt = 00
        for lf in correct_logical_forms:
            lf = str(lf)
            if lf.count('filter') >= 2:
                curr_cnt+=1.0
        if curr_cnt>0:
            cnt += 1.0

        if num_found == 0:
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = gzip.open(os.path.join(output_dir, f"{example_id}.gz"), "wb")
        for logical_form in correct_logical_forms:
            logical_form_line = (logical_form + "\n").encode('utf-8')
            output_file.write(logical_form_line)
        output_file.close()

    print("{}/{}".format(cnt, len(lines)))

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("--tables_directory", type=str, help="Tables directory", default = "/u/murtyjay/WikiTableQuestions")
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    argparser.add_argument("--output-dir", type=str, dest="output_dir", help="Output directory",
                           default="mml_output")
    argparser.add_argument("--lang", type=str, dest="lang", help="Language",
                           default="mapo")
    argparser.add_argument("--num-logical-forms", type=int, dest="num_logical_forms",
                           help="Number of logical forms to output", default=20)
    argparser.add_argument("--overrides", type=str, dest="overrides", help="Override config",
                           default="")
    argparser.add_argument("--beam_search", type=float, default=1.0, help="epsilon for greedy decoding")
    argparser.add_argument("--num_steps", type=int, dest="num_steps", help="Number of decoding steps",  default=-1)
    args = argparser.parse_args()
    make_data(args.input, args.tables_directory, args.archived_model, args.output_dir,
              args.num_logical_forms, args.overrides, args.lang, args.beam_search, args.num_steps)
