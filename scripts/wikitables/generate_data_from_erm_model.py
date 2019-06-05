#! /usr/bin/env python

# pylint: disable=invalid-name,wrong-import-position,protected-access
import sys
import os
import gzip
import math
import argparse
from multiprocessing import Process
from typing import List, Tuple

from allennlp.data import Instance
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.data.dataset_readers.semantic_parsing.wikitables.util import parse_example_line
from allennlp.models.model import Model
from allennlp.models.archival import load_archive

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))))

from weak_supervision.data.dataset_readers import WikiTablesVariableFreeDatasetReader

def make_data(input_example_lines: List[str],
              model: Model,
              dataset: List[Instance],
              output_dir: str,
              num_logical_forms: int,
              variable_free: bool,
              sort_with_agenda: bool,
              old_logical_forms_path: str) -> None:
    all_num_new_logical_forms = []
    for instance, example_line in zip(dataset, input_example_lines):
        outputs = model.forward_on_instance(instance)
        parsed_info = parse_example_line(example_line)
        example_id = parsed_info["id"]
        logical_forms = outputs["logical_form"]
        world = instance.fields["world"].metadata
        old_file_path = f"{old_logical_forms_path}/{example_id}.gz"
        if old_logical_forms_path is not None and os.path.exists(old_file_path):
            correct_logical_forms = [line.strip() for line in
                                     gzip.open(old_file_path, "rt").readlines()]
        else:
            correct_logical_forms = []
        # This is to measure how many new logical forms we add to the set using the ERM model. Makes
        # sense only if we have an older set of logical forms.
        original_set = set(correct_logical_forms)
        for logical_form in logical_forms:
            if variable_free:
                target_values = instance.fields["target_values"].metadata
                logical_form_is_correct = world.evaluate_logical_form(logical_form,
                                                                      target_values)
            else:
                logical_form_is_correct = model._executor.evaluate_logical_form(logical_form,
                                                                                example_line)
            if logical_form_is_correct:
                correct_logical_forms.append(logical_form)

        if sort_with_agenda:
            agenda = set(world.get_agenda())
            logical_forms_with_coverage: List[Tuple(int, str)] = []
            for logical_form in correct_logical_forms:
                actions = world.get_action_sequence(world.parse_logical_form(logical_form))
                coverage_score = 0
                for action in actions:
                    if action in agenda:
                        coverage_score += 1
                logical_forms_with_coverage.append((coverage_score, logical_form))
            correct_logical_forms = [x[1] for x in sorted(logical_forms_with_coverage, key=lambda x: x[0],
                                                          reverse=True)]
        correct_logical_forms = correct_logical_forms[:num_logical_forms]
        new_set = set(correct_logical_forms)
        num_new_logical_forms = len(new_set.difference(original_set))
        all_num_new_logical_forms.append(num_new_logical_forms)
        num_found = len(correct_logical_forms)
        print(f"{num_found} found for {example_id}. {num_new_logical_forms} are new!")
        if num_found == 0:
            continue
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_file = gzip.open(os.path.join(output_dir, f"{example_id}.gz"), "wt")
        for logical_form in correct_logical_forms:
            print(logical_form, file=output_file)
        output_file.close()
    average_num_new = sum(all_num_new_logical_forms)/len(all_num_new_logical_forms)
    print(f"Average number of new logical forms per question: {average_num_new}")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("input", type=str, help="Input file")
    argparser.add_argument("tables_directory", type=str, help="Tables directory")
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    argparser.add_argument("--output-dir", type=str, dest="output_dir", help="Output directory",
                           default="erm_output")
    argparser.add_argument("--num-logical-forms", type=int, dest="num_logical_forms",
                           help="Number of logical forms to output", default=100)
    argparser.add_argument("--beam-size", type=int, dest="beam_size",
                           help="Decoding beam size (default 100)", default=100)
    argparser.add_argument("--variable-free", dest="variable_free", action="store_true",
                           help="""Will use the variable free dataset reader, and assume the
                           archived model is trained on variable free language if set.""")
    argparser.add_argument("--sort-with-agenda", dest="sort_with_agenda", action="store_true",
                           help="Will output logical forms sorted by agenda")
    argparser.add_argument("--num-splits", dest="num_splits", type=int, default=1,
                           help="""Number of splits to make of the training data. Will use
                           multiprocessing if this value is greater than 1 (default is 1).""")
    argparser.add_argument("--old-logical-forms-path", dest="old_path", type=str,
                           help="""If there you want to update your offline searched forms from a
                           previous iteration instead of completely rewriting them, you can provide
                           the path here""")
    args = argparser.parse_args()
    if args.variable_free:
        new_tables_config = {}
        reader = WikiTablesVariableFreeDatasetReader(tables_directory=args.tables_directory,
                                                     keep_if_no_logical_forms=True,
                                                     output_agendas=True)
    else:
        # Note: Double { for escaping {.
        new_tables_config = f"{{model: {{tables_directory: {args.tables_directory}}}}}"
        reader = WikiTablesDatasetReader(tables_directory=args.tables_directory,
                                         keep_if_no_dpd=True,
                                         output_agendas=True)
    archive = load_archive(args.archived_model, overrides=new_tables_config)
    archived_model = archive.model
    archived_model.training = False
    archived_model._decoder_trainer._max_num_decoded_sequences = 1000
    archived_model._decoder_trainer._beam_size = args.beam_size
    archived_model._decoder_trainer._max_num_finished_states = args.beam_size
    input_dataset = reader.read(args.input)
    input_lines = []
    with open(args.input) as input_file:
        input_lines = input_file.readlines()

    if args.num_splits == 1:
        make_data(input_lines,
                  archived_model,
                  input_dataset,
                  args.output_dir,
                  args.num_logical_forms,
                  args.variable_free,
                  args.sort_with_agenda,
                  args.old_path)
    else:
        chunk_size = math.ceil(len(input_lines)/args.num_splits)
        start_index = 0
        dataset_list = list(input_dataset)
        for i in range(args.num_splits):
            if i == args.num_splits - 1:
                chunk_lines = input_lines[start_index:]
                chunk_data = dataset_list[start_index:]
            else:
                chunk_lines = input_lines[start_index:start_index + chunk_size]
                chunk_data = dataset_list[start_index:start_index + chunk_size]
            start_index += chunk_size
            process = Process(target=make_data, args=(chunk_lines,
                                                      archived_model,
                                                      chunk_data,
                                                      args.output_dir,
                                                      args.num_logical_forms,
                                                      args.variable_free,
                                                      args.sort_with_agenda,
                                                      args.old_path))
            print(f"Starting process {i}", file=sys.stderr)
            process.start()
