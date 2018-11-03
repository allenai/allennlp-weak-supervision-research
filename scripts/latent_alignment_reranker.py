import argparse
import gzip
import os

from allennlp.common import Params
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.dataset_readers import LatentAlignmentDatasetReader
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.models.archival import load_archive
from allennlp.data.dataset_readers.semantic_parsing.wikitables import util 

def rerank_dpd(model_file, input_examples_file, params_file, tables_directory, dpd_directory, output_directory):
    model = load_archive(model_file).model
    model.eval()


    params = Params.from_file(params_file)
    latent_alignment_reader = DatasetReader.from_params(params.pop('dataset_reader'))

    with open(input_examples_file) as input_file:
        input_lines = input_file.readlines()

    for line in input_lines:
        parsed_info = util.parse_example_line(line)
        example_id = parsed_info["id"]
        dpd_output_filename = os.path.join(dpd_directory, parsed_info["id"] + '.gz')
        try:
            dpd_file = gzip.open(dpd_output_filename)
            sempre_forms = [dpd.strip().decode('utf-8') for dpd in dpd_file]
            question = parsed_info['question']
            instance = latent_alignment_reader.text_to_instance(question, sempre_forms)
            output = model.forward_on_instance(instance)
            similarities = output['all_similarities'] 
            top_lfs = [lf for lf, score in sorted(zip(sempre_forms, similarities), key = lambda x: x[1], reverse = True)[:10]]
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            output_file = gzip.open(os.path.join(output_directory, f"{example_id}.gz"), "wb")
            for logical_form in top_lfs:
                logical_form_line = (logical_form + "\n").encode('utf-8')
                output_file.write(logical_form_line)
            output_file.close()

        except FileNotFoundError:
            continue
            


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("archived_model", type=str, help="Archived model.tar.gz")
    argparser.add_argument("input_examples_file", type=str, help="Input Examples file")
    argparser.add_argument("table_dir", type=str, help="Table directory")
    argparser.add_argument("dpd_dir", type=str, help="DPD directory")
    argparser.add_argument("params_file", type=str, help="able directory")
    argparser.add_argument("--output-dir", type=str, dest="out_dir", help="Output directory",
                           default="latent_alignment")

    args = argparser.parse_args()
    rerank_dpd(args.archived_model, args.input_examples_file, args.params_file, args.table_dir, args.dpd_dir, args.out_dir) 
