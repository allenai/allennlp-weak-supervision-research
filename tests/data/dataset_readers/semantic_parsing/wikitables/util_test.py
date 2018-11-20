# pylint: disable=invalid-name,no-self-use,protected-access
from allennlp.common import Params
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import WikiTablesDatasetReader
from allennlp.semparse.worlds import WikiTablesWorld
from weak_supervision.data.dataset_readers.semantic_parsing.wikitables import util as wikitables_util

class UtilTest(AllenNlpTestCase):
    def test_parse_example_line(self):
        # pylint: disable=no-self-use,protected-access
        with open("fixtures/data/wikitables/sample_data.examples") as filename:
            lines = filename.readlines()
        example_info = wikitables_util.parse_example_line(lines[0])
        question = 'what was the last year where this team was a part of the usl a-league?'
        assert example_info == {'id': 'nt-0',
                                'question': question,
                                'table_filename': 'tables/590.csv',
                                'target_values': ['2004']}

    def test_parse_labeled_example(line):
        # pylint: disable=no-self-use,protected-access
        with open("fixtures/data/wikitables/sample_data_labeled.examples") as filename:
            lines = filename.readlines()
        example_info = wikitables_util.parse_example_line_with_labels(lines[0])
        question = 'what was the last year where this team was a part of the usl a-league?'
	import pdb; pdb.set_trace();
        assert example_info == {'id': 'nt-0',
                                'question': question,
                                'table_filename': 'tables/590.csv',
                                'target_values': ['2004']}        

