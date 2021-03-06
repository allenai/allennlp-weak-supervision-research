# pylint: disable=no-self-use
# pylint: disable=invalid-name

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter

from weak_supervision.semparse.contexts import TableQuestionContext
from weak_supervision.semparse.contexts.table_question_context import Date


class TestTableQuestionContext(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = WordTokenizer(SpacyWordSplitter(pos_tags=True))

    def test_table_data(self):
        question = "what was the attendance when usl a league played?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/wikitables/sample_table.tagged'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        assert table_question_context.table_data == [{'date_column:year': Date(2001, -1, -1),
                                                      'string_column:year': '2001',
                                                      'number_column:year': 2001.0,
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_a_league',
                                                      'string_column:regular_season': '4th_western',
                                                      'number_column:regular_season': 4.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': 'did_not_qualify',
                                                      'number_column:open_cup': None,
                                                      'string_column:avg_attendance': '7_169',
                                                      'number_column:avg_attendance': 7169.0},
                                                     {'date_column:year': Date(2005, -1, -1),
                                                      'string_column:year': '2005',
                                                      'number_column:year': 2005.0,
                                                      'number_column:division': 2.0,
                                                      'string_column:division': '2',
                                                      'string_column:league': 'usl_first_division',
                                                      'string_column:regular_season': '5th',
                                                      'number_column:regular_season': 5.0,
                                                      'string_column:playoffs': 'quarterfinals',
                                                      'string_column:open_cup': '4th_round',
                                                      'number_column:open_cup': 4.0,
                                                      'string_column:avg_attendance': '6_028',
                                                      'number_column:avg_attendance': 6028.0}]

    def test_number_extraction(self):
        question = """how many players on the 191617 illinois fighting illini men's basketball team
                      had more than 100 points scored?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-7.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("191617", 5), ("100", 16)]

    def test_date_extraction(self):
        question = "how many laps did matt kenset complete on february 26, 2006."
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-8.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("2", 8), ("26", 9), ("2006", 11)]

    def test_date_extraction_2(self):
        question = """how many different players scored for the san jose earthquakes during their
                      1979 home opener against the timbers?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-6.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, number_entities = table_question_context.get_entities_from_question()
        assert number_entities == [("1979", 12)]

    def test_entity_extraction_from_question_with_quotes(self):
        question = "how many times does \"friendly\" appear in the competition column?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = 'fixtures/data/wikitables/tables/346.tagged'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        entities, _ = table_question_context.get_entities_from_question()
        assert entities == [('string:friendly', ['string_column:competition'])]

    def test_multiword_entity_extraction(self):
        question = "was the positioning better the year of the france venue or the year of the south korea venue?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-3.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        entities, _ = table_question_context.get_entities_from_question()
        assert entities == [("string:france", ["string_column:venue"]),
                            ("string:south_korea", ["string_column:venue"])]

    def test_rank_number_extraction(self):
        question = "what was the first tamil-language film in 1943?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-1.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        _, numbers = table_question_context.get_entities_from_question()
        assert numbers == [("1", 3), ('1943', 9)]

    def test_null_extraction(self):
        question = "on what date did the eagles score the least points?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-2.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        entities, numbers = table_question_context.get_entities_from_question()
        # "Eagles" does not appear in the table.
        assert entities == []
        assert numbers == []

    def test_numerical_column_type_extraction(self):
        question = """how many players on the 191617 illinois fighting illini men's basketball team
                      had more than 100 points scored?"""
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-7.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        data = table_question_context.table_data[0]
        assert "number_column:games_played" in data
        assert "number_column:field_goals" in data
        assert "number_column:free_throws" in data
        assert "number_column:points" in data

    def test_date_column_type_extraction_1(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-5.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        data = table_question_context.table_data[0]
        assert "date_column:first_elected" in data

    def test_date_column_type_extraction_2(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-9.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        data = table_question_context.table_data[0]
        assert "date_column:date_of_appointment" in data
        assert "date_column:date_of_election" in data

    def test_string_column_types_extraction(self):
        question = "how many were elected?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-10.table'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        data = table_question_context.table_data[0]
        assert "string_column:birthplace" in data
        assert "string_column:advocate" in data
        assert "string_column:notability" in data
        assert "string_column:name" in data

    def test_number_and_entity_extraction(self):
        question = "other than m1 how many notations have 1 in them?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f"{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-11.table"
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        string_entities, number_entities = table_question_context.get_entities_from_question()
        assert string_entities == [("string:m1", ["string_column:notation"]),
                                   ("string:1", ["string_column:position"])]
        assert number_entities == [("1", 2), ("1", 7)]

    def test_get_knowledge_graph(self):
        question = "other than m1 how many notations have 1 in them?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f"{self.FIXTURES_ROOT}/data/corenlp_processed_tables/TEST-11.table"
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        knowledge_graph = table_question_context.get_table_knowledge_graph()
        entities = knowledge_graph.entities
        # -1 is not in entities because there are no date columns in the table.
        assert sorted(entities) == ['1', 'number_column:notation',
                                    'number_column:position', 'string:1', 'string:m1',
                                    'string_column:mnemonic', 'string_column:notation',
                                    'string_column:position',
                                    'string_column:short_name', 'string_column:swara']
        neighbors = knowledge_graph.neighbors
        # Each number extracted from the question will have all number and date columns as
        # neighbors. Each string entity extracted from the question will only have the corresponding
        # column as the neighbor.
        assert set(neighbors['1']) == {'number_column:notation', 'number_column:position'}
        assert neighbors['string_column:mnemonic'] == []
        assert neighbors['string_column:short_name'] == []
        assert neighbors['string_column:swara'] == []
        assert neighbors['number_column:position'] == ['1']
        assert neighbors['number_column:notation'] == ['1']
        assert neighbors['string_column:position'] == ['string:1']
        assert neighbors['string:1'] == ['string_column:position']
        assert neighbors['string:m1'] == ['string_column:notation']
        assert neighbors['string_column:notation'] == ['string:m1']
        entity_text = knowledge_graph.entity_text
        assert entity_text == {'1': '1',
                               'string:m1': 'm1',
                               'string:1': '1',
                               'number_column:notation': 'notation',
                               'string_column:notation': 'notation',
                               'string_column:mnemonic': 'mnemonic',
                               'string_column:short_name': 'short name',
                               'string_column:swara': 'swara',
                               'string_column:position': 'position',
                               'number_column:position': 'position'}


    def test_knowledge_graph_has_correct_neighbors(self):
        question = "when was the attendance greater than 5000?"
        question_tokens = self.tokenizer.tokenize(question)
        test_file = f'{self.FIXTURES_ROOT}/data/wikitables/sample_table.tagged'
        table_question_context = TableQuestionContext.read_from_file(test_file, question_tokens)
        knowledge_graph = table_question_context.get_table_knowledge_graph()
        neighbors = knowledge_graph.neighbors
        # '5000' is neighbors with number and date columns. '-1' is in entities because there is a
        # date column, which is its only neighbor.
        assert set(neighbors.keys()) == {'date_column:year', 'number_column:year',
                                         'string_column:year', 'number_column:division',
                                         'string_column:division',
                                         'string_column:league', 'string_column:regular_season',
                                         'number_column:regular_season',
                                         'string_column:playoffs', 'string_column:open_cup',
                                         'number_column:open_cup', 'string_column:avg_attendance',
                                         'number_column:avg_attendance', '5000', '-1'}
        assert set(neighbors['date_column:year']) == {'5000', '-1'}
        assert neighbors['number_column:division'] == ['5000']
        assert neighbors['string_column:league'] == []
        assert neighbors['string_column:regular_season'] == []
        assert neighbors['string_column:playoffs'] == []
        assert neighbors['string_column:open_cup'] == []
        assert neighbors['number_column:avg_attendance'] == ['5000']
        assert set(neighbors['5000']) == {'date_column:year', 'number_column:division',
                                          'number_column:avg_attendance',
                                          'number_column:regular_season', 'number_column:year',
                                          'number_column:open_cup'}
        assert neighbors['-1'] == ['date_column:year']
