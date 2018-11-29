# pylint: disable=no-self-use,invalid-name,too-many-public-methods
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.tokenizers import WordTokenizer
from allennlp.semparse.worlds.world import ExecutionError

from weak_supervision.semparse.contexts import TableQuestionContext
from weak_supervision.semparse.executors import WikiTablesVariableFreeExecutor
from weak_supervision.semparse.executors.wikitables_variable_free_executor import Date


class TestWikiTablesVariableFreeExecutor(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        table_data = [{"date_column:date": Date(2001, 1, -1), "number_column:division": 2.0,
                       "num2_column:division": 5.0,
                       "string_column:league": "usl_a_league", "string_column:regular_season": "4th_western",
                       "string_column:playoffs": "quarterfinals", "string_column:open_cup": "did_not_qualify",
                       "number_column:avg_attendance": 7169.0},
                      {"date_column:date": Date(2005, 3, -1), "number_column:division": 2.0,
                       "num2_column:division": 3.0,
                       "string_column:league": "usl_first_division", "string_column:regular_season": "5th",
                       "string_column:playoffs": "quarterfinals", "string_column:open_cup": "4th_round",
                       "number_column:avg_attendance": 6028.0}]
        self.executor = WikiTablesVariableFreeExecutor(table_data)

    def test_execute_fails_with_unknown_function(self):
        logical_form = "(unknown_function all_rows string_column:league)"
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_select(self):
        logical_form = "(select_string all_rows string_column:league)"
        cell_list = self.executor.execute(logical_form)
        assert set(cell_list) == {'usl_a_league', 'usl_first_division'}

    def test_execute_works_with_select_number(self):
        logical_form = "(select_number all_rows number_column:division)"
        selected_number = self.executor.execute(logical_form)
        assert selected_number == [2.0, 2.0]
        logical_form = "(select_number all_rows num2_column:division)"
        selected_number = self.executor.execute(logical_form)
        # Returns the first number.
        assert selected_number == [5.0, 3.0]

    def test_execute_works_with_argmax(self):
        logical_form = "(select_string (argmax all_rows number_column:avg_attendance) string_column:league)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_argmax_on_dates(self):
        logical_form = "(select_string (argmax all_rows date_column:date) string_column:league)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ['usl_first_division']

    def test_execute_works_with_argmin(self):
        logical_form = "(select_date (argmin all_rows number_column:avg_attendance) date_column:date)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list[0] == Date(2005, 3, -1)

    def test_execute_works_with_argmin_on_dates(self):
        logical_form = "(select_string (argmin all_rows date_column:date) string_column:league)"
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ['usl_a_league']

    def test_execute_works_with_filter_number_greater(self):
        # Selecting cell values from all rows that have attendance greater than the min value of
        # attendance.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                                   (min_number all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                                   all_rows) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_greater all_rows number_column:avg_attendance
                            string:usl_first_division) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_greater(self):
        # Selecting cell values from all rows that have date greater than 2002.
        logical_form = """(select_string (filter_date_greater all_rows date_column:date
                                   (date 2002 -1 -1)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_greater all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_number_greater_equals(self):
        # Counting rows that have attendance greater than or equal to the min value of attendance.
        logical_form = """(count (filter_number_greater_equals all_rows number_column:avg_attendance
                                  (min_number all_rows number_column:avg_attendance)))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == [2.0]
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater all_rows number_column:avg_attendance all_rows))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_greater_equals all_rows number_column:avg_attendance
                                  string:usl_a_league))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_greater_equals(self):
        # Selecting cell values from all rows that have date greater than or equal to 2005 February
        # 1st.
        logical_form = """(select_string (filter_date_greater_equals all_rows date_column:date
                                   (date 2005 2 1)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_greater_equals all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_number_lesser(self):
        # Selecting cell values from all rows that have date lesser than 2005.
        logical_form = """(select_string (filter_number_lesser all_rows number_column:avg_attendance
                                    (max_number all_rows number_column:avg_attendance)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_lesser all_rows date_column:date
                                   (date 2005 -1 -1)) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_lesser all_rows date_column:date
                                          string:5th) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_lesser(self):
        # Selecting cell values from all rows that have date less that 2005 January
        logical_form = """(select_string (filter_date_lesser all_rows date_column:date
                                   (date 2005 1 -1)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ["usl_a_league"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_lesser all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_number_lesser_equals(self):
        # Counting rows that have year lesser than or equal to 2005.
        logical_form = """(count (filter_number_lesser_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == [2.0]
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_number_lesser_equals all_rows date_column:date
                                   (date 2005 -1 -1)) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_lesser_equals(self):
        # Selecting cell values from all rows that have date less that or equal to 2001 February 23
        logical_form = """(select_string (filter_date_lesser_equals all_rows date_column:date
                                   (date 2001 2 23)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_lesser_equals all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_number_equals(self):
        # Counting rows that have year equal to 2010.
        logical_form = """(count (filter_number_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == [0.0]
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_equals all_rows date_column:date (date 2010 -1 -1)))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select_string (filter_date_equals all_rows date_column:date
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_a_league']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_equals all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_number_not_equals(self):
        # Counting rows that have year not equal to 2010.
        logical_form = """(count (filter_number_not_equals all_rows number_column:avg_attendance 8000))"""
        count_result = self.executor.execute(logical_form)
        assert count_result == [2.0]
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_not_equals all_rows date_column:date (date 2010 -1 -1)))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)
        # Replacing the filter value with an invalid value.
        logical_form = """(count (filter_number_not_equals all_rows number_column:avg_attendance
                                  string:usl_a_league))"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_date_not_equals(self):
        # Selecting cell values from all rows that have date not equal to 2001
        logical_form = """(select_string (filter_date_not_equals all_rows date_column:date
                                   (date 2001 -1 -1)) string_column:league)"""
        cell_value_list = self.executor.execute(logical_form)
        assert cell_value_list == ['usl_first_division']
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_date_not_equals all_rows date_column:date
                                   2005) string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_filter_in(self):
        # Selecting "regular season" from rows that have "did not qualify" in "open cup" column.
        logical_form = """(select_string (filter_in all_rows string_column:open_cup string:did_not_qualify)
                                  string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["4th_western"]
        tokens = WordTokenizer().tokenize("how many songs charted above the 10th position on any chart?")
        context = TableQuestionContext.read_from_file("fixtures/data/wikitables/tables/654.tagged",
                                                      tokens)
        executor = WikiTablesVariableFreeExecutor(context.table_data)
        value = executor.execute("""(count (filter_in all_rows string_column:year (select_string all_rows
                                     string_column:uk)))""")
        assert value == [3.0]

    def test_execute_works_with_filter_not_in(self):
        # Selecting "regular season" from rows that do not have "did not qualify" in "open cup" column.
        logical_form = """(select_string (filter_not_in all_rows string_column:open_cup string:did_not_qualify)
                                   string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["5th"]
        # Replacing the filter value with an invalid value.
        logical_form = """(select_string (filter_not_in all_rows string_column:open_cup
                                          (date 2000 -1 -1)) string_column:regular_season)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_execute_works_with_first(self):
        # Selecting "regular season" from the first row.
        logical_form = """(select_string (first all_rows) string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_logs_warning_with_first_on_empty_list(self):
        # Selecting "regular season" from the first row where year is greater than 2010.
        with self.assertLogs("weak_supervision.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select_string (first (filter_date_greater all_rows date_column:date
                                                (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:weak_supervision.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get first row from an empty list: "
                          "['filter_date_greater', 'all_rows', 'date_column:date', ['date', '2010', '-1', '-1']]"])

    def test_execute_works_with_last(self):
        # Selecting "regular season" from the last row where year is not equal to 2010.
        logical_form = """(select_string (last (filter_date_not_equals all_rows date_column:date
                                         (date 2010 -1 -1)))
                                  string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["5th"]

    def test_execute_logs_warning_with_last_on_empty_list(self):
        # Selecting "regular season" from the last row where year is greater than 2010.
        with self.assertLogs("weak_supervision.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select_string (last (filter_date_greater all_rows date_column:date
                                                (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:weak_supervision.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get last row from an empty list: "
                          "['filter_date_greater', 'all_rows', 'date_column:date', ['date', '2010', '-1', '-1']]"])

    def test_execute_works_with_previous(self):
        # Selecting "regular season" from the row before last where year is not equal to 2010.
        logical_form = """(select_string (previous (last (filter_date_not_equals
                                                    all_rows date_column:date (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["4th_western"]

    def test_execute_logs_warning_with_previous_on_empty_list(self):
        # Selecting "regular season" from the row before the one where year is greater than 2010.
        with self.assertLogs("weak_supervision.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select_string (previous (filter_date_greater all_rows date_column:date
                                                         (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:weak_supervision.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get the previous row from an empty list: "
                          "['filter_date_greater', 'all_rows', 'date_column:date', ['date', '2010', '-1', '-1']]"])

    def test_execute_works_with_next(self):
        # Selecting "regular season" from the row after first where year is not equal to 2010.
        logical_form = """(select_string (next (first (filter_date_not_equals
                                                all_rows date_column:date (date 2010 -1 -1))))
                                  string_column:regular_season)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["5th"]

    def test_execute_logs_warning_with_next_on_empty_list(self):
        # Selecting "regular season" from the row after the one where year is greater than 2010.
        with self.assertLogs("weak_supervision.semparse.executors.wikitables_variable_free_executor") as log:
            logical_form = """(select_string (next (filter_date_greater all_rows date_column:date
                                                    (date 2010 -1 -1)))
                                      string_column:regular_season)"""
            self.executor.execute(logical_form)
        self.assertEqual(log.output,
                         ["WARNING:weak_supervision.semparse.executors.wikitables_variable_free_executor:"
                          "Trying to get the next row from an empty list: "
                          "['filter_date_greater', 'all_rows', 'date_column:date', ['date', '2010', '-1', '-1']]"])

    def test_execute_works_with_max_date(self):
        logical_form = """(max_date all_rows date_column:date)"""
        cell_list = self.executor.execute(logical_form)
        assert str(cell_list[0]) == "2005-3--1"

    def test_execute_works_with_min_date(self):
        logical_form = """(min_date all_rows date_column:date)"""
        cell_list = self.executor.execute(logical_form)
        assert str(cell_list[0]) == "2001-1--1"

    def test_execute_works_with_mode_number(self):
        # Most frequent division value.
        logical_form = """(mode_number all_rows number_column:division)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == [2.0]

    def test_execute_works_with_mode_string(self):
        logical_form = """(mode_string all_rows string_column:league)"""
        cell_list = self.executor.execute(logical_form)
        # Returns the string values with frequency 1 (which is the max frequency)
        assert cell_list == ["usl_a_league", "usl_first_division"]

    def test_execute_works_with_mode_date(self):
        logical_form = """(mode_date all_rows date_column:date)"""
        cell_list = self.executor.execute(logical_form)
        assert str(cell_list[0]) == "2001-1--1"

    def test_execute_works_with_same_as(self):
        # Select the "league" from all the rows that have the same value under "playoffs" as the
        # row that has the string "a league" under "league".
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league string:a_league)
                                   string_column:playoffs)
                           string_column:league)"""
        cell_list = self.executor.execute(logical_form)
        assert cell_list == ["usl_first_division"]

    def test_execute_works_with_sum(self):
        # Get total "avg attendance".
        logical_form = """(sum all_rows number_column:avg_attendance)"""
        sum_value = self.executor.execute(logical_form)
        assert sum_value == [13197.0]
        # Total "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(sum (filter_in all_rows string_column:playoffs string:quarterfinals)
                                number_column:avg_attendance)"""
        sum_value = self.executor.execute(logical_form)
        assert sum_value == [13197.0]

    def test_execute_works_with_average(self):
        # Get average "avg attendance".
        logical_form = """(average all_rows number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == [6598.5]
        # Average "avg attendance" where "playoffs" has "quarterfinals"
        logical_form = """(average (filter_in all_rows string_column:playoffs string:quarterfinals)
                                number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == [6598.5]
        # Average attendance where playoffs has "a_string_not_in_table". Average should be 0.
        logical_form = """(average (filter_in all_rows string_column:playoffs
                                    string:a_string_not_in_table) number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == [0.0]

    def test_execute_works_with_diff(self):
        # Difference in "avg attendance" between rows with "usl_a_league" and "usl_first_division"
        # in "league" columns.
        logical_form = """(diff (filter_in all_rows string_column:league string:usl_a_league)
                                (filter_in all_rows string_column:league string:usl_first_division)
                                number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        assert avg_value == [1141.0]
        logical_form = """(diff (filter_in all_rows string_column:league string:usl_first_division)
                                (filter_in all_rows string_column:league string:usl_a_league)
                                number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        # Difference is an absolute value.
        assert avg_value == [1141.0]
        logical_form = """(diff (filter_in all_rows string_column:league string:unknown_league)
                                (filter_in all_rows string_column:league string:usl_a_league)
                                number_column:avg_attendance)"""
        avg_value = self.executor.execute(logical_form)
        # First set of rows is empty. So diff is an empty list.
        assert avg_value == []

    def test_execute_fails_with_diff_on_non_numerical_columns(self):
        logical_form = """(diff (filter_in all_rows string_column:league string:usl_a_league)
                                (filter_in all_rows string_column:league string:usl_first_division)
                                string_column:league)"""
        with self.assertRaises(ExecutionError):
            self.executor.execute(logical_form)

    def test_date_comparison_works(self):
        assert Date(2013, 12, 31) > Date(2013, 12, 30)
        assert Date(2013, 12, 31) == Date(2013, 12, -1)
        assert Date(2013, -1, -1) >= Date(2013, 12, 31)
        # pylint: disable=singleton-comparison
        assert (Date(2013, 12, -1) > Date(2013, 12, 31)) == False
        assert (Date(2013, 12, 31) > 2013) == False
        assert (Date(2013, 12, 31) >= 2013) == False
        assert Date(2013, 12, 31) != 2013
        assert (Date(2018, 1, 1) >= Date(-1, 2, 1)) == False
        assert (Date(2018, 1, 1) < Date(-1, 2, 1)) == False
        # When year is unknown in both cases, we can compare months and days.
        assert Date(-1, 2, 1) < Date(-1, 2, 3)
        # If both year and month are not know in both cases, the comparison is undefined, and both
        # < and >= return False.
        assert (Date(-1, -1, 1) < Date(-1, -1, 3)) == False
        assert (Date(-1, -1, 1) >= Date(-1, -1, 3)) == False
        # Same when year is known, buth months are not.
        assert (Date(2018, -1, 1) < Date(2018, -1, 3)) == False
        # TODO (pradeep): Figure out whether this is expected behavior by looking at data.
        assert (Date(2018, -1, 1) >= Date(2018, -1, 3)) == False

    def test_number_comparison_works(self):
        # TableQuestionContext normlaizes all strings according to some rules. We want to ensure
        # that the original numerical values of number cells is being correctly processed here.
        tokens = WordTokenizer().tokenize("when was the attendance the highest?")
        tagged_file = self.FIXTURES_ROOT / "data" / "corenlp_processed_tables" / "TEST-2.table"
        context = TableQuestionContext.read_from_file(tagged_file, tokens)
        executor = WikiTablesVariableFreeExecutor(context.table_data)
        result = executor.execute("(select_date (argmax all_rows number_column:attendance) date_column:date)")
        assert result[0] == Date(-1, 11, 10)

    def test_evaluate_logical_form(self):
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league string:a_league)
                                   string_column:playoffs)
                           string_column:league)"""
        assert self.executor.evaluate_logical_form(logical_form, ["USL First Division"])

    def test_evaluate_logical_form_with_invalid_logical_form(self):
        logical_form = """(select_string (same_as (filter_in all_rows string_column:league INVALID_CONSTANT)
                                   string_column:playoffs)
                           string_column:league)"""
        assert not self.executor.evaluate_logical_form(logical_form, ["USL A-League",
                                                                      "USL First Division"])
