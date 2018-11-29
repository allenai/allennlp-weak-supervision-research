from typing import List, Dict, Tuple, Union, Any
from collections import defaultdict
import re
import logging
from unidecode import unidecode

from allennlp.semparse import util as semparse_util
from allennlp.semparse.worlds.world import ExecutionError
from allennlp.tools import wikitables_evaluator as evaluator

from weak_supervision.semparse.contexts import TableQuestionContext
from weak_supervision.semparse.contexts.table_question_context import (Date, CellValueType,
                                                                       MONTH_NUMBERS)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

NestedList = List[Union[str, List]]  # pylint: disable=invalid-name
RowListType = List[Dict[str, CellValueType]]  # pylint: disable=invalid-name


class WikiTablesVariableFreeExecutor:
    # pylint: disable=too-many-public-methods
    """
    Implements the functions in the variable free language we use, that's inspired by the one in
    "Memory Augmented Policy Optimization for Program Synthesis with Generalization" by Liang et al.

    Parameters
    ----------
    table_data : ``RowListType``
        All the rows in the table on which the executor will be used. The class expects each row to
        be represented as a dict from column names to corresponding cell values.
    """
    def __init__(self, table_data: List[Dict[str, CellValueType]]) -> None:
        self.table_data = table_data

    def __eq__(self, other):
        if not isinstance(other, WikiTablesVariableFreeExecutor):
            return False
        return self.table_data == other.table_data

    @staticmethod
    def _make_date(string: str) -> Date:
        parts = re.split('[-_]', string)
        year = -1
        month = -1
        day = -1
        for part in parts:
            if part.isdigit():
                # isdigit returns True for unicode numbers as well!
                part_int = int(unidecode(part))
                if len(part) == 4:
                    year = part_int
                else:
                    day = part_int
            if part in MONTH_NUMBERS:
                month = int(MONTH_NUMBERS[part])
        return Date(year, month, day)

    def execute(self, logical_form: str) -> Any:
        if not logical_form.startswith("("):
            logical_form = f"({logical_form})"
        logical_form = logical_form.replace(",", " ")
        expression_as_list = semparse_util.lisp_to_nested_expression(logical_form)
        # Expression list has an additional level of
        # nesting at the top. For example, if the
        # logical form is
        # "(select all_rows fb:row.row.league)",
        # the expression list will be
        # [['select', 'all_rows', 'fb:row.row.league']].
        # Removing the top most level of nesting.
        result = self._handle_expression(expression_as_list[0])
        return result

    def evaluate_logical_form(self, logical_form: str, target_list: List[str]) -> bool:
        """
        Takes a logical form, and the list of target values as strings from the original lisp
        string, and returns True iff the logical form executes to the target list.
        """
        # Normalize target value only if it is not a number.
        normalized_target_list = []
        for value in target_list:
            try:
                float(value)
                normalized_target_list.append(value)
            except ValueError:
                normalized_target_list.append(TableQuestionContext.normalize_string(value))
        try:
            denotation = self.execute(logical_form)
        except ExecutionError:
            logger.warning(f'Failed to execute: {logical_form}')
            return False
        if not isinstance(denotation, list):
            logger.warning(f"Logical form is not a list: {logical_form}")
            return False
        if denotation and isinstance(denotation[0], Date):
            normalized_target_list = [str(self._make_date(target)) for target in normalized_target_list]
        denotation_list = []
        denotation_list = [str(denotation_item) for denotation_item in denotation]
        target_value_list = evaluator.to_value_list(normalized_target_list)
        denotation_value_list = evaluator.to_value_list(denotation_list)
        return evaluator.check_denotation(target_value_list, denotation_value_list)

    ## Helper functions
    def _handle_expression(self, expression_list):
        if isinstance(expression_list, list) and len(expression_list) == 1:
            expression = expression_list[0]
        else:
            expression = expression_list
        if isinstance(expression, list):
            # This is a function application.
            function_name = expression[0]
        else:
            # This is a constant (like "all_rows" or "2005")
            return self._handle_constant(expression)
        try:
            function = getattr(self, function_name)
            return function(*expression[1:])
        except AttributeError:
            raise ExecutionError(f"Function not found: {function_name}")

    def _handle_constant(self, constant: str) -> Union[RowListType, str, float]:
        if constant == "all_rows":
            return self.table_data
        try:
            return float(constant)
        except ValueError:
            # The constant is not a number. Returning as-is if it is a string.
            if constant.startswith("string:"):
                return constant.replace("string:", "")
            raise ExecutionError(f"Cannot handle constant: {constant}")

    @staticmethod
    def _get_number_row_pairs_to_filter(row_list: RowListType,
                                        column_name: str,
                                        keep_none_values: bool = False) -> List[Tuple[float,
                                                                                      Dict[str, CellValueType]]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a number taken from that column, and the corresponding row
        as the second element. The output can be used to compare rows based on the numbers. Some of
        the values are None since not all rows contain values under a given column. In such cases,
        if you want to keep the values (say if you are using these in a negated condition like
        filter_*not_*), then set `keep_none_values` to True.
        """
        if not row_list:
            return []
        cell_row_pairs = [(row[column_name], row) for row in row_list
                          if row[column_name] is not None or keep_none_values]
        return cell_row_pairs

    @staticmethod
    def _get_date_row_pairs_to_filter(row_list: RowListType,
                                      column_name: str,
                                      keep_none_values: bool = False) -> List[Tuple[Date,
                                                                                    Dict[str, CellValueType]]]:
        """
        Helper method that takes a row list and a column name, and returns a list of tuples, each
        containing as the first element a date taken from that column, and the corresponding row as
        the second element. The output can be used to compare rows based on the dates. Some of
        the values are None since not all rows contain values under a given column. In such cases,
        if you want to keep the values (say if you are using these in a negated condition like
        filter_*not_*), then set `keep_none_values` to True.

        """
        if not row_list:
            return []
        cell_row_pairs = [(row[column_name], row) for row in row_list
                          if row[column_name] is not None or keep_none_values]
        return cell_row_pairs

    def _get_row_index(self, row: Dict[str, str]) -> int:
        """
        Takes a row and returns its index in the full list of rows. If the row does not occur in the
        table (which should never happen because this function will only be called with a row that
        is the result of applying one or more functions on the table rows), the method returns -1.
        """
        row_index = -1
        for index, table_row in enumerate(self.table_data):
            if table_row == row:
                row_index = index
                break
        return row_index

    ## Functions in the language
    def select_string(self, row_expression_list: NestedList, column_name: str) -> List[str]:
        """
        Select function takes a list of rows and a column name and returns a list of strings as
        in cells.
        """
        row_list = self._handle_expression(row_expression_list)
        assert column_name.startswith("string_column:")
        return [row[column_name] for row in row_list if row[column_name]]

    def select_number(self, row_expression_list: NestedList, column_name: str) -> List[float]:
        """
        Select function takes a row (as a list) and a column name and returns the numbers in that
        column.
        """
        row_list = self._handle_expression(row_expression_list)
        assert column_name.startswith("number_column:") or column_name.startswith("num2_column")
        return [row[column_name] for row in row_list if row[column_name] is not None]

    def select_date(self, row_expression_list: NestedList, column_name: str) -> List[Date]:
        """
        Select function takes a row as a list and a column name and returns the dates in that column.
        """
        row_list = self._handle_expression(row_expression_list)
        assert column_name.startswith("date_column:")
        return [row[column_name] for row in row_list if row[column_name] is not None]

    def argmax(self, row_expression_list: NestedList, column_name: str) -> RowListType:
        """
        Takes a list of rows and a column name and returns a list containing a single row (dict from
        columns to cells) that has the maximum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of `_select` and `_all_rows`.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        if "date_column:" in column_name:
            value_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        else:
            value_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)  # type: ignore
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        max_cell_value = sorted(value_row_pairs, key=lambda x: x[0], reverse=True)[0][0]
        return [row for value, row in value_row_pairs if value == max_cell_value]

    def argmin(self, row_expression_list: NestedList, column_name: str) -> RowListType:
        """
        Takes a list of rows and a column and returns a list containing a single row (dict from
        columns to cells) that has the minimum numerical value in the given column. We return a list
        instead of a single dict to be consistent with the return type of `_select` and `_all_rows`.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        if "date_column:" in column_name:
            value_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        else:
            value_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)  # type: ignore
        if not value_row_pairs:
            return []
        # Returns a list containing the row with the max cell value.
        min_cell_value = sorted(value_row_pairs, key=lambda x: x[0])[0][0]
        return [row for value, row in value_row_pairs if value == min_cell_value]

    def filter_number_greater(self,
                              row_expression_list: NestedList,
                              column_name: str,
                              value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is greater than the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value > value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_number_greater_equals(self,
                                     row_expression_list: NestedList,
                                     column_name: str,
                                     value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is greater than or equal to the given
        value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value >= value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_number_lesser(self,
                             row_expression_list: NestedList,
                             column_name: str,
                             value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is less than the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value < value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_number_lesser_equals(self,
                                    row_expression_list: NestedList,
                                    column_name: str,
                                    value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column is lesser than or equal to the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value <= value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_number_equals(self,
                             row_expression_list: NestedList,
                             column_name: str,
                             value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column equals the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value == value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_number_not_equals(self,
                                 row_expression_list: NestedList,
                                 column_name: str,
                                 value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column is not equal to the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name, True)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, float):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value != value for value in filter_values]):
                return_list.append(row)
        return return_list

    # Note that the following six methods are identical to the ones above, except that the filter
    # values are obtained from `_get_date_row_pairs_to_filter`.
    def filter_date_greater(self,
                            row_expression_list: NestedList,
                            column_name: str,
                            value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is greater than the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value > value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_date_greater_equals(self,
                                   row_expression_list: NestedList,
                                   column_name: str,
                                   value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is greater than or equal to the given
        value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value >= value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_date_lesser(self,
                           row_expression_list: NestedList,
                           column_name: str,
                           value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows as an expression, a column, and a numerical value expression and
        returns all the rows where the value in that column is less than the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value < value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_date_lesser_equals(self,
                                  row_expression_list: NestedList,
                                  column_name: str,
                                  value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column is lesser than or equal to the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value <= value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_date_equals(self,
                           row_expression_list: NestedList,
                           column_name: str,
                           value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column equals the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value == value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_date_not_equals(self,
                               row_expression_list: NestedList,
                               column_name: str,
                               value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a numerical value and returns all the rows where the
        value in that column is not equal to the given value.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name, True)
        filter_values = self._handle_expression(value_expression)
        if isinstance(filter_values, list) and filter_values:
            if not isinstance(filter_values[0], Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
        else:
            if not isinstance(filter_values, Date):
                raise ExecutionError(f"Invalid filter value: {value_expression}")
            filter_values = [filter_values]
        return_list = []
        for cell_value, row in cell_row_pairs:
            if all([cell_value != value for value in filter_values]):
                return_list.append(row)
        return return_list

    def filter_in(self,
                  row_expression_list: NestedList,
                  column_name: str,
                  value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a string value and returns all the rows where the value
        in that column contains the given string.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        expression_evaluation = self._handle_expression(value_expression)
        if isinstance(expression_evaluation, list) and expression_evaluation:
            if not all([isinstance(item, str) for item in expression_evaluation]):
                raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
            filter_values = expression_evaluation
        elif isinstance(expression_evaluation, str):
            filter_values = [expression_evaluation]
        else:
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.
        result_list = []
        for row in row_list:
            if row[column_name] is None:
                continue
            for filter_value in filter_values:
                if filter_value in row[column_name]:
                    result_list.append(row)
                    break
        return result_list

    def filter_not_in(self,
                      row_expression_list: NestedList,
                      column_name: str,
                      value_expression: NestedList) -> RowListType:
        """
        Takes a list of rows, a column, and a string value and returns all the rows where the value
        in that column does not contain the given string.
        """
        row_list = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        expression_evaluation = self._handle_expression(value_expression)
        if isinstance(expression_evaluation, list) and expression_evaluation:
            if not all([isinstance(item, str) for item in expression_evaluation]):
                raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
            filter_values = expression_evaluation
        elif isinstance(expression_evaluation, str):
            filter_values = [expression_evaluation]
        else:
            raise ExecutionError(f"Unexpected filter value for filter_in: {value_expression}")
        # Assuming filter value has underscores for spaces. The cell values also have underscores
        # for spaces, so we do not need to replace them here.
        result_list = []
        for row in row_list:
            cell_value = row[column_name]
            if cell_value is None:
                result_list.append(row)
                continue
            no_filter_value_in_cell = True
            for filter_value in filter_values:
                if filter_value in cell_value:
                    no_filter_value_in_cell = False
                    break
            if no_filter_value_in_cell:
                result_list.append(row)
        return result_list

    def first(self, row_expression_list: NestedList) -> RowListType:
        """
        Takes an expression that evaluates to a list of rows, and returns the first one in that
        list.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            logger.warning("Trying to get first row from an empty list: %s", row_expression_list)
            return []
        return [row_list[0]]

    def last(self, row_expression_list: NestedList) -> RowListType:
        """
        Takes an expression that evaluates to a list of rows, and returns the last one in that
        list.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            logger.warning("Trying to get last row from an empty list: %s", row_expression_list)
            return []
        return [row_list[-1]]

    def previous(self, row_expression_list: NestedList) -> RowListType:
        """
        Takes an expression that evaluates to a single row, and returns the row (as a list to be
        consistent with the rest of the API), that occurs before the input row in the original set
        of rows. If the input row happens to be the top row, we will return an empty list.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            logger.warning("Trying to get the previous row from an empty list: %s",
                           row_expression_list)
            return []
        input_row_indices = [self._get_row_index(row) for row in row_list]
        previous_row_indices = [i-1 for i in input_row_indices if i > 0]
        return [self.table_data[i] for i in previous_row_indices]

    def next(self, row_expression_list: NestedList) -> RowListType:
        """
        Takes an expression that evaluates to a single row, and returns the row (as a list to be
        consistent with the rest of the API), that occurs after the input row in the original set
        of rows. If the input row happens to be the last row, we will return an empty list.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            logger.warning("Trying to get the next row from an empty list: %s", row_expression_list)
            return []
        input_row_indices = [self._get_row_index(row) for row in row_list]
        last_row_index = len(self.table_data) - 1
        next_row_indices = [i+1 for i in input_row_indices if i+1 <= last_row_index and i != -1]
        return [self.table_data[i] for i in next_row_indices]

    def count(self, row_expression_list: NestedList) -> List[float]:
        """
        Takes an expression that evaluates to a a list of rows and returns their count (as a list
        with one float to be consistent with the other functions like max that also return numbers).
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        return [float(len(row_list))]

    def max_number(self,
                   row_expression_list: NestedList,
                   column_name: str) -> List[float]:
        """
        Takes an expression list that evaluates to a  list of rows and a column name, and returns the max
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        if not cell_row_pairs:
            return []
        return [max([value for value, _ in cell_row_pairs])]

    def min_number(self,
                   row_expression_list: NestedList,
                   column_name: str) -> List[float]:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the min
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        if not cell_row_pairs:
            return []
        return [min([value for value, _ in cell_row_pairs])]

    def max_date(self,
                 row_expression_list: NestedList,
                 column_name: str) -> List[Date]:
        """
        Takes an expression list that evaluates to a  list of rows and a column name, and returns the max
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        if not cell_row_pairs:
            return []
        return [max([value for value, _ in cell_row_pairs])]

    def min_date(self,
                 row_expression_list: NestedList,
                 column_name: str) -> List[Date]:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the min
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_date_row_pairs_to_filter(row_list, column_name)
        if not cell_row_pairs:
            return []
        return [min([value for value, _ in cell_row_pairs])]

    def sum(self,
            row_expression_list: NestedList,
            column_name) -> List[float]:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the sum
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        return [sum([value for value, _ in cell_row_pairs])]

    def average(self,
                row_expression_list: NestedList,
                column_name: str) -> List[float]:
        """
        Takes an expression list that evaluates to a  list of rows and a column, and returns the mean
        of the values under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        cell_row_pairs = self._get_number_row_pairs_to_filter(row_list, column_name)
        if not cell_row_pairs:
            return [0.0]
        return [sum([value for value, _ in cell_row_pairs]) / len(cell_row_pairs)]

    def mode_string(self,
                    row_expression_list: NestedList,
                    column_name: str) -> List[str]:
        """
        Takes an expression that evaluates to a list of rows, and a column and returns the most
        frequent values (one or more) under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in row_list:
            cell_value = row[column_name]
            # Ignore if cell value is None or an empty string
            if not cell_value:
                continue
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    def mode_number(self,
                    row_expression_list: NestedList,
                    column_name: str) -> List[float]:
        """
        Takes an expression that evaluates to a list of rows, and a column and returns the most
        frequent values (one or more) under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in row_list:
            cell_value = row[column_name]
            if cell_value is None:
                continue
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    def mode_date(self,
                  row_expression_list: NestedList,
                  column_name: str) -> List[Date]:
        """
        Takes an expression that evaluates to a list of rows, and a column and returns the most
        frequent values (one or more) under that column in those rows.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        value_frequencies: Dict[str, int] = defaultdict(int)
        max_frequency = 0
        most_frequent_list: List[str] = []
        for row in row_list:
            cell_value = row[column_name]
            if cell_value is None:
                continue
            value_frequencies[cell_value] += 1
            frequency = value_frequencies[cell_value]
            if frequency > max_frequency:
                max_frequency = frequency
                most_frequent_list = [cell_value]
            elif frequency == max_frequency:
                most_frequent_list.append(cell_value)
        return most_frequent_list

    def same_as(self,
                row_expression_list: NestedList,
                column_name: str) -> RowListType:
        """
        Takes an expression that evaluates to a row, and a column and returns a list of rows from
        the full set of rows that contain the same value under the given column as the given row.
        The returned set will not include the original row.
        """
        row_list: RowListType = self._handle_expression(row_expression_list)
        if not row_list:
            return []
        if len(row_list) > 1:
            logger.warning("same_as function got multiple rows. Taking the first one: "
                           f"{row_expression_list}")
        cell_value = row_list[0][column_name]
        return_list = []
        for row in self.table_data:
            if row[column_name] == cell_value and row != row_list[0]:
                return_list.append(row)
        return return_list

    def diff(self,
             first_row_expression_list: NestedList,
             second_row_expression_list: NestedList,
             column_name: str) -> List[float]:
        """
        Takes an expressions that evaluate to two rows, and a column name, and returns the
        difference between the values under that column in those two rows.
        """
        first_row_list = self._handle_expression(first_row_expression_list)
        second_row_list = self._handle_expression(second_row_expression_list)
        if not first_row_list or not second_row_list:
            return []
        if len(first_row_list) > 1:
            logger.warning("diff got multiple rows for first argument. Taking the first one: "
                           f"{first_row_expression_list}")
        if len(second_row_list) > 1:
            logger.warning("diff got multiple rows for second argument. Taking the first one: "
                           f"{second_row_expression_list}")
        first_row = first_row_list[0]
        second_row = second_row_list[0]
        try:
            first_value = float(first_row[column_name])
            second_value = float(second_row[column_name])
            return [abs(first_value - second_value)]
        except ValueError:
            raise ExecutionError(f"Invalid column for diff: {column_name}")
        except TypeError:
            # This means one of the values is None. It happens when one of the rows does not have a
            # value in the corresponding column.
            return []

    @staticmethod
    def date(year_string: str, month_string: str, day_string: str) -> List[Date]:
        """
        Takes three numbers as strings, and returns a ``Date`` object whose year, month, and day are
        the three numbers in that order.
        """
        try:
            year = int(str(year_string))
            month = int(str(month_string))
            day = int(str(day_string))
            return [Date(year, month, day)]
        except ValueError:
            raise ExecutionError(f"Invalid date! Got {year_string}, {month_string}, {day_string}")
