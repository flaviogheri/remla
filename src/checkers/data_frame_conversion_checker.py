'''
This checker is used to detect the usage of the
deprecated attribute `df.values` in pandas DataFrame.
'''

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class DataFrameConversionChecker(BaseChecker):
    '''
    This pylint checker will check for the usage of
    the deprecated attribute `df.values` in pandas DataFrame.
    '''

    __implements__ = IAstroidChecker

    name = 'dataframe-conversion'
    msgs = {
        'W0004': (
            'Use df.to_numpy() instead of df.values.',
            'dataframe-conversion',
            'Using df.to_numpy() reduces error-prone code.'
        ),
    }

    def visit_attribute(self, node):
        '''
        This method will check if the node is a df.values call.
        '''
        if (isinstance(node.expr, astroid.Name) and
                node.expr.name == 'df' and
                node.attrname == 'values'):
            self.add_message('dataframe-conversion', node=node)


def register(linter):
    '''
    This function registers the DataFrameConversionChecker with the linter.
    '''
    linter.register_checker(DataFrameConversionChecker(linter))
