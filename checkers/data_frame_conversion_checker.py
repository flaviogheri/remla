import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class DataFrameConversionChecker(BaseChecker):
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
        if (isinstance(node.expr, astroid.Name) and
                node.expr.name == 'df' and
                node.attrname == 'values'):
            self.add_message('dataframe-conversion', node=node)


def register(linter):
    linter.register_checker(DataFrameConversionChecker(linter))
