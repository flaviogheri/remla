import astroid
import pandas as pd
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class NanEquivalenceChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = 'nan-equivalence'
    msgs = {
        'W0001': (
            'Comparison of Pandas DataFrame elements with np.nan may return False. Use .isna() for NaN comparison.',
            'nan-equivalence',
            'Developers need to be careful when using the NaN comparison',
        ),
    }

    def visit_compare(self, node):
        if len(node.ops) > 0 and node.ops[0][0] == '==':
            if (
                isinstance(node.left, astroid.Name) and 
                node.left.name == 'df'
            ):
                if (
                    len(node.ops[0]) >= 2 and
                    isinstance(node.ops[0][1], astroid.Attribute) and
                    node.ops[0][1].attrname == 'nan' and
                    isinstance(node.ops[0][1].expr, astroid.Name) and
                    node.ops[0][1].expr.name == 'np'
                ):
                    self.add_message('nan-equivalence', node=node)


def register(linter):
    linter.register_checker(NanEquivalenceChecker(linter))
