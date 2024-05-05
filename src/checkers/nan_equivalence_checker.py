'''
This checker is used to detect the comparison of NaN values in the code.
It is recommended to use the .isna() method for NaN comparison.
'''

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class NanEquivalenceChecker(BaseChecker):
    '''
    This pylint checker will check for NaN comparison in the code.
    '''
    __implements__ = IAstroidChecker

    name = 'nan-equivalence'
    msgs = {
        'W0001': (
            'Use .isna() for NaN comparison.',
            'nan-equivalence',
            'Developers need to be careful when using the NaN comparison',
        ),
    }

    def visit_compare(self, node):
        '''
        This method will check if the node is a NaN comparison operation.
        '''
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
    '''
    This function registers the NanEquivalenceChecker with the linter.
    '''
    linter.register_checker(NanEquivalenceChecker(linter))
