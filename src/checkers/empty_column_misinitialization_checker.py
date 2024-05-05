'''
This checker is used to detect the misinitialization
of empty columns in a pandas DataFrame.
'''

import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class EmptyColumnMisinitializationChecker(BaseChecker):
    '''
    This pylint checker will check for the misinitialization
    of empty columns.
    '''
    __implements__ = IAstroidChecker

    name = 'empty-column-misinitialization'
    msgs = {
        'W0003': (
            'Use np.nan instead of 0 or empty string for empty columns.',
            'empty-column-initialization',
            'Developers need to be careful when initializing empty columns',
        ),
    }

    def visit_assign(self, node):
        '''
        This method will check if the node is an empty column initialization.
        '''
        if (
            isinstance(node.targets[0], astroid.Subscript) and
            isinstance(node.targets[0].value, astroid.Name) and
            node.targets[0].value.name == 'df'
        ):
            if isinstance(node.value, astroid.Const):
                if node.value.value in [0, '']:
                    self.add_message('empty-column-initialization', node=node)


def register(linter):
    '''
    This function registers the
    EmptyColumnMisinitializationChecker with the linter.
    '''
    linter.register_checker(EmptyColumnMisinitializationChecker(linter))
