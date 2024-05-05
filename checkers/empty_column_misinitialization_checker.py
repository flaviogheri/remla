import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker

class EmptyColumnMisinitializationChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = 'empty-column-misinitialization'
    msgs = {
        'W0003': (
            'Empty column initialization detected. Use np.nan instead of 0 or empty string.',
            'empty-column-initialization',
            'Developers need to be careful when initializing empty columns',
        ),
    }

    def visit_assign(self, node):
        if (
            isinstance(node.targets[0], astroid.Subscript) and
            isinstance(node.targets[0].value, astroid.Name) and
            node.targets[0].value.name == 'df'
        ):
            if isinstance(node.value, astroid.Const):
                if node.value.value in [0, '']:
                    self.add_message('empty-column-initialization', node=node)



def register(linter):
    linter.register_checker(EmptyColumnMisinitializationChecker(linter))
