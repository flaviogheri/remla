import astroid
from pylint.checkers import BaseChecker
from pylint.interfaces import IAstroidChecker


class ChainIndexingChecker(BaseChecker):
    __implements__ = IAstroidChecker

    name = 'chain-indexing'
    msgs = {
        'W0002': (
            'Use .loc or .iloc for multidimensional indexing.',
            'chain-indexing',
            'Developers need to be careful when using chain indexing',
        ),
    }

    def visit_subscript(self, node):
        if isinstance(node.value, astroid.Subscript):
            self.add_message('chain-indexing', node=node)


def register(linter):
    linter.register_checker(ChainIndexingChecker(linter))
