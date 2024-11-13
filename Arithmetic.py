'''
This is a "simple" homework to practice parsing grammars and working with the resulting parse tree.
'''


import lark
from math import floor, ceil

grammar = r"""
    start: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: atom
        | product "*" atom  -> mul
        | product "/" atom  -> div
        | product "%" atom  -> mod
        | product "**" atom -> exp
        | product "(" sum ")" -> impmul

    ?atom: NUMBER           -> number
        | "(" sum ")"       -> paren

    NUMBER: /-?[0-9]+/

    %import common.WS_INLINE
    %ignore WS_INLINE
"""
parser = lark.Lark(grammar)


class Interpreter(lark.visitors.Interpreter):
    '''
    Compute the value of the expression.
    The interpreter class processes nodes "top down",
    starting at the root and recursively evaluating subtrees.

    Addition, Subtraction, Multiplication:

    >>> interpreter = Interpreter()
    >>> interpreter.visit(parser.parse("1"))
    1
    >>> interpreter.visit(parser.parse("-1"))
    -1
    >>> interpreter.visit(parser.parse("1+2"))
    3
    >>> interpreter.visit(parser.parse("1-2"))
    -1
    >>> interpreter.visit(parser.parse("(1+2)*3"))
    9
    >>> interpreter.visit(parser.parse("1+2*3"))
    7
    >>> interpreter.visit(parser.parse("1*2+3"))
    5
    >>> interpreter.visit(parser.parse("1*(2+3)"))
    5
    >>> interpreter.visit(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> interpreter.visit(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> interpreter.visit(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> interpreter.visit(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14

    Modular division:

    >>> interpreter.visit(parser.parse("1%2"))
    1
    >>> interpreter.visit(parser.parse("3%2"))
    1
    >>> interpreter.visit(parser.parse("(1+2)%3"))
    0

    Exponentiation:

    >>> interpreter.visit(parser.parse("2**1"))
    2
    >>> interpreter.visit(parser.parse("2**2"))
    4
    >>> interpreter.visit(parser.parse("2**3"))
    8
    >>> interpreter.visit(parser.parse("1+2**3"))
    9
    >>> interpreter.visit(parser.parse("(1+2)**3"))
    27
    >>> interpreter.visit(parser.parse("1+2**3+4"))
    13
    >>> interpreter.visit(parser.parse("(1+2)**(3+4)"))
    2187
    >>> interpreter.visit(parser.parse("(1+2)**3-4"))
    23

    Exponentiation with negative exponents:

    >>> interpreter.visit(parser.parse("2**-1"))
    0
    >>> interpreter.visit(parser.parse("2**(-1)"))
    0
    >>> interpreter.visit(parser.parse("(1+2)**(3-4)"))
    0
    >>> interpreter.visit(parser.parse("1+2**(3-4)"))
    1
    >>> interpreter.visit(parser.parse("1+2**(-3)*4"))
    1

    Implicit multiplication:

    >>> interpreter.visit(parser.parse("1+2(3)"))
    7
    >>> interpreter.visit(parser.parse("1(2(3))"))
    6
    >>> interpreter.visit(parser.parse("(1)(2)(3)"))
    6
    >>> interpreter.visit(parser.parse("(1)(2)+(3)"))
    5
    >>> interpreter.visit(parser.parse("(1+2)(3+4)"))
    21
    >>> interpreter.visit(parser.parse("(1+2)(3(4))"))
    36
    '''

    def number(self, tree):
        return int(tree.children[0].value)

    def start(self, tree):
        return self.visit(tree.children[0]) 

    def add(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 + v1

    def sub(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 - v1

    def mul(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 * v1
    
    def div(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 // v1
    
    def mod(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 % v1
    
    def exp(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return floor(pow(v0, v1))
        
    def impmul(self, tree):
        v0 = self.visit(tree.children[0])
        v1 = self.visit(tree.children[1])
        return v0 * v1

    def paren(self, tree):
        return self.start(tree)


class Simplifier(lark.Transformer):
    '''
    Compute the value of the expression.
    The lark.Transformer class processes nodes "bottom up",
    starting at the leaves and ending at the root.
    In general, the Transformer class is less powerful than the Interpreter class.
    But in the case of simple arithmetic expressions,
    both classes can be used to evaluate the expression.

    Addition, Subtraction, Multiplication:

    >>> simplifier = Simplifier()
    >>> simplifier.transform(parser.parse("1"))
    1
    >>> simplifier.transform(parser.parse("-1"))
    -1
    >>> simplifier.transform(parser.parse("1+2"))
    3
    >>> simplifier.transform(parser.parse("1-2"))
    -1
    >>> simplifier.transform(parser.parse("(1+2)*3"))
    9
    >>> simplifier.transform(parser.parse("1+2*3"))
    7
    >>> simplifier.transform(parser.parse("1*2+3"))
    5
    >>> simplifier.transform(parser.parse("1*(2+3)"))
    5
    >>> simplifier.transform(parser.parse("(1*2)+3*4*(5-6)"))
    -10
    >>> simplifier.transform(parser.parse("((1*2)+3*4)*(5-6)"))
    -14
    >>> simplifier.transform(parser.parse("(1*(2+3)*4)*(5-6)"))
    -20
    >>> simplifier.transform(parser.parse("((1*2+(3)*4))*(5-6)"))
    -14

    Modular division:

    >>> simplifier.transform(parser.parse("1%2"))
    1
    >>> simplifier.transform(parser.parse("3%2"))
    1
    >>> simplifier.transform(parser.parse("(1+2)%3"))
    0

    Exponentiation:

    >>> simplifier.transform(parser.parse("2**1"))
    2
    >>> simplifier.transform(parser.parse("2**2"))
    4
    >>> simplifier.transform(parser.parse("2**3"))
    8
    >>> simplifier.transform(parser.parse("1+2**3"))
    9
    >>> simplifier.transform(parser.parse("(1+2)**3"))
    27
    >>> simplifier.transform(parser.parse("1+2**3+4"))
    13
    >>> simplifier.transform(parser.parse("(1+2)**(3+4)"))
    2187
    >>> simplifier.transform(parser.parse("(1+2)**3-4"))
    23

    Exponentiation with negative exponents:

    >>> simplifier.transform(parser.parse("2**-1"))
    0
    >>> simplifier.transform(parser.parse("2**(-1)"))
    0
    >>> simplifier.transform(parser.parse("(1+2)**(3-4)"))
    0
    >>> simplifier.transform(parser.parse("1+2**(3-4)"))
    1
    >>> simplifier.transform(parser.parse("1+2**(-3)*4"))
    1

    Implicit multiplication:

    >>> simplifier.transform(parser.parse("1+2(3)"))
    7
    >>> simplifier.transform(parser.parse("1(2(3))"))
    6
    >>> simplifier.transform(parser.parse("(1)(2)(3)"))
    6
    >>> simplifier.transform(parser.parse("(1)(2)+(3)"))
    5
    >>> simplifier.transform(parser.parse("(1+2)(3+4)"))
    21
    >>> simplifier.transform(parser.parse("(1+2)(3(4))"))
    36
    '''

    def number(self, children):
        return int(children[0].value)

    def start(self, children):
        return children[0]

    def add(self, children):
        return children[0] + children[1]

    def sub(self, children):
        return children[0] - children[1]

    def mul(self, children):
        return children[0] * children[1]
    
    def div(self, children):
        return children[0] // children[1]
    
    def mod(self, children):
        return children[0] % children[1]
    
    def exp(self, children):
        return floor(pow(children[0], children[1]))
        
    def impmul(self, children):
        return children[0] * children[1]

    def paren(self, children):
        return children[0]


########################################
# other transformations
########################################


def minify(expr):
    '''
    "Minifying" code is the process of removing unnecessary characters.
    In our arithmetic language, this means removing unnecessary whitespace and unnecessary parentheses.
    It is common to minify code in order to save disk space and bandwidth.
    For example, google penalizes a web site's search ranking if they don't minify their html/javascript code.

    >>> minify("1 + 2")
    '1+2'
    >>> minify("1 + ((((2))))")
    '1+2'
    >>> minify("1 + (2*3)")
    '1+2*3'
    >>> minify("1 + (2/3)")
    '1+2/3'
    >>> minify("(1 + 2)*3")
    '(1+2)*3'
    >>> minify("(1 - 2)*3")
    '(1-2)*3'
    >>> minify("(1 - 2)+3")
    '1-2+3'
    >>> minify("(1 + 2)+(3 + 4)")
    '1+2+3+4'
    >>> minify("(1 + 2)*(3 + 4)")
    '(1+2)*(3+4)'
    >>> minify("1 + (((2)*(3)) + 4)")
    '1+2*3+4'
    >>> minify("1 + (((2)*(3)) + 4 * ((5 + 6) - 7))")
    '1+2*3+4*(5+6-7)'
    '''

    class Mini(lark.Transformer):

        def remover(self, input):
            while input[0] == "(" and input[-1] == ")":
                input = input[1:-1]
            return input
        
        def highRemover(self, input):
            if input[0] != "(" or input[-1] != ")":
                return input
            level = 0
            for symbol in input[1:-1]:
                if symbol == "(":
                    level += 1
                elif symbol == ")":
                    level -= 1
                if level == 0:
                    if symbol == "+" or symbol == "-":
                        return input
            return input[1:-1]

        def number(self, children):
            return children[0].value

        def start(self, children):
            return children[0]

        def add(self, children):
            v0 = self.remover(children[0])
            v1 = self.remover(children[1])
            return v0 + "+" + v1

        def sub(self, children):
            v0 = self.remover(children[0])
            v1 = self.remover(children[1])
            return v0 + "-" + v1

        def mul(self, children):
            v0 = self.highRemover(children[0])
            v1 = self.highRemover(children[1])
            return v0 + "*" + v1
        
        def div(self, children):
            v0 = self.highRemover(children[0])
            v1 = self.highRemover(children[1])
            return v0 + "/" + v1
        
        def mod(self, children):
            v0 = self.highRemover(children[0])
            v1 = self.highRemover(children[1])
            return v0 + "%" + v1
        
        def exp(self, children):
            v0 = self.highRemover(children[0])
            v1 = self.highRemover(children[1])
            return v0 + "**" + v1
            
        def impmul(self, children):
            return children[0] + "(" + children[1] + ")"

        def paren(self, children):
            return "(" + children[0] + ")"

    mini = Mini()
    return mini.transform(parser.parse(expr))

def infix_to_rpn(expr):
    '''
    This function takes an expression in standard infix notation and converts it into an expression in reverse polish notation.
    This type of translation task is commonly done by first converting the input expression into an AST (i.e. by calling parser.parse),
    and then simplifying the AST in a leaf-to-root manner (i.e. using the Transformer class).

    HINT:
    If you need help understanding reverse polish notation,
    see the eval_rpn function.

    >>> infix_to_rpn('1')
    '1'
    >>> infix_to_rpn('1+2')
    '1 2 +'
    >>> infix_to_rpn('1-2')
    '1 2 -'
    >>> infix_to_rpn('(1+2)*3')
    '1 2 + 3 *'
    >>> infix_to_rpn('1+2*3')
    '1 2 3 * +'
    >>> infix_to_rpn('1*2+3')
    '1 2 * 3 +'
    >>> infix_to_rpn('1*(2+3)')
    '1 2 3 + *'
    >>> infix_to_rpn('(1*2)+3+4*(5-6)')
    '1 2 * 3 + 4 5 6 - * +'
    '''
    
    class Rpn(lark.Transformer):

        def number(self, children):
            return children[0].value

        def start(self, children):
            return children[0]

        def add(self, children):
            return children[0] + " " + children[1] + " " + "+"

        def sub(self, children):
            return children[0] + " " + children[1] + " " + "-"

        def mul(self, children):
            return children[0] + " " + children[1] + " " + "*"
        
        def div(self, children):
            return children[0] + " " + children[1] + " " + "/"
        
        def mod(self, children):
            return children[0] + " " + children[1] + " " + "%"
        
        def exp(self, children):
            return children[0] + " " + children[1] + " " + "**"
            
        def impmul(self, children):
            return children[0] + " " + children[1] + " " + "*"

        def paren(self, children):
            return children[0]

    rpn = Rpn()
    return rpn.transform(parser.parse(expr))


def eval_rpn(expr):
    '''
    This function evaluates an expression written in RPN.

    RPN (Reverse Polish Notation) is an alternative syntax for arithmetic.
    It was widely used in the first scientific calculators because it is much easier to parse than standard infix notation.
    For example, parentheses are never needed to disambiguate order of operations.
    Parsing of RPN is so easy, that it is usually done at the same time as evaluation without a separate parsing phase.
    More complicated languages (like the infix language above) are basically always implemented with separate parsing/evaluation phases.

    You can find more details on wikipedia: <https://en.wikipedia.org/wiki/Reverse_Polish_notation>.

    >>> eval_rpn("1")
    1
    >>> eval_rpn("1 2 +")
    3
    >>> eval_rpn("1 2 -")
    -1
    >>> eval_rpn("1 2 + 3 *")
    9
    >>> eval_rpn("1 2 3 * +")
    7
    >>> eval_rpn("1 2 * 3 +")
    5
    >>> eval_rpn("1 2 3 + *")
    5
    >>> eval_rpn("1 2 * 3 + 4 5 6 - * +")
    1
    '''
    tokens = expr.split()
    stack = []
    operators = {
        '+': lambda a, b: a+b,
        '-': lambda a, b: a-b,
        '*': lambda a, b: a*b,
        '/': lambda a, b: a//b,
        }
    for token in tokens:
        if token not in operators.keys():
            stack.append(int(token))
        else:
            assert len(stack) >= 2
            v1 = stack.pop()
            v2 = stack.pop()
            stack.append(operators[token](v2, v1))
    assert len(stack) == 1
    return stack[0]