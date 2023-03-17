// Package expression provides functionality for Expression to parse and compute math expressions written in
// prefix-notation (lisp-like)
package expression

import (
	"fmt"
	"nn/pkg/wraperr"
)

// Expression represents operation, symbol (variable) or constant.
type Expression struct {
	*operation
	*tuple

	*symbol

	*number

	representation *sTree
}

// NewExpression parses given input and provides Expression to compute it.
//
// Throws ErrParse.
//
// Example of valid inputs:
//     (+ 1 2)
//     (sin (* 1 2))
//     (* x0 x1)
func NewExpression(input string) (expr *Expression, err error) {
	defer wraperr.WrapError(ErrParse, &err)

	tree, err := splitRecursively(input) // build strings tree from input
	if err != nil {
		return nil, err
	}

	return newExpression(tree) // build expression from tree
}

// newExpression walks over provided sTree to parse it to Expression
func newExpression(tree *sTree) (*Expression, error) {
	root := []byte(tree.root)
	res := &Expression{representation: tree}

	if tree.hasChildren() { // only operation has children
		// first child of operation is operation's token
		token := tree.children[0].root
		operation, err := getOperation(token)
		if err != nil {
			return nil, fmt.Errorf("first token %q of %q is not an valid operation: %w", token, tree.root, err)
		}
		// other children of operation is it's arguments (e.g. tuple of expressions)
		tuple, err := newTuple(tree.children[1:])
		if err != nil {
			return nil, err
		}
		if !operation.checkInputsCount(len(tuple.expressions)) {
			return nil, fmt.Errorf("wrong arguments count for operation %q: required %d, provided %d",
				token, operation.inputsCount, len(tuple.expressions))
		}
		res.operation, res.tuple = operation, tuple
		return res, nil
	} else if numberPattern.Match(root) {
		num, err := newNumber(tree.root)
		if err != nil {
			return nil, err
		}
		res.number = num
		return res, nil
	} else if symbolPattern.Match(root) {
		sym, err := newSymbol(tree.root)
		if err != nil {
			return nil, err
		}
		res.symbol = sym
		return res, nil
	} else {
		return nil, fmt.Errorf("error parsing tree to expression:\n$%s", tree.string())
	}
}

// Exec computes expression for given arguments. Arguments count must be greater or equal to symbols (variables)
// count in Expression.
//
// Throws ErrExec.
//
// Example:
//     expr, _ := NewExpression(`(* 3 x0)`)
//     res1, _ := expr.Exec([]float64{2})
//     res2, _ := expr.Exec([]float64{10})
//     fmt.Println(res1, res2) // 6 30
func (e *Expression) Exec(x []float64) (res float64, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if e.number != nil {
		return e.number.exec(x)
	} else if e.symbol != nil {
		return e.symbol.exec(x)
	} else if e.operation != nil && e.tuple != nil {
		args, err := e.tuple.exec(x)
		if err != nil {
			return 0, err
		}
		return e.operation.exec(args)
	} else {
		return 0, fmt.Errorf("invalid expression: %+v", e)
	}
}

func (e *Expression) Copy() *Expression {
	expression, _ := newExpression(e.representation.copy())
	return expression
}

func (e *Expression) Equal(expression *Expression) bool {
	if e == nil || expression == nil {
		if (e != nil && expression == nil) || (e == nil && expression != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

	return e.representation.equal(expression.representation)
}

func (e *Expression) String() string {
	if e == nil {
		return "<nil>"
	}
	return e.representation.root
}

func (e *Expression) ShortString() string {
	if e == nil {
		return "<nil>"
	}
	return e.representation.root
}

func (e *Expression) PrettyString() string {
	if e == nil {
		return "<nil>"
	}
	return e.representation.string()
}
