package expression

import (
	"fmt"
	"nn/pkg/wraperr"
)

type Expression struct {
	*operation
	*tuple

	*symbol

	*number

	representation *sTree
}

func NewExpression(input string) (*Expression, error) {
	res, err := func(input string) (*Expression, error) {
		tree, err := splitRecursively(input)
		if err != nil {
			return nil, err
		}

		return newExpression(tree)
	}(input)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrParse, err)
	}

	return res, nil
}

func newExpression(tree *sTree) (*Expression, error) {
	asBytes := []byte(tree.root)
	res := &Expression{representation: tree}

	if tree.hasChildren() {
		token := tree.children[0].root
		operation, err := getOperation(token)
		if err != nil {
			return nil, fmt.Errorf("first token %q of %q is not an valid operation: %w", token, tree.root, err)
		}
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
	} else if numberPattern.Match(asBytes) {
		num, err := newNumber(tree.root)
		if err != nil {
			return nil, err
		}
		res.number = num
		return res, nil
	} else if symbolPattern.Match(asBytes) {
		sym, err := newSymbol(tree.root)
		if err != nil {
			return nil, err
		}
		res.symbol = sym
		return res, nil
	} else {
		//operation, err := getOperation(tree.root)
		//if err != nil {
		//	return nil, err
		//}
		//return &Expression{
		//	operation:      operation,
		//	representation: tree.root,
		//}, nil
		return nil, fmt.Errorf("error parsing tree to expression:\n$%s", tree.string())
	}
	// return nil, fmt.Errorf("error parsing tree to expression:\n$%s", tree.string())
}

func (e *Expression) Exec(x []float64) (float64, error) {
	res, err := func(x []float64) (float64, error) {
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
	}(x)

	if err != nil {
		return 0, wraperr.NewWrapErr(ErrExec, err)
	}

	return res, nil
}

func (e *Expression) Copy() *Expression {
	expression, _ := newExpression(e.representation.copy())
	return expression
}

func (e *Expression) Equal(expression *Expression) bool {
	return e.representation.equal(expression.representation)
}

func (e *Expression) String() string {
	return e.representation.root
}

func (e *Expression) ShortString() string {
	return e.representation.root
}

func (e *Expression) PrettyString() string {
	return e.representation.string()
}
