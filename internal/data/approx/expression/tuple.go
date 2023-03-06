package expression

type tuple struct {
	expressions []*Expression
}

func newTuple(trees []*sTree) (*tuple, error) {
	expressions := make([]*Expression, len(trees))
	for i, tree := range trees {
		expression, err := newExpression(tree)
		if err != nil {
			return nil, err
		}
		expressions[i] = expression
	}

	return &tuple{expressions: expressions}, nil
}

func (t *tuple) exec(x []float64) ([]float64, error) {
	results := make([]float64, len(t.expressions))
	for i, expression := range t.expressions {
		result, err := expression.Exec(x)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}
