package operation

import (
	"fmt"
	"math"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

func generateMask(rows, cols int, probability percent.Percent) *matrix.Matrix {
	values := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		values[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			if probability.Hit() {
				values[i][j] = 1
			}
		}
	}

	mask, err := matrix.NewMatrixRaw(values)
	if err != nil {
		panic(err)
	}

	return mask
}

func NewDropout(keepProbability percent.Percent) (*ConstOperation, error) {
	params := []*matrix.Matrix{nil}
	return &ConstOperation{
		Operation: &Operation{name: "dropout"},
		p:         params,
		output: func(x *matrix.Matrix, p []*matrix.Matrix) (*matrix.Matrix, error) {
			mask := generateMask(x.Rows(), x.Cols(), keepProbability)
			p[0] = mask
			return x.Mul(p[0])
		},
		gradient: func(dy *matrix.Matrix, p []*matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.Mul(p[0])
		},
	}, nil
}

func NewSigmoidParam(coeffs *vector.Vector) (*ConstOperation, error) {
	res, err := func() (*ConstOperation, error) {
		if coeffs == nil {
			return nil, fmt.Errorf("no coeffs provided: %v", coeffs)
		}
		param, err := matrix.NewMatrix([]*vector.Vector{coeffs})
		if err != nil {
			return nil, err
		}

		params := []*matrix.Matrix{param}
		return &ConstOperation{
			Operation: &Operation{
				name: "parametrized sigmoid activation",
			},
			p: params,
			output: func(x *matrix.Matrix, p []*matrix.Matrix) (*matrix.Matrix, error) {
				multiplied, err := x.MulRowM(p[0])
				if err != nil {
					return nil, err
				}
				return multiplied.ApplyFunc(func(value float64) float64 {
					return 1 / (1 + math.Exp(-value))
				}), nil
			},
			gradient: func(dy *matrix.Matrix, p []*matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
				return dy.ApplyFunc(func(value float64) float64 {
					return value * (1 - value)
				}).MulRowM(p[0])
			},
		}, nil
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return res, nil
}
