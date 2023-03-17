package operation

import (
	"fmt"
	"math"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

// generateMask return Matrix containing only values 0 and 1 distributed by given probability (count of 1 is defined
// by <probability>)
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

// NewDropout return dropout operation:
//     - each call will be generated mask of 0 and 1;
//     - shape of mask match shape of input;
//     - y = x * mask;
//     - dx = dy * mask.
//
// Throws ErrCreate error.
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

// NewSigmoidParam return operation:
//     y = f(x) = 1 / (1 + exp(-Ki * x));
//     dx = f(dy) = Ki * dy * (1 - dy),
//     where Ki is i'th coeff of <coeffs>. Coeffs count must match layer size.
//
// Throws ErrCreate error.
func NewSigmoidParam(coeffs *vector.Vector) (oper *ConstOperation, err error) {
	defer wraperr.WrapError(ErrCreate, &err)

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
}
