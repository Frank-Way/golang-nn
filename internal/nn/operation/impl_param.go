package operation

import (
	"fmt"
	"nn/pkg/mmath/matrix"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
)

// NewBiasOperation returns operation of adding bias
//
// Throws ErrCreate error
func NewBiasOperation(bias *vector.Vector) (*ParamOperation, error) {
	if bias == nil {
		return nil, wraperr.NewWrapErr(ErrCreate, fmt.Errorf("no bias provided: %v", bias))
	}
	biasAsMatrix, _ := matrix.NewMatrix([]*vector.Vector{bias.Copy()})
	return &ParamOperation{
		Operation: &Operation{name: "bias add"},
		p:         biasAsMatrix,
		output: func(x *matrix.Matrix, b *matrix.Matrix) (*matrix.Matrix, error) {
			return x.AddRowM(b)
		},
		gradient: func(dy *matrix.Matrix, b *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.Copy(), nil
		},
		gradParam: func(dy *matrix.Matrix, b *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.SumAxedM(matrix.Vertical)
		},
	}, nil
}

// NewWeightOperation returns operation of multiply weights
//
// Throws ErrCreate error
func NewWeightOperation(weight *matrix.Matrix) (*ParamOperation, error) {
	if weight == nil {
		return nil, wraperr.NewWrapErr(ErrCreate, fmt.Errorf("no weight provided: %v", weight))
	}
	return &ParamOperation{
		Operation: &Operation{name: "weight multiply"},
		p:         weight.Copy(),
		output: func(x *matrix.Matrix, w *matrix.Matrix) (*matrix.Matrix, error) {
			return x.MatMul(w)
		},
		gradient: func(dy *matrix.Matrix, w *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
			return dy.MatMul(w.T())
		},
		gradParam: func(dy *matrix.Matrix, w *matrix.Matrix, x *matrix.Matrix) (*matrix.Matrix, error) {
			return x.T().MatMul(dy)
		},
	}, nil
}
