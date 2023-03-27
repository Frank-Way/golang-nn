package test

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/nn/operation"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/mmath/matrix"
	"nn/pkg/percent"
	"testing"
)

func TestOperation_Copy(t *testing.T) {
	tests := []struct {
		testutils.Base
		oper    interface{}
		in      *matrix.Matrix
		outGrad *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "weight"},
			oper: fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
		},
		{
			Base: testutils.Base{Name: "weight after forward"},
			oper: fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "weight after backward"},
			oper:    fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "bias"},
			oper: fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
		},
		{
			Base: testutils.Base{Name: "bias after forward"},
			oper: fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base:    testutils.Base{Name: "bias after backward"},
			oper:    fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "linear"},
			oper: fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "linear after forward"},
			oper: fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "linear after backward"},
			oper:    fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "sigmoid"},
			oper: fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "sigmoid after forward"},
			oper: fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "sigmoid after backward"},
			oper:    fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "tanh"},
			oper: fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "tanh after forward"},
			oper: fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "tanh after backward"},
			oper:    fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "dropout"},
			oper: fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
		},
		{
			Base: testutils.Base{Name: "dropout after forward"},
			oper: fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "dropout after backward"},
			oper:    fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			switch oper := test.oper.(type) {
			case *operation.Operation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				cp := oper.Copy()
				require.True(t, cp != oper)
				require.True(t, cp.Equal(oper))
				require.True(t, oper.Equal(cp))
				require.True(t, cp.EqualApprox(oper))
				require.True(t, oper.EqualApprox(cp))
			case *operation.ParamOperation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				cp := oper.Copy()
				require.True(t, cp != oper)
				require.True(t, cp.Equal(oper))
				require.True(t, oper.Equal(cp))
				require.True(t, cp.EqualApprox(oper))
				require.True(t, oper.EqualApprox(cp))
			case *operation.ConstOperation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				cp := oper.Copy()
				require.True(t, cp != oper)
				require.True(t, cp.Equal(oper))
				require.True(t, oper.Equal(cp))
				require.True(t, cp.EqualApprox(oper))
				require.True(t, oper.EqualApprox(cp))
			}
		})
	}
}

func TestOperation_Strings(t *testing.T) {
	tests := []struct {
		testutils.Base
		oper    interface{}
		in      *matrix.Matrix
		outGrad *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "weight"},
			oper: fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
		},
		{
			Base: testutils.Base{Name: "weight after forward"},
			oper: fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "weight after backward"},
			oper:    fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 2, Cols: 3}}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "bias"},
			oper: fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
		},
		{
			Base: testutils.Base{Name: "bias after forward"},
			oper: fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base:    testutils.Base{Name: "bias after backward"},
			oper:    fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Size: 3}}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "linear"},
			oper: fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "linear after forward"},
			oper: fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "linear after backward"},
			oper:    fabrics.NewActivation(t, fabrics.LinearAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "sigmoid"},
			oper: fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "sigmoid after forward"},
			oper: fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "sigmoid after backward"},
			oper:    fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "tanh"},
			oper: fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
		},
		{
			Base: testutils.Base{Name: "tanh after forward"},
			oper: fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "tanh after backward"},
			oper:    fabrics.NewActivation(t, fabrics.TanhAct, fabrics.ActivationParameters{}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "dropout"},
			oper: fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
		},
		{
			Base: testutils.Base{Name: "dropout after forward"},
			oper: fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
			in:   fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "dropout after backward"},
			oper:    fabrics.NewDropout(t, fabrics.DropoutParameters{Percent: percent.Percent10}),
			in:      fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 5, Cols: 2}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			switch oper := test.oper.(type) {
			case *operation.Operation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				t.Log("ShortString():\n" + oper.ShortString())
				t.Log("FormatObject():\n" + oper.String())
				t.Log("PrettyStringField():\n" + oper.PrettyString())
			case *operation.ParamOperation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				t.Log("ShortString():\n" + oper.ShortString())
				t.Log("FormatObject():\n" + oper.String())
				t.Log("PrettyStringField():\n" + oper.PrettyString())
			case *operation.ConstOperation:
				if test.in != nil {
					_, err := oper.Forward(test.in)
					require.NoError(t, err)

					if test.outGrad != nil {
						_, err := oper.Backward(test.outGrad)
						require.NoError(t, err)
					}
				}
				t.Log("ShortString():\n" + oper.ShortString())
				t.Log("FormatObject():\n" + oper.String())
				t.Log("PrettyStringField():\n" + oper.PrettyString())
			}
		})
	}
}

func Test_OperationPipeline(t *testing.T) {
	var err error
	wweight := fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}})
	weight := fabrics.NewWeight(t, fabrics.WeightParameters{MatrixParameters: fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}}}).(*operation.ParamOperation)
	bbias := fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}})
	bias := fabrics.NewBias(t, fabrics.BiasParameters{VectorParameters: fabrics.VectorParameters{Values: []float64{1, 2}}}).(*operation.ParamOperation)
	act := fabrics.NewActivation(t, fabrics.SigmoidAct, fabrics.ActivationParameters{}).(*operation.Operation)

	in := fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{1, 2, 3}})
	outExpected, err := in.MatMul(wweight)
	require.NoError(t, err)
	outExpected, err = outExpected.AddRowM(bbias)
	require.NoError(t, err)
	outExpected = outExpected.ApplyFunc(func(a float64) float64 { return 1 / (1 + math.Exp(-a)) })

	outGrad := fabrics.NewMatrix(t, fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}})
	inGradExpected := outGrad.ApplyFunc(func(a float64) float64 { return a - a*a })
	biasGradExpected, err := inGradExpected.SumAxedM(matrix.Vertical)
	require.NoError(t, err)
	inGradExpected = inGradExpected.Copy()
	weightGradExpected, err := in.T().MatMul(inGradExpected)
	require.NoError(t, err)
	inGradExpected, err = inGradExpected.MatMul(wweight.T())
	require.NoError(t, err)

	var optim operation.Optimizer = func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}

	weightExpected, err := optim(wweight, weightGradExpected)
	require.NoError(t, err)
	biasExpected, err := optim(bbias, biasGradExpected)

	operations := []operation.IOperation{weight, bias, act}

	x := in
	for i := range operations {
		x, err = operations[i].Forward(x)
		require.NoError(t, err)
	}
	require.True(t, x.EqualApprox(outExpected))

	dy := outGrad
	for i := range operations {
		dy, err = operations[len(operations)-i-1].Backward(dy)
		require.NoError(t, err)
	}
	require.True(t, dy.EqualApprox(inGradExpected))

	for i := range operations {
		switch oper := operations[i].(type) {
		case *operation.ParamOperation:
			err = oper.ApplyOptim(optim)
			require.NoError(t, err)
		}
	}

	require.True(t, weight.Parameter().EqualApprox(weightExpected))
	require.True(t, bias.Parameter().EqualApprox(biasExpected))
	t.Log("weight.PrettyString()\n" + weight.PrettyString())
	t.Log("bias.PrettyString()\n" + bias.PrettyString())
	t.Log("act.PrettyString()\n" + act.PrettyString())
}
