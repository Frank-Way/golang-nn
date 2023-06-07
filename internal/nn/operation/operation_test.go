package operation

import (
	"github.com/stretchr/testify/require"
	"math"
	"nn/internal/nn"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/mmath/matrix"
	"nn/pkg/percent"
	"testing"
)

func newOperation(t *testing.T, kind nn.Kind, args ...interface{}) IOperation {
	o, err := Create(kind, args...)
	require.NoError(t, err)
	return o
}

func TestOperation_Copy(t *testing.T) {
	tests := []struct {
		testutils.Base
		oper    IOperation
		in      *matrix.Matrix
		outGrad *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "weight"},
			oper: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
		},
		{
			Base: testutils.Base{Name: "weight after forward"},
			oper: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "weight after backward"},
			oper:    newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "bias"},
			oper: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
		},
		{
			Base: testutils.Base{Name: "bias after forward"},
			oper: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base:    testutils.Base{Name: "bias after backward"},
			oper:    newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "linear"},
			oper: newOperation(t, LinearActivation),
		},
		{
			Base: testutils.Base{Name: "linear after forward"},
			oper: newOperation(t, LinearActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "linear after backward"},
			oper:    newOperation(t, LinearActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "sigmoid"},
			oper: newOperation(t, SigmoidActivation),
		},
		{
			Base: testutils.Base{Name: "sigmoid after forward"},
			oper: newOperation(t, SigmoidActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "sigmoid after backward"},
			oper:    newOperation(t, SigmoidActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "tanh"},
			oper: newOperation(t, TanhActivation),
		},
		{
			Base: testutils.Base{Name: "tanh after forward"},
			oper: newOperation(t, TanhActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "tanh after backward"},
			oper:    newOperation(t, TanhActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "dropout"},
			oper: newOperation(t, Dropout, percent.Percent10),
		},
		{
			Base: testutils.Base{Name: "dropout after forward"},
			oper: newOperation(t, Dropout, percent.Percent10),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "dropout after backward"},
			oper:    newOperation(t, Dropout, percent.Percent10),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			if test.in != nil {
				_, err := test.oper.Forward(test.in)
				require.NoError(t, err)

				if test.outGrad != nil {
					_, err := test.oper.Backward(test.outGrad)
					require.NoError(t, err)
				}
			}
			cp := test.oper.Copy()
			require.True(t, cp != test.oper)
			require.True(t, cp.Equal(test.oper))
			require.True(t, test.oper.Equal(cp))
			require.True(t, cp.EqualApprox(test.oper))
			require.True(t, test.oper.EqualApprox(cp))
		})
	}
}

func TestOperation_Strings(t *testing.T) {
	testutils.SetupLogger()
	tests := []struct {
		testutils.Base
		oper    IOperation
		in      *matrix.Matrix
		outGrad *matrix.Matrix
	}{
		{
			Base: testutils.Base{Name: "weight"},
			oper: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
		},
		{
			Base: testutils.Base{Name: "weight after forward"},
			oper: newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "weight after backward"},
			oper:    newOperation(t, WeightMultiply, testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 3})),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "bias"},
			oper: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
		},
		{
			Base: testutils.Base{Name: "bias after forward"},
			oper: newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base:    testutils.Base{Name: "bias after backward"},
			oper:    newOperation(t, BiasAdd, testfactories.NewVector(t, testfactories.VectorParameters{Size: 3})),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 3}),
		},
		{
			Base: testutils.Base{Name: "linear"},
			oper: newOperation(t, LinearActivation),
		},
		{
			Base: testutils.Base{Name: "linear after forward"},
			oper: newOperation(t, LinearActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "linear after backward"},
			oper:    newOperation(t, LinearActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "sigmoid"},
			oper: newOperation(t, SigmoidActivation),
		},
		{
			Base: testutils.Base{Name: "sigmoid after forward"},
			oper: newOperation(t, SigmoidActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "sigmoid after backward"},
			oper:    newOperation(t, SigmoidActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "tanh"},
			oper: newOperation(t, TanhActivation),
		},
		{
			Base: testutils.Base{Name: "tanh after forward"},
			oper: newOperation(t, TanhActivation),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "tanh after backward"},
			oper:    newOperation(t, TanhActivation),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base: testutils.Base{Name: "dropout"},
			oper: newOperation(t, Dropout, percent.Percent50),
		},
		{
			Base: testutils.Base{Name: "dropout after forward"},
			oper: newOperation(t, Dropout, percent.Percent50),
			in:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
		{
			Base:    testutils.Base{Name: "dropout after backward"},
			oper:    newOperation(t, Dropout, percent.Percent50),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 5, Cols: 2}),
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			if test.in != nil {
				_, err := test.oper.Forward(test.in)
				require.NoError(t, err)

				if test.outGrad != nil {
					_, err := test.oper.Backward(test.outGrad)
					require.NoError(t, err)
				}
			}
			t.Log("ShortString():\n" + test.oper.ShortString())
			t.Log("String():\n" + test.oper.String())
			t.Log("PrettyString():\n" + test.oper.PrettyString())
		})
	}
}

func Test_OperationPipeline(t *testing.T) {
	testutils.SetupLogger()
	var err error
	wweight := testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}})
	weight := newOperation(t, WeightMultiply,
		testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}})).(*ParamOperation)
	bbias := testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2, Values: []float64{1, 2}})
	bias := newOperation(t, BiasAdd,
		testfactories.NewVector(t, testfactories.VectorParameters{Values: []float64{1, 2}})).(*ParamOperation)
	act := newOperation(t, SigmoidActivation).(*Operation)

	in := testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{1, 2, 3}})
	outExpected, err := in.MatMul(wweight)
	require.NoError(t, err)
	outExpected, err = outExpected.AddRowM(bbias)
	require.NoError(t, err)
	outExpected = outExpected.ApplyFunc(func(a float64) float64 { return 1 / (1 + math.Exp(-a)) })

	outGrad := testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}})
	inGradExpected := outGrad.ApplyFunc(func(a float64) float64 { return a - a*a })
	biasGradExpected, err := inGradExpected.SumAxedM(matrix.Vertical)
	require.NoError(t, err)
	inGradExpected = inGradExpected.Copy()
	weightGradExpected, err := in.T().MatMul(inGradExpected)
	require.NoError(t, err)
	inGradExpected, err = inGradExpected.MatMul(wweight.T())
	require.NoError(t, err)

	var optim Optimizer = func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}

	weightExpected, err := optim(wweight, weightGradExpected)
	require.NoError(t, err)
	biasExpected, err := optim(bbias, biasGradExpected)

	operations := []IOperation{weight, bias, act}

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
		case *ParamOperation:
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
