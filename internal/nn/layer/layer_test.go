package layer

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"nn/internal/nn"
	"nn/internal/nn/operation"
	"nn/internal/nn/operation/operationtestutils"
	"nn/internal/testutils"
	"nn/internal/testutils/testfactories"
	"nn/pkg/mmath/matrix"
	"nn/pkg/percent"
	"testing"
)

func newLayer(t *testing.T, kind nn.Kind, args ...interface{}) ILayer {
	l, err := Create(kind, args...)
	require.NoError(t, err)
	return l
}

func TestLayer_Strings(t *testing.T) {
	optimizer := func(param, grad *matrix.Matrix) (*matrix.Matrix, error) {
		return param.Sub(grad)
	}
	testcases := []struct {
		l         ILayer
		in        *matrix.Matrix
		outGrad   *matrix.Matrix
		optimizer operation.Optimizer
	}{
		{
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
			),
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
			),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
			),
			in: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: newLayer(t, DenseLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
			),
		},
		{
			l: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in:        testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad:   testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 10}),
			optimizer: optimizer,
		},
		{
			l: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in:      testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
			outGrad: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 10}),
		},
		{
			l: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
				percent.Percent50,
			),
			in: testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 1, Cols: 2}),
		},
		{
			l: newLayer(t, DenseDropLayer,
				testfactories.NewMatrix(t, testfactories.MatrixParameters{Rows: 2, Cols: 10}),
				testfactories.NewVector(t, testfactories.VectorParameters{Size: 10}),
				operationtestutils.NewOperation(t, operation.SigmoidParamActivation, testfactories.NewVector(t, testfactories.VectorParameters{Size: 10})),
				percent.Percent50,
			),
		},
	}

	testutils.SetupLogger()
	for i, tc := range testcases {
		t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
			if tc.in != nil {
				_, err := tc.l.Forward(tc.in)
				require.NoError(t, err)

				if tc.outGrad != nil {
					_, err = tc.l.Backward(tc.outGrad)
					require.NoError(t, err)

					if tc.optimizer != nil {
						err = tc.l.ApplyOptim(tc.optimizer)
						require.NoError(t, err)
					}
				}
			}

			t.Log("ShortString()\n" + tc.l.ShortString())
			t.Log("String()\n" + tc.l.String())
			t.Log("PrettyString()\n" + tc.l.PrettyString())
		})
	}
}
