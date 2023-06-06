package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/optim"
	"nn/internal/testutils"
	"nn/pkg/mmath/matrix"
	"testing"
)

func TestSGD(t *testing.T) {
	testutils.SetupLogger()
	matrixFactory := func(rows int, cols int, value float64) *matrix.Matrix {
		m, err := matrix.NewMatrixOf(rows, cols, value)
		require.NoError(t, err)
		return m
	}
	defaultSizeMatrixFactory := func(value float64) *matrix.Matrix {
		return matrixFactory(2, 2, value)
	}
	matrixSubtractor := func(m1 *matrix.Matrix, m2 *matrix.Matrix) *matrix.Matrix {
		m, err := m1.Sub(m2)
		require.NoError(t, err)
		return m
	}
	defaultLearnRate := 0.05
	testcases := []struct {
		testutils.Base
		parameters *optim.SGDParameters
		param      *matrix.Matrix
		grad       *matrix.Matrix
		expected   *matrix.Matrix
	}{
		{
			Base:     testutils.Base{Name: "nil parameters"},
			param:    defaultSizeMatrixFactory(2),
			grad:     defaultSizeMatrixFactory(1),
			expected: matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base:       testutils.Base{Name: "empty parameters"},
			parameters: &optim.SGDParameters{},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base:       testutils.Base{Name: "incorrect inputs", Err: optim.ErrExec},
			parameters: &optim.SGDParameters{},
			param:      defaultSizeMatrixFactory(2),
			grad:       matrixFactory(1, 1, 1),
		},
		{
			Base:       testutils.Base{Name: "only learn rate"},
			parameters: &optim.SGDParameters{LearnRate: 2},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(2)),
		},
		{
			Base:       testutils.Base{Name: "only stop learn rate"},
			parameters: &optim.SGDParameters{StopLearnRate: 0.001},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base:       testutils.Base{Name: "only stop learn rate bigger default learn rate"},
			parameters: &optim.SGDParameters{StopLearnRate: 2},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base:       testutils.Base{Name: "only epochs count"},
			parameters: &optim.SGDParameters{EpochsCount: 500},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base:       testutils.Base{Name: "default exponential"},
			parameters: &optim.SGDParameters{DecrementType: optim.ExponentialDecrement},
			param:      defaultSizeMatrixFactory(2),
			grad:       defaultSizeMatrixFactory(1),
			expected:   matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(defaultLearnRate)),
		},
		{
			Base: testutils.Base{Name: "all exponential"},
			parameters: &optim.SGDParameters{
				LearnRate:     1,
				StopLearnRate: 0.00001,
				EpochsCount:   5,
				DecrementType: optim.ExponentialDecrement,
			},
			param:    defaultSizeMatrixFactory(2),
			grad:     defaultSizeMatrixFactory(1),
			expected: matrixSubtractor(defaultSizeMatrixFactory(2), defaultSizeMatrixFactory(1).MulNum(1)),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			sgd := optim.NewSGD(tc.parameters)
			newParam, err := sgd(tc.param, tc.grad)
			if tc.Err == nil {
				require.NoError(t, err)
				require.True(t, tc.expected.EqualApprox(newParam))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tc.Err)
			}
		})
	}
}
