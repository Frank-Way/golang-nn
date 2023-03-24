package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/nn/loss"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"testing"
)

func TestMSELoss_Forward(t *testing.T) {
	tests := []struct {
		testutils.Base
		out      fabrics.MatrixParameters
		targets  fabrics.MatrixParameters
		expected float64
	}{
		{
			Base:     testutils.Base{Name: "3x1 outs, 3x1 targets", Err: nil},
			out:      fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{1, 2, 3}},
			targets:  fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{4, 5, 6}},
			expected: (((1-4)*(1-4) + (2-5)*(2-5) + (3-6)*(3-6)) / 2.0) / 3.0,
		},
		{
			Base:     testutils.Base{Name: "3x2 outs, 3x2 targets", Err: nil},
			out:      fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}},
			targets:  fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{7, 8, 9, 10, 11, 12}},
			expected: (((1-7)*(1-7) + (2-8)*(2-8) + (3-9)*(3-9) + (4-10)*(4-10) + (5-11)*(5-11) + (6-12)*(6-12)) / 2.0) / 3.0,
		},
		{
			Base:    testutils.Base{Name: "3x2 outs, 3x1 targets", Err: loss.ErrExec},
			out:     fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}},
			targets: fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{7, 8, 9}},
		},
		{
			Base:    testutils.Base{Name: "3x2 outs, 2x2 targets", Err: loss.ErrExec},
			out:     fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}},
			targets: fabrics.MatrixParameters{Rows: 2, Cols: 2, Values: []float64{7, 8, 9, 10}},
		},
	}
	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			mse := loss.NewMSELoss()
			out := fabrics.NewMatrix(t, test.out)
			targets := fabrics.NewMatrix(t, test.targets)

			actual, err := mse.Forward(targets, out)
			if test.Err == nil {
				require.NoError(t, err)
				require.Equal(t, test.expected, actual)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestMSELoss_Backward(t *testing.T) {
	tests := []struct {
		testutils.Base
		out      fabrics.MatrixParameters
		targets  fabrics.MatrixParameters
		expected fabrics.MatrixParameters
		forward  bool
	}{
		{
			Base:    testutils.Base{Name: "3x1 outs, 3x1 targets", Err: nil},
			out:     fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{1, 2, 3}},
			targets: fabrics.MatrixParameters{Rows: 3, Cols: 1, Values: []float64{4, 5, 6}},
			expected: fabrics.MatrixParameters{
				Rows:   3,
				Cols:   1,
				Values: []float64{(1 - 4) / 3.0, (2 - 5) / 3.0, (3 - 6) / 3.0},
			},
			forward: true,
		},
		{
			Base:    testutils.Base{Name: "3x2 outs, 3x2 targets", Err: nil},
			out:     fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}},
			targets: fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{7, 8, 9, 10, 11, 12}},
			expected: fabrics.MatrixParameters{
				Rows: 3,
				Cols: 2,
				Values: []float64{
					(1 - 7) / 3.0, (2 - 8) / 3.0, (3 - 9) / 3.0,
					(4 - 10) / 3.0, (5 - 11) / 3.0, (6 - 12) / 3.0,
				},
			},
			forward: true,
		},
		{
			Base:    testutils.Base{Name: "3x2 outs, 3x2 targets, no forward", Err: loss.ErrExec},
			out:     fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{1, 2, 3, 4, 5, 6}},
			targets: fabrics.MatrixParameters{Rows: 3, Cols: 2, Values: []float64{7, 8, 9, 10, 11, 12}},
		},
	}
	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			mse := loss.NewMSELoss()
			out := fabrics.NewMatrix(t, test.out)
			targets := fabrics.NewMatrix(t, test.targets)

			if test.forward {
				_, err := mse.Forward(targets, out)
				require.NoError(t, err)
			}
			backward, err := mse.Backward()
			if test.Err == nil {
				require.NoError(t, err)
				expected := fabrics.NewMatrix(t, test.expected)
				require.True(t, expected.EqualApprox(backward))
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}
