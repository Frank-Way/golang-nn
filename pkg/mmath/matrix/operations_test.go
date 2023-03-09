package matrix

import (
	"github.com/stretchr/testify/require"
	"math"
	"math/rand"
	"nn/pkg/mmath/vector"
	"testing"
)

func TestMatrix_T(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		expected []float64
	}{
		{name: "1x1 matrix", in: []float64{1}, rows: 1, cols: 1, expected: []float64{1}},
		{name: "1x2 matrix", in: []float64{1, 2}, rows: 1, cols: 2, expected: []float64{1, 2}},
		{name: "1x3 matrix", in: []float64{1, 2, 3}, rows: 1, cols: 3, expected: []float64{1, 2, 3}},
		{name: "2x1 matrix", in: []float64{1, 2}, rows: 2, cols: 1, expected: []float64{1, 2}},
		{name: "2x2 matrix", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expected: []float64{1, 3, 2, 4}},
		{name: "2x3 matrix", in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3, expected: []float64{1, 4, 2, 5, 3, 6}},
		{name: "3x1 matrix", in: []float64{1, 2, 3}, rows: 3, cols: 1, expected: []float64{1, 2, 3}},
		{name: "3x2 matrix", in: []float64{1, 2, 3, 4, 5, 6}, rows: 3, cols: 2, expected: []float64{1, 3, 5, 2, 4, 6}},
		{name: "3x3 matrix", in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, rows: 3, cols: 3, expected: []float64{1, 4, 7, 2, 5, 8, 3, 6, 9}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			transposed := matrix.T()
			require.NotNil(t, transposed)

			flat := transposed.RawFlat()
			require.Equal(t, len(test.expected), len(flat))
			require.Equal(t, test.cols, transposed.rows)
			require.Equal(t, test.rows, transposed.cols)
			for i, value := range test.expected {
				require.Equal(t, value, flat[i])
			}
		})
	}
}

func TestMatrix_MatMul(t *testing.T) {
	tests := []struct {
		testBase
		a matrixInput
		b matrixInput
	}{
		{
			testBase: testBase{name: "2x3 matmul 3x4", expected: []float64{38, 44, 50, 56, 83, 98, 113, 128}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 3, cols: 4},
		},
		{
			testBase: testBase{name: "3x4 matmul 2x3, error", err: ErrOperationExec},
			b:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 3, cols: 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.MatMul(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: test.a.rows, cols: test.b.cols}, test.err, err, matrix)
		})
	}
}

func randomMatrix(rows, cols int) *Matrix {
	values := make([]float64, rows*cols)
	for i := 0; i < rows*cols; i++ {
		values[i] = rand.Float64()
	}

	matrix, _ := NewMatrixRawFlat(rows, cols, values)
	return matrix
}

func BenchmarkMatrix_MatMul(b *testing.B) {
	tests := []struct {
		name  string
		rowsA int
		colsA int
		rowsB int
		colsB int
	}{
		{name: "2x3 matmul 3x4", rowsA: 2, colsA: 3, rowsB: 3, colsB: 4},
		{name: "5x6 matmul 6x7", rowsA: 5, colsA: 6, rowsB: 6, colsB: 7},
		{name: "8x9 matmul 9x10", rowsA: 8, colsA: 9, rowsB: 9, colsB: 10},
		{name: "20x30 matmul 30x40", rowsA: 20, colsA: 30, rowsB: 30, colsB: 40},
		{name: "64x2 matmul 2x16", rowsA: 64, colsA: 2, rowsB: 2, colsB: 16},
		{name: "200x300 matmul 300x400", rowsA: 200, colsA: 300, rowsB: 300, colsB: 400},
	}

	for _, test := range tests {
		one := randomMatrix(test.rowsA, test.colsA)
		two := randomMatrix(test.rowsB, test.colsB)

		b.Run("single goroutine # "+test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				one.matMulImplSingle(two)
			}
		})
		b.Run("multiple goroutines # "+test.name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				one.matMulImplMulti(two)
			}
		})
	}
}

type matrixInput struct {
	in   []float64
	rows int
	cols int
}

func newMatrix(t *testing.T, in matrixInput) *Matrix {
	matrix, err := NewMatrixRawFlat(in.rows, in.cols, in.in)
	require.NoError(t, err)

	return matrix
}

type testBase struct {
	name     string
	expected []float64
	err      error
}

type twoMatrixTest struct {
	testBase
	a matrixInput
	b matrixInput
}

type matrixVectorTest struct {
	testBase
	a matrixInput
	b []float64
}

type matrixNumberTest struct {
	testBase
	a matrixInput
	b float64
}

func makeAssertions(t *testing.T, expected matrixInput, expectedErr error, err error, matrix *Matrix) {
	if expectedErr == nil {
		require.NoError(t, err)
		require.Equal(t, expected.rows, matrix.rows)
		require.Equal(t, expected.cols, matrix.cols)
		flat := matrix.RawFlat()
		require.Equal(t, len(expected.in), len(flat))
		for i, value := range expected.in {
			require.Equal(t, value, flat[i])
		}
	} else {
		require.Error(t, err)
		require.ErrorIs(t, err, expectedErr)
	}
}

func TestMatrix_Add(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 + 1x1", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 + 2x2", expected: []float64{5, 7, 9, 11}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
		{
			testBase: testBase{name: "1x1 + 2x2, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.Add(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_Sub(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 - 1x1", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 - 2x2", expected: []float64{-3, -3, -3, -3}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
		{
			testBase: testBase{name: "1x1 - 2x2, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.Sub(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_Mul(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 * 1x1", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 * 2x2", expected: []float64{4, 10, 18, 28}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
		{
			testBase: testBase{name: "1x1 * 2x2, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.Mul(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_Div(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 / 1x1", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 / 2x2", expected: []float64{1.0 / 4.0, 2.0 / 5.0, 3.0 / 6.0, 4.0 / 7.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
		{
			testBase: testBase{name: "1x1 / 2x2, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{4, 5, 6, 7}, rows: 2, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.Div(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func newVector(t *testing.T, in []float64) *vector.Vector {
	vec, err := vector.NewVector(in)
	require.NoError(t, err)

	return vec
}

func TestMatrix_AddRow(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 + 1 row", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 + 2 row", expected: []float64{6, 8, 8, 10}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "2x1 + 2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.AddRow(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_SubRow(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 - 1 row", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 - 2 row", expected: []float64{-4, -4, -2, -2}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "2x1 - 2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.SubRow(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_MulRow(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 * 1 row", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 * 2 row", expected: []float64{5, 12, 15, 24}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "2x1 * 2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.MulRow(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_DivRow(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 / 1 row", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 / 2 row", expected: []float64{1.0 / 5.0, 2.0 / 6.0, 3.0 / 5.0, 4.0 / 6.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "2x1 / 2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.DivRow(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_AddCol(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 + 1 col", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 + 2 col", expected: []float64{6, 7, 9, 10}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "1x2 + 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.AddCol(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_SubCol(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 - 1 col", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 - 2 col", expected: []float64{-4, -3, -3, -2}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "1x2 - 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.SubCol(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_MulCol(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 * 1 col", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 * 2 col", expected: []float64{5, 10, 18, 24}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "1x2 * 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.MulCol(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_DivCol(t *testing.T) {
	tests := []matrixVectorTest{
		{
			testBase: testBase{name: "1x1 / 1 col", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        []float64{2},
		},
		{
			testBase: testBase{name: "2x2 / 2 col", expected: []float64{1.0 / 5.0, 2.0 / 5.0, 3.0 / 6.0, 4.0 / 6.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        []float64{5, 6},
		},
		{
			testBase: testBase{name: "1x2 / 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        []float64{3, 4},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newVector(t, test.b)

			matrix, err := a.DivCol(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_AddRowM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 + 1x1 row", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 + 1x2 row", expected: []float64{6, 8, 8, 10}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 + 1x2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 + 2x1 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.AddRowM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_SubRowM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 - 1x1 row", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 - 1x2 row", expected: []float64{-4, -4, -2, -2}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 - 1x2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 - 2x1 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.SubRowM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_MulRowM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 * 1x1 row", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 * 1x2 row", expected: []float64{5, 12, 15, 24}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 * 1x2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 * 2x1 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.MulRowM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_DivRowM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 / 1x1 row", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 / 1x2 row", expected: []float64{1.0 / 5.0, 2.0 / 6.0, 3.0 / 5.0, 4.0 / 6.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 / 1x2 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
		{
			testBase: testBase{name: "2x1 / 2x1 row, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.DivRowM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_AddColM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 + 1x1 col", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 + 2x1 col", expected: []float64{6, 7, 9, 10}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "1x2 + 2x1 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "2x1 + 1x2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.AddColM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_SubColM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 - 1 col", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 - 2 col", expected: []float64{-4, -3, -3, -2}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "1x2 - 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "2x1 - 1x2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.SubColM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_MulColM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 * 1 col", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 * 2 col", expected: []float64{5, 10, 18, 24}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "1x2 * 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "2x1 * 1x2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.MulColM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_DivColM(t *testing.T) {
	tests := []twoMatrixTest{
		{
			testBase: testBase{name: "1x1 / 1 col", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        matrixInput{in: []float64{2}, rows: 1, cols: 1},
		},
		{
			testBase: testBase{name: "2x2 / 2 col", expected: []float64{1.0 / 5.0, 2.0 / 5.0, 3.0 / 6.0, 4.0 / 6.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "1x2 / 2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			b:        matrixInput{in: []float64{3, 4}, rows: 2, cols: 1},
		},
		{
			testBase: testBase{name: "2x1 / 1x2 col, error", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			b:        matrixInput{in: []float64{3, 4}, rows: 1, cols: 2},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			matrix, err := a.DivColM(b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, err, matrix)
		})
	}
}

func TestMatrix_AddNum(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "1x1 + scalar", expected: []float64{3}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        2,
		},
		{
			testBase: testBase{name: "2x2 + scalar", expected: []float64{6, 7, 8, 9}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			matrix := a.AddNum(test.b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, nil, matrix)
		})
	}
}

func TestMatrix_SubNum(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "1x1 - scalar", expected: []float64{-1}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        2,
		},
		{
			testBase: testBase{name: "2x2 - scalar", expected: []float64{-4, -3, -2, -1}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			matrix := a.SubNum(test.b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, nil, matrix)
		})
	}
}

func TestMatrix_MulNum(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "1x1 * scalar", expected: []float64{2}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        2,
		},
		{
			testBase: testBase{name: "2x2 * scalar", expected: []float64{5, 10, 15, 20}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			matrix := a.MulNum(test.b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, nil, matrix)
		})
	}
}

func TestMatrix_DivNum(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "1x1 / scalar", expected: []float64{1.0 / 2.0}},
			a:        matrixInput{in: []float64{1}, rows: 1, cols: 1},
			b:        2,
		},
		{
			testBase: testBase{name: "2x2 / scalar", expected: []float64{1.0 / 5.0, 2.0 / 5.0, 3.0 / 5.0, 4.0 / 5.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			matrix := a.DivNum(test.b)
			makeAssertions(t, matrixInput{in: test.expected, rows: a.rows, cols: a.cols}, test.err, nil, matrix)
		})
	}
}

func TestMatrix_Sum(t *testing.T) {
	matrix, err := NewMatrixRawFlat(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	require.NoError(t, err)

	sum := matrix.Sum()
	require.Equal(t, float64(1+2+3+4+5+6+7+8+9), sum)
}

func TestMatrix_SumAxed(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "2x3, sum horizontal", expected: []float64{6, 15}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        0,
		},
		{
			testBase: testBase{name: "2x3, sum vertical", expected: []float64{5, 7, 9}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        1,
		},
		{
			testBase: testBase{name: "2x3, unknown axis, err", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			vec, err := a.SumAxed(Axis(uint8(test.b)))
			matrix, _ := NewMatrix([]*vector.Vector{vec})
			if test.b == float64(0) {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.rows}, test.err, err, matrix)
			} else {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.cols}, test.err, err, matrix)
			}
		})
	}
}

func TestMatrix_Max(t *testing.T) {
	matrix, err := NewMatrixRawFlat(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	require.NoError(t, err)

	max := matrix.Max()
	require.Equal(t, 9.0, max)
}

func TestMatrix_MaxAxed(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "2x3, max horizontal", expected: []float64{3, 6}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        0,
		},
		{
			testBase: testBase{name: "2x3, max vertical", expected: []float64{4, 5, 6}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        1,
		},
		{
			testBase: testBase{name: "2x3, unknown axis, err", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			vec, err := a.MaxAxed(Axis(uint8(test.b)))
			matrix, _ := NewMatrix([]*vector.Vector{vec})
			if test.b == float64(0) {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.rows}, test.err, err, matrix)
			} else {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.cols}, test.err, err, matrix)
			}
		})
	}
}

func TestMatrix_Min(t *testing.T) {
	matrix, err := NewMatrixRawFlat(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	require.NoError(t, err)

	min := matrix.Min()
	require.Equal(t, 1.0, min)
}

func TestMatrix_MinAxed(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "2x3, min horizontal", expected: []float64{1, 4}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        0,
		},
		{
			testBase: testBase{name: "2x3, min vertical", expected: []float64{1, 2, 3}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        1,
		},
		{
			testBase: testBase{name: "2x3, unknown axis, err", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			vec, err := a.MinAxed(Axis(uint8(test.b)))
			matrix, _ := NewMatrix([]*vector.Vector{vec})
			if test.b == float64(0) {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.rows}, test.err, err, matrix)
			} else {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.cols}, test.err, err, matrix)
			}
		})
	}
}

func TestMatrix_Avg(t *testing.T) {
	matrix, err := NewMatrixRawFlat(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	require.NoError(t, err)

	avg := matrix.Avg()
	require.Equal(t, float64(1+2+3+4+5+6+7+8+9)/9.0, avg)
}

func TestMatrix_AvgAxed(t *testing.T) {
	tests := []matrixNumberTest{
		{
			testBase: testBase{name: "2x3, avg horizontal", expected: []float64{(1 + 2 + 3) / 3.0, (4 + 5 + 6) / 3.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        0,
		},
		{
			testBase: testBase{name: "2x3, avg vertical", expected: []float64{(1 + 4) / 2.0, (2 + 5) / 2.0, (3 + 6) / 2.0}},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        1,
		},
		{
			testBase: testBase{name: "2x3, unknown axis, err", err: ErrOperationExec},
			a:        matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3},
			b:        5,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)

			vec, err := a.AvgAxed(Axis(uint8(test.b)))
			matrix, _ := NewMatrix([]*vector.Vector{vec})
			if test.b == float64(0) {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.rows}, test.err, err, matrix)
			} else {
				makeAssertions(t, matrixInput{in: test.expected, rows: 1, cols: a.cols}, test.err, err, matrix)
			}
		})
	}
}

func TestMatrix_Abs(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 3, []float64{1, -2, 3, -4, 5, 6})
	require.NoError(t, err)

	res := matrix.Abs()
	flat := res.RawFlat()
	require.Equal(t, 6, len(flat))
	for i, value := range []float64{1, 2, 3, 4, 5, 6} {
		require.Equal(t, value, flat[i])
	}
}

func TestMatrix_Exp(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 3, []float64{1, 2, 3, 4, 5, 6})
	require.NoError(t, err)

	res := matrix.Exp()
	flat := res.RawFlat()
	require.Equal(t, 6, len(flat))
	f := math.Exp
	for i, value := range []float64{1, 2, 3, 4, 5, 6} {
		require.Equal(t, f(value), flat[i])
	}
}

func TestMatrix_Sqrt(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 3, []float64{1, 2, 3, 4, 5, 6})
	require.NoError(t, err)

	res := matrix.Sqrt()
	flat := res.RawFlat()
	require.Equal(t, 6, len(flat))
	f := math.Sqrt
	for i, value := range []float64{1, 2, 3, 4, 5, 6} {
		require.Equal(t, f(value), flat[i])
	}
}

func TestMatrix_Sqr(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 3, []float64{1, 2, 3, 4, 5, 6})
	require.NoError(t, err)

	res := matrix.Sqr()
	flat := res.RawFlat()
	require.Equal(t, 6, len(flat))
	f := func(value float64) float64 {
		return math.Pow(value, 2)
	}
	for i, value := range []float64{1, 2, 3, 4, 5, 6} {
		require.Equal(t, f(value), flat[i])
	}
}

func TestMatrix_Pow(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 3, []float64{1, 2, 3, 4, 5, 6})
	require.NoError(t, err)

	res := matrix.Pow(3.5)
	flat := res.RawFlat()
	require.Equal(t, 6, len(flat))
	f := math.Pow
	for i, value := range []float64{1, 2, 3, 4, 5, 6} {
		require.Equal(t, f(value, 3.5), flat[i])
	}
}

func TestMatrix_SubMatrix(t *testing.T) {
	tests := []struct {
		testBase
		a            matrixInput
		rowsStart    int
		rowsStop     int
		rowsStep     int
		colsStart    int
		colsStop     int
		colsStep     int
		expectedRows int
		expectedCols int
	}{
		{
			testBase:     testBase{name: "4x3[0:4:1][0:3:1]", expected: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
			a:            matrixInput{in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 4, cols: 3},
			rowsStart:    0,
			rowsStop:     4,
			rowsStep:     1,
			colsStart:    0,
			colsStop:     3,
			colsStep:     1,
			expectedRows: 4,
			expectedCols: 3,
		},
		{
			testBase:     testBase{name: "4x3[1:4:2][0:3:2]", expected: []float64{4, 6, 10, 12}},
			a:            matrixInput{in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 4, cols: 3},
			rowsStart:    1,
			rowsStop:     4,
			rowsStep:     2,
			colsStart:    0,
			colsStop:     3,
			colsStep:     2,
			expectedRows: 2,
			expectedCols: 2,
		},
		{
			testBase:  testBase{name: "{{1,2,3},{4,5,6},{7,8,9},{10,11,12}}[-1:-5:0][-2:-3:-1]", err: ErrOperationExec},
			a:         matrixInput{in: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 4, cols: 3},
			rowsStart: -1,
			rowsStop:  -5,
			rowsStep:  0,
			colsStart: -1,
			colsStop:  -5,
			colsStep:  -3,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix := newMatrix(t, test.a)

			subMatrix, err := matrix.SubMatrix(test.rowsStart, test.rowsStop, test.rowsStep, test.colsStart, test.colsStop, test.colsStep)

			makeAssertions(t, matrixInput{in: test.expected, rows: test.expectedRows, cols: test.expectedCols}, test.err, err, subMatrix)
		})
	}
}

func TestMatrix_HStack(t *testing.T) {
	tests := []struct {
		testBase
		inputs       []matrixInput
		expectedRows int
		expectedCols int
	}{
		{
			testBase: testBase{name: "hstack 1x1, 1x3, 1x2", expected: []float64{1, 2, 3, 4, 5, 6}},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
				{in: []float64{2, 3, 4}, rows: 1, cols: 3},
				{in: []float64{5, 6}, rows: 1, cols: 2},
			},
			expectedRows: 1,
			expectedCols: 6,
		},
		{
			testBase: testBase{name: "hstack 2x1, 2x3, 2x2", expected: []float64{1, 3, 4, 5, 9, 10, 2, 6, 7, 8, 11, 12}},
			inputs: []matrixInput{
				{in: []float64{1, 2}, rows: 2, cols: 1},
				{in: []float64{3, 4, 5, 6, 7, 8}, rows: 2, cols: 3},
				{in: []float64{9, 10, 11, 12}, rows: 2, cols: 2},
			},
			expectedRows: 2,
			expectedCols: 6,
		},
		{
			testBase: testBase{name: "hstack 3x1, 3x3, 3x2", expected: []float64{1, 4, 5, 6, 13, 14, 2, 7, 8, 9, 15, 16, 3, 10, 11, 12, 17, 18}},
			inputs: []matrixInput{
				{in: []float64{1, 2, 3}, rows: 3, cols: 1},
				{in: []float64{4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 3, cols: 3},
				{in: []float64{13, 14, 15, 16, 17, 18}, rows: 3, cols: 2},
			},
			expectedRows: 3,
			expectedCols: 6,
		},
		{
			testBase: testBase{name: "hstack 1x1, error", err: ErrOperationExec},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
			},
		},
		{
			testBase: testBase{name: "hstack 1x1 and 2x1, error", err: ErrOperationExec},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
				{in: []float64{2, 3}, rows: 2, cols: 1},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := newMatrix(t, test.inputs[0])
			matrices := make([]*Matrix, 0)
			for _, input := range test.inputs[1:] {
				matrices = append(matrices, newMatrix(t, input))
			}

			stacked, err := base.HStack(matrices)
			makeAssertions(t, matrixInput{in: test.expected, rows: test.expectedRows, cols: test.expectedCols}, test.err, err, stacked)
		})
	}
}

func TestMatrix_VStack(t *testing.T) {
	tests := []struct {
		testBase
		inputs       []matrixInput
		expectedRows int
		expectedCols int
	}{
		{
			testBase: testBase{name: "vstack 1x1, 3x1, 2x1", expected: []float64{1, 2, 3, 4, 5, 6}},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
				{in: []float64{2, 3, 4}, rows: 3, cols: 1},
				{in: []float64{5, 6}, rows: 2, cols: 1},
			},
			expectedRows: 6,
			expectedCols: 1,
		},
		{
			testBase: testBase{name: "vstack 1x2, 3x2, 2x2", expected: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}},
			inputs: []matrixInput{
				{in: []float64{1, 2}, rows: 1, cols: 2},
				{in: []float64{3, 4, 5, 6, 7, 8}, rows: 3, cols: 2},
				{in: []float64{9, 10, 11, 12}, rows: 2, cols: 2},
			},
			expectedRows: 6,
			expectedCols: 2,
		},
		{
			testBase: testBase{name: "hstack 1x3, 3x3, 2x3", expected: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18}},
			inputs: []matrixInput{
				{in: []float64{1, 2, 3}, rows: 1, cols: 3},
				{in: []float64{4, 5, 6, 7, 8, 9, 10, 11, 12}, rows: 3, cols: 3},
				{in: []float64{13, 14, 15, 16, 17, 18}, rows: 2, cols: 3},
			},
			expectedRows: 6,
			expectedCols: 3,
		},
		{
			testBase: testBase{name: "vstack 1x1, error", err: ErrOperationExec},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
			},
		},
		{
			testBase: testBase{name: "vstack 1x1 and 1x2, error", err: ErrOperationExec},
			inputs: []matrixInput{
				{in: []float64{1}, rows: 1, cols: 1},
				{in: []float64{2, 3}, rows: 1, cols: 2},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := newMatrix(t, test.inputs[0])
			matrices := make([]*Matrix, 0)
			for _, input := range test.inputs[1:] {
				matrices = append(matrices, newMatrix(t, input))
			}

			stacked, err := base.VStack(matrices)
			makeAssertions(t, matrixInput{in: test.expected, rows: test.expectedRows, cols: test.expectedCols}, test.err, err, stacked)
		})
	}
}

func TestMatrix_Equal(t *testing.T) {
	tests := []struct {
		name     string
		a        matrixInput
		b        matrixInput
		expected bool
	}{
		{
			name:     "equal matrices",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			expected: true,
		},
		{
			name:     "different rows count",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2}, rows: 1, cols: 2},
			expected: false,
		},
		{
			name:     "different cols count",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
			expected: false,
		},
		{
			name:     "different size",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 1, cols: 4},
			expected: false,
		},
		{
			name:     "another different size",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 4, cols: 1},
			expected: false,
		},
		{
			name:     "different values",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{1, 2, 5, 4}, rows: 2, cols: 2},
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)

			res1 := a.Equal(b)
			res2 := b.Equal(a)

			require.Equal(t, res1, res2)
			require.Equal(t, test.expected, res1)
		})
	}
}

func TestMatrix_Order(t *testing.T) {
	tests := []struct {
		testBase
		matrix  matrixInput
		indices []int
	}{
		{
			testBase: testBase{name: "{{1},{2},{3},{4}}[2,1,3,0]", expected: []float64{3, 2, 4, 1}},
			matrix:   matrixInput{in: []float64{1, 2, 3, 4}, rows: 4, cols: 1},
			indices:  []int{2, 1, 3, 0},
		},
		{
			testBase: testBase{name: "{{1,2},{3,4},{5,6}}[2,0,1]", expected: []float64{5, 6, 1, 2, 3, 4}},
			matrix:   matrixInput{in: []float64{1, 2, 3, 4, 5, 6}, rows: 3, cols: 2},
			indices:  []int{2, 0, 1},
		},
		{
			testBase: testBase{name: "{{1},{2},{3},{4}}[2,0,1], error", err: ErrOperationExec},
			matrix:   matrixInput{in: []float64{1, 2, 3, 4}, rows: 4, cols: 1},
			indices:  []int{2, 0, 1},
		},
		{
			testBase: testBase{name: "{{1},{2},{3},{4}}[2,0,1,1], error", err: ErrOperationExec},
			matrix:   matrixInput{in: []float64{1, 2, 3, 4}, rows: 4, cols: 1},
			indices:  []int{2, 0, 1, 1},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix := newMatrix(t, test.matrix)
			ordered, err := matrix.Order(test.indices)

			makeAssertions(t, matrixInput{in: test.expected, rows: test.matrix.rows, cols: test.matrix.cols}, test.err, err, ordered)
		})
	}
}

func TestCartesianProduct(t *testing.T) {
	tests := []struct {
		name     string
		inputs   [][]float64
		expected matrixInput
		err      error
	}{
		{
			name:     "{1, 2} X {3, 4, 5}",
			inputs:   [][]float64{{1, 2}, {3, 4, 5}},
			expected: matrixInput{in: []float64{1, 3, 1, 4, 1, 5, 2, 3, 2, 4, 2, 5}, rows: 6, cols: 2},
		},
		{
			name:     "{1, 2}",
			inputs:   [][]float64{{1, 2}},
			expected: matrixInput{in: []float64{1, 2}, rows: 2, cols: 1},
		},
		{
			name:     "{1, 2} X {3}",
			inputs:   [][]float64{{1, 2}, {3}},
			expected: matrixInput{in: []float64{1, 3, 2, 3}, rows: 2, cols: 2},
		},
		{
			name:     "{1, 2} X {3} X {4, 5}",
			inputs:   [][]float64{{1, 2}, {3}, {4, 5}},
			expected: matrixInput{in: []float64{1, 3, 4, 1, 3, 5, 2, 3, 4, 2, 3, 5}, rows: 4, cols: 3},
		},
		{
			name:     "{1, 2} X {3, 4, 5} X {6, 7}",
			inputs:   [][]float64{{1, 2}, {3, 4, 5}, {6, 7}},
			expected: matrixInput{in: []float64{1, 3, 6, 1, 3, 7, 1, 4, 6, 1, 4, 7, 1, 5, 6, 1, 5, 7, 2, 3, 6, 2, 3, 7, 2, 4, 6, 2, 4, 7, 2, 5, 6, 2, 5, 7}, rows: 12, cols: 3},
		},
		{name: "empty inputs", inputs: [][]float64{}, err: ErrOperationExec},
		{name: "no inputs", err: ErrOperationExec},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var inputs []*vector.Vector
			for _, input := range test.inputs {
				vec, err := vector.NewVector(input)
				require.NoError(t, err)
				inputs = append(inputs, vec)
			}
			product, err := CartesianProduct(inputs)
			if test.err == nil {
				require.NoError(t, err)
				t.Log("\n" + product.PrettyString())
				makeAssertions(t, test.expected, test.err, err, product)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}
