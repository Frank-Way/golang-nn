package matrix

import (
	"github.com/stretchr/testify/require"
	"nn/pkg/mmath/vector"
	"testing"
)

func TestNewMatrix(t *testing.T) {
	tests := []struct {
		name string
		in   [][]float64
		err  error
	}{
		{name: "2x2 matrix, no error", in: [][]float64{{1, 2}, {3, 4}}},
		{name: "1x1 matrix, no error", in: [][]float64{{1}}},
		{name: "1x2 matrix, no error", in: [][]float64{{1, 3}}},
		{name: "empty input, error", err: ErrCreate},
		{name: "different sizes, error", in: [][]float64{{1, 2}, {3, 4, 5}}, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vectors := make([]*vector.Vector, len(test.in))
			for i, row := range test.in {
				vec, _ := vector.NewVector(row)
				vectors[i] = vec
			}

			matrix, err := NewMatrix(vectors)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.in), matrix.rows)
				require.Equal(t, len(test.in[0]), matrix.cols)
				for i, row := range test.in {
					for j, value := range row {
						actual, err := matrix.vectors[i].Get(j)
						require.NoError(t, err)
						require.Equal(t, value, actual)
					}
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestNewMatrixFlat(t *testing.T) {
	tests := []struct {
		name string
		in   []float64
		rows int
		cols int
		err  error
	}{
		{name: "2x2 matrix, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
		{name: "1x1 matrix, no error", in: []float64{1}, rows: 1, cols: 1},
		{name: "1x2 matrix, no error", in: []float64{1, 3}, rows: 1, cols: 2},
		{name: "empty input, error", err: ErrCreate},
		{name: "wrong sizes, error", in: []float64{1, 2, 3, 4, 5}, rows: 2, cols: 2, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			vec, _ := vector.NewVector(test.in)
			matrix, err := NewMatrixFlat(test.rows, test.cols, vec)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.rows, matrix.rows)
				require.Equal(t, test.cols, matrix.cols)
				for i := 0; i < test.rows; i++ {
					for j := 0; j < test.cols; j++ {
						actual, err := matrix.vectors[i].Get(j)
						expected := test.in[i*test.rows+j]
						require.NoError(t, err)
						require.Equal(t, expected, actual)
					}
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestNewMatrixRaw(t *testing.T) {
	tests := []struct {
		name     string
		in       [][]float64
		nilCheck bool
		err      error
	}{
		{name: "2x2 matrix, no error", in: [][]float64{{1, 2}, {3, 4}}},
		{name: "1x1 matrix, no error", in: [][]float64{{1}}},
		{name: "1x2 matrix, no error", in: [][]float64{{1, 3}}},
		{name: "empty input, error", err: ErrCreate},
		{name: "different sizes, error", in: [][]float64{{1, 2}, {3, 4, 5}}, err: ErrCreate},
		{name: "nil row, error", in: [][]float64{{1, 2}, {3, 4}}, nilCheck: true, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.nilCheck {
				test.in = append(test.in, nil)
			}
			matrix, err := NewMatrixRaw(test.in)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.in), matrix.rows)
				require.Equal(t, len(test.in[0]), matrix.cols)
				for i, row := range test.in {
					for j, value := range row {
						actual, err := matrix.vectors[i].Get(j)
						require.NoError(t, err)
						require.Equal(t, value, actual)
					}
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestNewMatrixRawFlat(t *testing.T) {
	tests := []struct {
		name string
		in   []float64
		rows int
		cols int
		err  error
	}{
		{name: "2x2 matrix, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
		{name: "1x1 matrix, no error", in: []float64{1}, rows: 1, cols: 1},
		{name: "1x2 matrix, no error", in: []float64{1, 3}, rows: 1, cols: 2},
		{name: "empty input, error", err: ErrCreate},
		{name: "wrong sizes, error", in: []float64{1, 2, 3, 4, 5}, rows: 2, cols: 2, err: ErrCreate},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.rows, matrix.rows)
				require.Equal(t, test.cols, matrix.cols)
				for i := 0; i < test.rows; i++ {
					for j := 0; j < test.cols; j++ {
						actual, err := matrix.vectors[i].Get(j)
						expected := test.in[i*test.rows+j]
						require.NoError(t, err)
						require.Equal(t, expected, actual)
					}
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestNewMatrixOf(t *testing.T) {
	matrix, err := NewMatrixOf(2, 3, 2)
	require.NoError(t, err)
	require.Equal(t, 2, matrix.rows)
	require.Equal(t, 3, matrix.cols)
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			value, err := matrix.Get(i, j)
			require.NoError(t, err)
			require.Equal(t, 2.0, value)
		}
	}

	matrix, err = NewMatrixOf(0, 0, 2)
	require.Error(t, err)
	require.ErrorIs(t, err, ErrCreate)
}

func TestZeros(t *testing.T) {
	matrix, err := Zeros(3, 2)
	require.NoError(t, err)
	require.Equal(t, 3, matrix.rows)
	require.Equal(t, 2, matrix.cols)
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			value, err := matrix.Get(i, j)
			require.NoError(t, err)
			require.Equal(t, 0.0, value)
		}
	}
}

func TestMatrix_String(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		expected string
	}{
		{name: "2x2 matrix", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expected: "[[1 2] [3 4]]"},
		{name: "1x4 matrix", in: []float64{1, 2, 3, 4}, rows: 1, cols: 4, expected: "[[1 2 3 4]]"},
		{name: "4x1 matrix", in: []float64{1, 2, 3, 4}, rows: 4, cols: 1, expected: "[[1] [2] [3] [4]]"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			actual := matrix.String()
			require.Equal(t, test.expected, actual)
		})
	}
}

func TestMatrix_PrettyString(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		expected string
	}{
		{name: "1x1 matrix", in: []float64{1}, rows: 1, cols: 1, expected: "[ 1 ]"},
		{name: "1x2 matrix", in: []float64{1, 2}, rows: 1, cols: 2, expected: "[ 1 2 ]"},
		{name: "1x3 matrix", in: []float64{1, 2, 3}, rows: 1, cols: 3, expected: "[ 1 2 3 ]"},
		{name: "2x1 matrix", in: []float64{1, 2}, rows: 2, cols: 1, expected: "⸢ 1 ⸣\n⸤ 2 ⸥"},
		{name: "2x2 matrix", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expected: "⸢ 1 2 ⸣\n⸤ 3 4 ⸥"},
		{name: "2x3 matrix", in: []float64{1, 2, 3, 4, 5, 6}, rows: 2, cols: 3, expected: "⸢ 1 2 3 ⸣\n⸤ 4 5 6 ⸥"},
		{name: "3x1 matrix", in: []float64{1, 2, 3}, rows: 3, cols: 1, expected: "⸢ 1 ⸣\n│ 2 │\n⸤ 3 ⸥"},
		{name: "3x2 matrix", in: []float64{1, 2, 3, 4, 5, 6}, rows: 3, cols: 2, expected: "⸢ 1 2 ⸣\n│ 3 4 │\n⸤ 5 6 ⸥"},
		{name: "3x3 matrix", in: []float64{1, -2, 3, 4, -5, 6, 777, 888, 999}, rows: 3, cols: 3, expected: "⸢   1  -2   3 ⸣\n│   4  -5   6 │\n⸤ 777 888 999 ⸥"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			str := matrix.PrettyString()
			require.Equal(t, test.expected, str)
		})
	}
}

func TestMatrix_Get(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		row      int
		col      int
		expected float64
		err      error
	}{
		{name: "{{1,2},{3,4}}[0][0], no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 0, col: 0, expected: 1},
		{name: "{{1,2},{3,4}}[0][1], no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 0, col: 1, expected: 2},
		{name: "{{1,2},{3,4}}[1][0], no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 1, col: 0, expected: 3},
		{name: "{{1,2},{3,4}}[1][1], no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 1, col: 1, expected: 4},
		{name: "{{1,2,3,4}}[0][3], no error", in: []float64{1, 2, 3, 4}, rows: 1, cols: 4, row: 0, col: 3, expected: 4},
		{name: "{{1,2},{3,4}}[2][0], error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 2, col: 0, err: ErrNotFound},
		{name: "{{1,2},{3,4}}[-1][0], error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: -1, col: 0, err: ErrNotFound},
		{name: "{{1,2},{3,4}}[0][2], error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 0, col: 2, err: ErrNotFound},
		{name: "{{1,2},{3,4}}[0][-1], error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 0, col: -1, err: ErrNotFound},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			actual, err := matrix.Get(test.row, test.col)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, test.expected, actual)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestMatrix_Raw(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 2, []float64{1, 2, 3, 4})
	require.NoError(t, err)

	raw := matrix.Raw()
	require.Equal(t, 2, len(raw))
	require.Equal(t, 2, len(raw[0]))
	require.Equal(t, 2, len(raw[1]))
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected, err := matrix.Get(i, j)
			require.NoError(t, err)
			actual := raw[i][j]
			require.Equal(t, expected, actual)
		}
	}
}

func TestMatrix_RawFlat(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 2, []float64{1, 2, 3, 4})
	require.NoError(t, err)

	rawFlat := matrix.RawFlat()
	require.Equal(t, 4, len(rawFlat))
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			expected, err := matrix.Get(i, j)
			require.NoError(t, err)
			actual := rawFlat[i*2+j]
			require.Equal(t, expected, actual)
		}
	}
}

func TestMatrix_Copy(t *testing.T) {
	matrix, err := NewMatrixRawFlat(2, 2, []float64{1, 2, 3, 4})
	require.NoError(t, err)

	cp := matrix.Copy()
	require.True(t, matrix != cp)
	require.Equal(t, matrix.rows, cp.rows)
	require.Equal(t, matrix.cols, cp.cols)

	for i := 0; i < cp.rows; i++ {
		for j := 0; j < cp.cols; j++ {
			expected, _ := matrix.Get(i, j)
			actual, _ := cp.Get(i, j)
			require.Equal(t, expected, actual)
		}
	}
}

func TestMatrix_Rows(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		expected int
	}{
		{name: "{{1,2},{3,4}}, 2 rows", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expected: 2},
		{name: "{{1},{2},{3},{4}}, 4 rows", in: []float64{1, 2, 3, 4}, rows: 4, cols: 1, expected: 4},
		{name: "{{1,2,3,4}}, 1 rows", in: []float64{1, 2, 3, 4}, rows: 1, cols: 4, expected: 1},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			rows := matrix.Rows()
			require.Equal(t, test.expected, rows)
		})
	}
}

func TestMatrix_Cols(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		expected int
	}{
		{name: "{{1,2},{3,4}}, 2 cols", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expected: 2},
		{name: "{{1},{2},{3},{4}}, 1 cols", in: []float64{1, 2, 3, 4}, rows: 4, cols: 1, expected: 1},
		{name: "{{1,2,3,4}}, 4 cols", in: []float64{1, 2, 3, 4}, rows: 1, cols: 4, expected: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			cols := matrix.Cols()
			require.Equal(t, test.expected, cols)
		})
	}
}

func TestMatrix_Size(t *testing.T) {
	tests := []struct {
		name         string
		in           []float64
		rows         int
		cols         int
		expectedRows int
		expectedCols int
	}{
		{name: "{{1,2},{3,4}}, 2 rows 2 cols", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, expectedRows: 2, expectedCols: 2},
		{name: "{{1},{2},{3},{4}}, 4 rows 1 cols", in: []float64{1, 2, 3, 4}, rows: 4, cols: 1, expectedRows: 4, expectedCols: 1},
		{name: "{{1,2,3,4}}, 1 rows 4 cols", in: []float64{1, 2, 3, 4}, rows: 1, cols: 4, expectedRows: 1, expectedCols: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			rows, cols := matrix.Size()
			require.Equal(t, test.expectedRows, rows)
			require.Equal(t, test.expectedCols, cols)
		})
	}
}

func TestMatrix_GetRow(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		row      int
		expected []float64
		err      error
	}{
		{name: "{{1,2},{3,4}}, 0 row, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 0, expected: []float64{1, 2}},
		{name: "{{1,2},{3,4}}, 1 row, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 1, expected: []float64{3, 4}},
		{name: "{{1,2},{3,4}}, 2 row, error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: 2, err: ErrNotFound},
		{name: "{{1,2},{3,4}}, -1 row, error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, row: -1, err: ErrNotFound},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			row, err := matrix.GetRow(test.row)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), row.Size())
				for i, value := range test.expected {
					actual, err := row.Get(i)
					require.NoError(t, err)
					require.Equal(t, value, actual)
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestMatrix_GetCol(t *testing.T) {
	tests := []struct {
		name     string
		in       []float64
		rows     int
		cols     int
		col      int
		expected []float64
		err      error
	}{
		{name: "{{1,2},{3,4}}, 0 col, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, col: 0, expected: []float64{1, 3}},
		{name: "{{1,2},{3,4}}, 1 col, no error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, col: 1, expected: []float64{2, 4}},
		{name: "{{1,2},{3,4}}, 2 col, error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, col: 2, err: ErrNotFound},
		{name: "{{1,2},{3,4}}, -1 col, error", in: []float64{1, 2, 3, 4}, rows: 2, cols: 2, col: -1, err: ErrNotFound},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			matrix, err := NewMatrixRawFlat(test.rows, test.cols, test.in)
			require.NoError(t, err)

			col, err := matrix.GetCol(test.col)
			if test.err == nil {
				require.NoError(t, err)
				require.Equal(t, len(test.expected), col.Size())
				for i, value := range test.expected {
					actual, err := col.Get(i)
					require.NoError(t, err)
					require.Equal(t, value, actual)
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.err)
			}
		})
	}
}

func TestMatrix_CheckEqualShape(t *testing.T) {
	tests := []struct {
		name     string
		a        matrixInput
		b        matrixInput
		nilCheck bool
		err      bool
	}{
		{
			name: "2x2 == 2x2",
			a:    matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:    matrixInput{in: []float64{5, 6, 7, 8}, rows: 2, cols: 2},
		},
		{
			name: "2x2 != 1x4",
			a:    matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:    matrixInput{in: []float64{5, 6, 7, 8}, rows: 1, cols: 4},
			err:  true,
		},
		{
			name:     "2x2 != nil",
			a:        matrixInput{in: []float64{1, 2, 3, 4}, rows: 2, cols: 2},
			b:        matrixInput{in: []float64{5, 6, 7, 8}, rows: 2, cols: 2},
			nilCheck: true,
			err:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := newMatrix(t, test.a)
			b := newMatrix(t, test.b)
			if test.nilCheck {
				b = nil
			}

			check := a.CheckEqualShape(b)
			if test.err {
				require.Error(t, check)
			} else {
				require.NoError(t, check)
			}
		})
	}
}
