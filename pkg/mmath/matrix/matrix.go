// Package matrix provides functionality for Matrix (wrap on slice of vector.Vector).
package matrix

import (
	"fmt"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
	"strings"
)

// Matrix holds slice of vector.Vector. It provides some useful methods. It can be treated as immutable by using Copy().
//
// Example:
//
// Matrix 2x3 of 6 floats [1, 2, 3, 4, 5, 6]:
//    | 1 2 3 |
//    | 4 5 6 |
type Matrix struct {
	vectors []*vector.Vector
	rows    int
	cols    int
}

// NewMatrix creates Matrix from given slice of vectors. There must be at least one value in slice.
// All vectors must have same size.
//
// Throws ErrCreate error.
func NewMatrix(vectors []*vector.Vector) (m *Matrix, err error) {
	defer wraperr.WrapError(ErrCreate, &err)

	if len(vectors) < 1 {
		return nil, fmt.Errorf("no vectors provided: %v", vectors)
	}

	rows := len(vectors)

	for i, row := range vectors {
		if row == nil {
			return nil, fmt.Errorf("%d;th row is nil: %v", i, row)
		}
	}

	cols := vectors[0].Size()

	for i, row := range vectors {
		if row.Size() != cols {
			return nil, fmt.Errorf("%d'th row length mismatch first row length: %d != %d", i, row.Size(), cols)
		}
	}

	return &Matrix{
		vectors: vectors,
		rows:    rows,
		cols:    cols,
	}, nil
}

// NewMatrixFlat creates Matrix from given Vector with given rows and cols count. Rows and cols must be
// non-zero positive values. Size of vector must be `rows*cols`.
//
// Throws ErrCreate error.
func NewMatrixFlat(rows, cols int, flat *vector.Vector) (m *Matrix, err error) {
	defer wraperr.WrapError(ErrCreate, &err)

	if flat == nil {
		return nil, fmt.Errorf("no vector provided: %v", flat)
	} else if rows < 1 {
		return nil, fmt.Errorf("negative or zero rows count: %d", rows)
	} else if cols < 1 {
		return nil, fmt.Errorf("negative or zero cols count: %d", cols)
	} else if flat.Size() != rows*cols {
		return nil, fmt.Errorf("wrong vectors count provided for rows*cols matrix: %d != %d*%d=%d",
			flat.Size(), rows, cols, rows*cols)
	}

	vectors, err := flat.Split(cols)
	if err != nil {
		return nil, err
	}

	if len(vectors) != rows {
		return nil, fmt.Errorf("split parts count of vector sized %d for matrix sized %dx%d mismatches: %d != %d",
			flat.Size(), rows, cols, len(vectors), rows)
	}

	return NewMatrix(vectors)
}

// NewMatrixRaw creates Matrix from given slice of slice of floats. There must be at least one value in slice.
// All inner slices must be non-nil and have same sizes.
//
// Throws ErrCreate error.
func NewMatrixRaw(values [][]float64) (m *Matrix, err error) {
	defer wraperr.WrapError(ErrCreate, &err)

	if len(values) < 1 {
		return nil, fmt.Errorf("no values provided: %v", values)
	}
	var vectors []*vector.Vector
	for _, row := range values {
		vec, err := vector.NewVector(row)
		if err != nil {
			return nil, err
		}
		vectors = append(vectors, vec)
	}
	return NewMatrix(vectors)
}

// NewMatrixRawFlat creates Matrix from given slice of floats with given rows and cols count. Rows and cols must be
// non-zero positive values. Size of slice must be `rows*cols`.
//
// Throws ErrCreate error.
func NewMatrixRawFlat(rows, cols int, values []float64) (*Matrix, error) {
	vec, err := vector.NewVector(values)
	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}
	return NewMatrixFlat(rows, cols, vec)
}

// NewMatrixOf creates Matrix with given rows and cols count filled with given value. Rows and cols must be
// non-zero positive values.
//
// Throws ErrCreate error.
//
// Example:
//     NewMatrixOf(2, 1, 3) = | 3 |
//                            | 3 |
func NewMatrixOf(rows, cols int, value float64) (*Matrix, error) {
	flat, err := vector.NewVectorOf(value, rows*cols)
	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return NewMatrixFlat(rows, cols, flat)
}

func Zeros(rows, cols int) (*Matrix, error) {
	return NewMatrixOf(rows, cols, 0)
}

// Example:
//     | 1 2 3 |.String() = `[[1 2 3] [4 5 6]]`
//     | 4 5 6 |
func (m *Matrix) String() string {
	if m == nil {
		return "<nil>"
	}
	vecStrings := make([]string, m.rows)
	for i, vec := range m.vectors {
		vecStrings[i] = vec.String()
	}
	return fmt.Sprintf("[%s]", strings.Join(vecStrings, " "))
}

// Example:
//     | 1 2 3 |.PrettyString() = `| 1 2 3 |
//     | 4 5 6 |                   | 4 5 6 |`
func (m *Matrix) PrettyString() string {
	if m == nil {
		return "<nil>"
	}
	l, r := "|", "|"

	if m.cols == 1 {
		col, _ := m.GetCol(0)
		return col.PrettyString()
	}

	strValues := make([]string, m.rows)

	for j := 0; j < m.cols; j++ {
		col, _ := m.GetCol(j)
		pString := col.PrettyString()
		pStrings := strings.Split(pString, "\n")
		for i := 0; i < m.rows; i++ {
			pStrings[i] = pStrings[i][2 : len(pStrings[i])-2]
			if j == 0 {
				strValues[i] = pStrings[i]
			} else {
				strValues[i] = strValues[i] + " " + pStrings[i]
			}
		}
	}

	format := "%s %s %s"
	for i, str := range strValues {
		strValues[i] = fmt.Sprintf(format, l, str, r)
	}

	return strings.Join(strValues, "\n")
}

// Example:
//     | 1 2 3 |.ShortString() = `matrix 2x3`
//     | 4 5 6 |
func (m *Matrix) ShortString() string {
	if m == nil {
		return "<nil>"
	}
	return fmt.Sprintf("matrix %dx%d", m.rows, m.cols)
}

// Get return value for given row and col.
//
// Throws ErrNotFound error.
func (m *Matrix) Get(row, col int) (value float64, err error) {
	defer wraperr.WrapError(ErrNotFound, &err)

	if m == nil {
		return 0, ErrNil
	} else if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		return 0, fmt.Errorf("wrong row and col for matrix %dx%d: %d, %d", m.rows, m.cols, row, col)
	}

	value, err = m.vectors[row].Get(col)
	if err != nil {
		return 0, err
	}

	return value, nil
}

// Raw return Matrix as slice of slices of floats
func (m *Matrix) Raw() [][]float64 {
	if m == nil {
		return nil
	}
	values := make([][]float64, m.rows)
	for i, row := range m.vectors {
		values[i] = row.Raw()
	}
	return values
}

// RawFlat return Matrix as slice of floats
func (m *Matrix) RawFlat() []float64 {
	if m == nil {
		return nil
	}
	values := make([]float64, m.rows*m.cols)
	for i, row := range m.vectors {
		rawRow := row.Raw()
		for j, value := range rawRow {
			values[i*m.cols+j] = value
		}
	}
	return values
}

// Copy return deep copy of Matrix
func (m *Matrix) Copy() *Matrix {
	if m == nil {
		return nil
	}
	matrix, _ := NewMatrixRaw(m.Raw())
	return matrix
}

// Size return rows and cols count of Matrix
func (m *Matrix) Size() (rows int, cols int) {
	return m.Rows(), m.Cols()
}

// Rows return rows count of Matrix
func (m *Matrix) Rows() int {
	if m == nil {
		return 0
	}
	return m.rows
}

// Cols return cols count of Matrix
func (m *Matrix) Cols() int {
	if m == nil {
		return 0
	}
	return m.cols
}

// GetRow return row as vector.
//
// Throws ErrNotFound error.
func (m *Matrix) GetRow(row int) (vec *vector.Vector, err error) {
	defer wraperr.WrapError(ErrNotFound, &err)

	if m == nil {
		return nil, ErrNil
	} else if row < 0 || row >= m.rows {
		return nil, fmt.Errorf("can not get %d'th row of %dx%d matrix", row, m.rows, m.cols)
	}

	return m.vectors[row].Copy(), nil
}

// GetRow return col as vector.
//
// Throws ErrNotFound error.
func (m *Matrix) GetCol(col int) (vec *vector.Vector, err error) {
	defer wraperr.WrapError(ErrNotFound, &err)

	if m == nil {
		return nil, ErrNil
	} else if col < 0 || col >= m.cols {
		return nil, fmt.Errorf("can not get %d'th col of %dx%d matrix", col, m.rows, m.cols)
	}

	values := make([]float64, m.rows)
	for i := 0; i < m.rows; i++ {
		value, err := m.vectors[i].Get(col)
		if err != nil {
			return nil, err
		}
		values[i] = value
	}

	return vector.NewVector(values)
}

// CheckEqualShape return error if provided matrix size mismatch this size. Function return nil if sizes match.
func (m *Matrix) CheckEqualShape(matrix *Matrix) error {
	if m == nil {
		return ErrNil
	} else if matrix == nil {
		return fmt.Errorf("no matrix provided: %v", matrix)
	} else if m.rows != matrix.rows || m.cols != matrix.cols {
		return fmt.Errorf("matrix sizes mismatch: %dx%d != %dx%d", m.rows, matrix.rows, m.cols, matrix.cols)
	}
	return nil
}
