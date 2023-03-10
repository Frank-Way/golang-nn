package matrix

import (
	"fmt"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
	"strings"
)

type Matrix struct {
	vectors []*vector.Vector
	rows    int
	cols    int
}

func NewMatrix(vectors []*vector.Vector) (*Matrix, error) {
	res, err := func(vectors []*vector.Vector) (*Matrix, error) {
		if vectors == nil || len(vectors) < 1 {
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
	}(vectors)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return res, nil
}

func NewMatrixFlat(rows, cols int, flat *vector.Vector) (*Matrix, error) {
	vectors, err := func(rows, cols int, flat *vector.Vector) ([]*vector.Vector, error) {
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

		return vectors, nil
	}(rows, cols, flat)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return NewMatrix(vectors)
}

func NewMatrixRaw(values [][]float64) (*Matrix, error) {
	if values == nil {
		return nil, wraperr.NewWrapErr(ErrCreate, fmt.Errorf("no values provided: %v", values))
	}
	var vectors []*vector.Vector
	for _, row := range values {
		vec, err := vector.NewVector(row)
		if err != nil {
			return nil, wraperr.NewWrapErr(ErrCreate, err)
		}
		vectors = append(vectors, vec)
	}
	return NewMatrix(vectors)
}

func NewMatrixRawFlat(rows, cols int, values []float64) (*Matrix, error) {
	vec, err := vector.NewVector(values)
	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}
	return NewMatrixFlat(rows, cols, vec)
}

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

func (m *Matrix) String() string {
	vecStrings := make([]string, m.rows)
	for i, vec := range m.vectors {
		vecStrings[i] = vec.String()
	}
	return fmt.Sprintf("[%s]", strings.Join(vecStrings, " "))
}

func (m *Matrix) PrettyString() string {
	lu, ld, lm, l, ru, rd, rm, r := "⸢", "⸤", "│", "[", "⸣", "⸥", "│", "]"

	if m.cols == 1 {
		col, _ := m.GetCol(0)
		return col.PrettyString()
	}
	if m.rows == 1 {
		row, _ := m.GetRow(0)
		s := row.String()
		s = s[1 : len(s)-1]
		return fmt.Sprintf("%s %s %s", l, s, r)
	}

	strValues := make([]string, m.rows)

	for j := 0; j < m.cols; j++ {
		col, _ := m.GetCol(j)
		pString := col.PrettyString()
		pStrings := strings.Split(pString, "\n")
		for i := 0; i < m.rows; i++ {
			pStrings[i] = pStrings[i][4 : len(pStrings[i])-4]
			if j == 0 {
				strValues[i] = pStrings[i]
			} else {
				strValues[i] = strValues[i] + " " + pStrings[i]
			}
		}
	}

	for i, str := range strValues {
		format := "%s %s %s"
		if i == 0 {
			strValues[i] = fmt.Sprintf(format, lu, str, ru)
		} else if i == m.rows-1 {
			strValues[i] = fmt.Sprintf(format, ld, str, rd)
		} else {
			strValues[i] = fmt.Sprintf(format, lm, str, rm)
		}
	}

	return strings.Join(strValues, "\n")
}

func (m *Matrix) Get(row, col int) (float64, error) {
	if row < 0 || row >= m.rows || col < 0 || col >= m.cols {
		return 0, wraperr.NewWrapErr(ErrNotFound,
			fmt.Errorf("wrong row and col for matrix %dx%d: %d, %d", m.rows, m.cols, row, col))
	}

	value, _ := m.vectors[row].Get(col)
	return value, nil
}

func (m *Matrix) Raw() [][]float64 {
	values := make([][]float64, m.rows)
	for i, row := range m.vectors {
		values[i] = row.Raw()
	}
	return values
}

func (m *Matrix) RawFlat() []float64 {
	values := make([]float64, m.rows*m.cols)
	for i, row := range m.vectors {
		rawRow := row.Raw()
		for j, value := range rawRow {
			values[i*m.cols+j] = value
		}
	}
	return values
}

func (m *Matrix) Copy() *Matrix {
	matrix, _ := NewMatrixRaw(m.Raw())
	return matrix
}

func (m *Matrix) Size() (rows int, cols int) {
	return m.rows, m.cols
}

func (m *Matrix) Rows() int {
	return m.rows
}

func (m *Matrix) Cols() int {
	return m.cols
}

func (m *Matrix) GetRow(row int) (*vector.Vector, error) {
	if row < 0 || row >= m.rows {
		return nil, wraperr.NewWrapErr(ErrNotFound,
			fmt.Errorf("can not get %d'th row of %dx%d matrix", row, m.rows, m.cols))
	}

	return m.vectors[row].Copy(), nil
}

func (m *Matrix) GetCol(col int) (*vector.Vector, error) {
	res, err := func(col int) (*vector.Vector, error) {
		if col < 0 || col >= m.cols {
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
	}(col)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrNotFound, err)
	}

	return res, nil
}

func (m *Matrix) CheckEqualShape(matrix *Matrix) error {
	if m.rows != matrix.rows || m.cols != matrix.cols {
		return fmt.Errorf("matrix sizes mismatch: %dx%d != %dx%d", m.rows, matrix.rows, m.cols, matrix.cols)
	}
	return nil
}
