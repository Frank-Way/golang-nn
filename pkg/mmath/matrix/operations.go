package matrix

import (
	"fmt"
	"math"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
	"sync"
)

func (m *Matrix) T() *Matrix {
	cols := make([]*vector.Vector, m.cols)

	for j := 0; j < m.cols; j++ {
		col, _ := m.GetCol(j)
		cols[j] = col
	}

	matrix, _ := NewMatrix(cols)
	return matrix
}

const ParallelThreshold = 64

func (m *Matrix) MatMul(matrix *Matrix) (*Matrix, error) {
	if m.rows*matrix.cols > ParallelThreshold {
		return m.matMulImplMulti(matrix)
	}
	return m.matMulImplSingle(matrix)
}

func (m *Matrix) matMulImplSingle(matrix *Matrix) (*Matrix, error) {
	res, err := func(matrix *Matrix) (*Matrix, error) {
		N, M := m.rows, matrix.cols
		if m.cols != matrix.rows {
			return nil, fmt.Errorf("can't mul matrices sized %dx%d and %dx%d", m.rows, m.cols, matrix.rows, matrix.cols)
		}

		rows := m.vectors
		cols := make([]*vector.Vector, M)
		var vec *vector.Vector
		var err error
		for j := 0; j < M; j++ {
			vec, err = matrix.GetCol(j)
			if err != nil {
				return nil, err
			}
			cols[j] = vec
		}

		values := make([][]float64, N)
		var value float64
		var rawRow []float64
		for i, row := range rows {
			rawRow = make([]float64, M)
			for j, col := range cols {
				value, err = row.MulScalar(col)
				if err != nil {
					return nil, err
				}
				rawRow[j] = value
			}
			values[i] = rawRow
		}

		return NewMatrixRaw(values)
	}(matrix)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) matMulImplMulti(matrix *Matrix) (*Matrix, error) {
	res, err := func(matrix *Matrix) (*Matrix, error) {
		N, M := m.rows, matrix.cols
		if m.cols != matrix.rows {
			return nil, fmt.Errorf("can't mul matrices sized %dx%d and %dx%d", m.rows, m.cols, matrix.rows, matrix.cols)
		}

		rows := m.vectors
		cols := make([]*vector.Vector, M)
		var vec *vector.Vector
		var err error
		for j := 0; j < M; j++ {
			vec, err = matrix.GetCol(j)
			if err != nil {
				return nil, err
			}
			cols[j] = vec
		}

		values := make([][]float64, N)
		var value float64
		var rawRow []float64
		wg := sync.WaitGroup{}
		for i, row := range rows {
			wg.Add(1)
			rawRow = make([]float64, M)
			values[i] = rawRow
			go func(row *vector.Vector, arr []float64) {
				defer wg.Done()
				for j, col := range cols {
					value, _ = row.MulScalar(col)
					arr[j] = value
				}
			}(row, rawRow)
		}

		wg.Wait()
		return NewMatrixRaw(values)
	}(matrix)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) doOperation(matrix *Matrix, operation func(a, b float64) float64, description string) (*Matrix, error) {
	res, err := func(matrix *Matrix, operation func(a, b float64) float64) (*Matrix, error) {
		rows, cols := matrix.Size()
		if rows != m.rows || cols != m.cols {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, rows, cols)
		}

		values := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			row := make([]float64, cols)
			for j := 0; j < cols; j++ {
				a, _ := m.Get(i, j)
				b, _ := matrix.Get(i, j)
				row[j] = operation(a, b)
			}
			values[i] = row
		}

		return NewMatrixRaw(values)
	}(matrix, operation)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec,
			fmt.Errorf("can not perform '%s'", description))
	}

	return res, nil
}

func (m *Matrix) Add(matrix *Matrix) (*Matrix, error) {
	return m.doOperation(matrix, func(a, b float64) float64 {
		return a + b
	}, "add")
}

func (m *Matrix) Sub(matrix *Matrix) (*Matrix, error) {
	return m.doOperation(matrix, func(a, b float64) float64 {
		return a - b
	}, "sub")
}

func (m *Matrix) Mul(matrix *Matrix) (*Matrix, error) {
	return m.doOperation(matrix, func(a, b float64) float64 {
		return a * b
	}, "mul")
}

func (m *Matrix) Div(matrix *Matrix) (*Matrix, error) {
	return m.doOperation(matrix, func(a, b float64) float64 {
		return a / b
	}, "div")
}

func stackRow(row *vector.Vector, count int) *Matrix {
	rows := make([]*vector.Vector, count)
	for i := 0; i < count; i++ {
		rows[i] = row.Copy()
	}

	matrix, _ := NewMatrix(rows)

	return matrix
}

func (m *Matrix) AddRow(row *vector.Vector) (*Matrix, error) {
	return m.Add(stackRow(row, m.rows))
}

func (m *Matrix) SubRow(row *vector.Vector) (*Matrix, error) {
	return m.Sub(stackRow(row, m.rows))
}

func (m *Matrix) MulRow(row *vector.Vector) (*Matrix, error) {
	return m.Mul(stackRow(row, m.rows))
}

func (m *Matrix) DivRow(row *vector.Vector) (*Matrix, error) {
	return m.Div(stackRow(row, m.rows))
}

func stackCol(col *vector.Vector, count int) *Matrix {
	cols := make([]*vector.Vector, count)
	for i := 0; i < count; i++ {
		cols[i] = col.Copy()
	}

	matrix, _ := NewMatrix(cols)

	return matrix.T()
}

func (m *Matrix) AddCol(col *vector.Vector) (*Matrix, error) {
	return m.Add(stackCol(col, m.cols))
}

func (m *Matrix) SubCol(col *vector.Vector) (*Matrix, error) {
	return m.Sub(stackCol(col, m.cols))
}

func (m *Matrix) MulCol(col *vector.Vector) (*Matrix, error) {
	return m.Mul(stackCol(col, m.cols))
}

func (m *Matrix) DivCol(col *vector.Vector) (*Matrix, error) {
	return m.Div(stackCol(col, m.cols))
}

func (m *Matrix) AddNum(number float64) *Matrix {
	matrix, _ := NewMatrixOf(m.rows, m.cols, number)
	res, _ := m.Add(matrix)

	return res
}

func (m *Matrix) SubNum(number float64) *Matrix {
	matrix, _ := NewMatrixOf(m.rows, m.cols, number)
	res, _ := m.Sub(matrix)

	return res
}

func (m *Matrix) MulNum(number float64) *Matrix {
	matrix, _ := NewMatrixOf(m.rows, m.cols, number)
	res, _ := m.Mul(matrix)

	return res
}

func (m *Matrix) DivNum(number float64) *Matrix {
	matrix, _ := NewMatrixOf(m.rows, m.cols, number)
	res, _ := m.Div(matrix)

	return res
}

type Axis uint8

const (
	Horizontal Axis = iota
	Vertical
)

func (m *Matrix) SumAxed(axis Axis) (*vector.Vector, error) {
	res, err := func(axis Axis) (*vector.Vector, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Sum()
			}
			return vector.NewVector(values)
		case Vertical:
			values := make([]float64, m.cols)
			t := m.T()
			for j, col := range t.vectors {
				values[j] = col.Sum()
			}
			return vector.NewVector(values)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}(axis)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Sum() float64 {
	vec, _ := m.SumAxed(Horizontal)
	return vec.Sum()
}

func (m *Matrix) MaxAxed(axis Axis) (*vector.Vector, error) {
	res, err := func(axis Axis) (*vector.Vector, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Max()
			}
			return vector.NewVector(values)
		case Vertical:
			values := make([]float64, m.cols)
			t := m.T()
			for j, col := range t.vectors {
				values[j] = col.Max()
			}
			return vector.NewVector(values)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}(axis)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Max() float64 {
	vec, _ := m.MaxAxed(Horizontal)
	return vec.Max()
}

func (m *Matrix) MinAxed(axis Axis) (*vector.Vector, error) {
	res, err := func(axis Axis) (*vector.Vector, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Min()
			}
			return vector.NewVector(values)
		case Vertical:
			values := make([]float64, m.cols)
			t := m.T()
			for j, col := range t.vectors {
				values[j] = col.Min()
			}
			return vector.NewVector(values)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}(axis)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Min() float64 {
	vec, _ := m.MinAxed(Horizontal)
	return vec.Min()
}

func (m *Matrix) AvgAxed(axis Axis) (*vector.Vector, error) {
	res, err := func(axis Axis) (*vector.Vector, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Avg()
			}
			return vector.NewVector(values)
		case Vertical:
			values := make([]float64, m.cols)
			t := m.T()
			for j, col := range t.vectors {
				values[j] = col.Avg()
			}
			return vector.NewVector(values)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}(axis)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Avg() float64 {
	vec, _ := m.AvgAxed(Horizontal)
	return vec.Avg()
}

func (m *Matrix) Abs() *Matrix {
	cp := m.Copy()
	for i := 0; i < m.rows; i++ {
		cp.vectors[i] = cp.vectors[i].Abs()
	}

	return cp
}

func (m *Matrix) Exp() *Matrix {
	cp := m.Copy()
	for i := 0; i < m.rows; i++ {
		cp.vectors[i] = cp.vectors[i].Exp()
	}

	return cp
}

func (m *Matrix) Pow(scale float64) *Matrix {
	cp := m.Copy()
	for i := 0; i < m.rows; i++ {
		cp.vectors[i] = cp.vectors[i].Pow(scale)
	}

	return cp
}

func (m *Matrix) Sqr() *Matrix {
	cp := m.Copy()
	for i := 0; i < m.rows; i++ {
		cp.vectors[i] = cp.vectors[i].Sqr()
	}

	return cp
}

func (m *Matrix) Sqrt() *Matrix {
	cp := m.Copy()
	for i := 0; i < m.rows; i++ {
		cp.vectors[i] = cp.vectors[i].Sqrt()
	}

	return cp
}

func (m *Matrix) SubMatrix(rowsStart, rowsStop, rowsStep, colsStart, colsStop, colsStep int) (*Matrix, error) {
	res, err := func(rowsStart, rowsStop, rowsStep, colsStart, colsStop, colsStep int) (*Matrix, error) {
		l := rowsStop - rowsStart
		resLen := int(math.Ceil(float64(l) / float64(rowsStep)))
		if l < 1 {
			return nil, fmt.Errorf("wrong start and stop row indicies for sub matrix: %d >= %d", rowsStart, rowsStop)
		} else if rowsStart < 0 {
			return nil, fmt.Errorf("negative start row index for sub matrix: %d", rowsStart)
		} else if rowsStop < 1 {
			return nil, fmt.Errorf("negative or zero stop row index for sub matrix: %d", rowsStop)
		} else if rowsStep < 1 {
			return nil, fmt.Errorf("negative or zero row step for sub matrix: %d", rowsStep)
		} else if m.rows < rowsStop {
			return nil, fmt.Errorf("stop row index for sub matrix out of matrix rows count: %d > %d", rowsStop, m.rows)
		} else if resLen < 1 {
			return nil, fmt.Errorf("zero resulting sub matrix length with start %d stop %d step %d rows %d", rowsStart, rowsStop, rowsStep, m.rows)
		}

		values := make([]*vector.Vector, resLen)

		for i, cnt := rowsStart, 0; i < rowsStop && cnt < resLen; i, cnt = i+rowsStep, cnt+1 {
			slice, err := m.vectors[i].Slice(colsStart, colsStop, colsStep)
			if err != nil {
				return nil, err
			}
			values[cnt] = slice
		}

		return NewMatrix(values)
	}(rowsStart, rowsStop, rowsStep, colsStart, colsStop, colsStep)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) HStack(matrices []*Matrix) (*Matrix, error) {
	res, err := func(matrices []*Matrix) (*Matrix, error) {
		if len(matrices) < 1 {
			return nil, fmt.Errorf("no matrices provided for horizontal stacking")
		}

		var err error
		vectors := m.Copy().vectors
		for k := 0; k < len(matrices); k++ {
			if matrices[k].rows != m.rows {
				return nil, fmt.Errorf("%d'th matrix rows count mismatch first matrix rows count: %d != %d",
					k, matrices[k].rows, m.rows)
			}

			for i := 0; i < m.rows; i++ {
				vectors[i], err = vectors[i].Concatenate(matrices[k].vectors[i])
				if err != nil {
					return nil, err
				}
			}
		}

		return NewMatrix(vectors)
	}(matrices)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) VStack(matrices []*Matrix) (*Matrix, error) {
	res, err := func(matrices []*Matrix) (*Matrix, error) {
		if len(matrices) < 1 {
			return nil, fmt.Errorf("no matrices provided for vertical stacking")
		}

		vectors := m.Copy().vectors
		for k := 0; k < len(matrices); k++ {
			vectors = append(vectors, matrices[k].Copy().vectors...)
		}

		return NewMatrix(vectors)
	}(matrices)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Equal(matrix *Matrix) bool {
	if m.rows != matrix.rows {
		return false
	}

	for i := 0; i < m.rows; i++ {
		if !m.vectors[i].Equal(matrix.vectors[i]) {
			return false
		}
	}

	return true
}

func (m *Matrix) Order(indices []int) (*Matrix, error) {
	res, err := func(indices []int) (*Matrix, error) {
		if len(indices) != m.rows {
			return nil, fmt.Errorf("wrong indices count for ordering matrix %dx%d: %d != %d",
				m.rows, m.cols, len(indices), m.rows)
		}
		var present bool
		for i := 0; i < m.rows; i++ {
			present = false
			for j := 0; j < m.rows; j++ {
				present = indices[j] == i
				if present {
					break
				}
			}
			if !present {
				return nil, fmt.Errorf("order indices is not valid permutatuion of [0;%d), %d is omited", m.rows, i)
			}
		}

		vectors := make([]*vector.Vector, m.rows)
		for i, index := range indices {
			vectors[i] = m.vectors[index].Copy()
		}

		return NewMatrix(vectors)
	}(indices)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func CartesianProduct(vectors []*vector.Vector) (*Matrix, error) {
	res, err := func(vectors []*vector.Vector) (*Matrix, error) {
		if vectors == nil || len(vectors) < 1 {
			return nil, fmt.Errorf("no vectors provided for cartesian product: %v", vectors)
		}

		if len(vectors) == 1 {
			return NewMatrixFlat(vectors[0].Size(), 1, vectors[0])
		}

		cols := len(vectors)

		extendings := make([]int, cols)
		extendings[cols-1] = 1
		for i := cols - 2; i >= 0; i-- {
			extendings[i] = extendings[i+1] * vectors[i+1].Size()
		}

		var err error
		extended := make([]*vector.Vector, cols)
		for i, extend := range extendings {
			extended[i], err = vectors[i].Extend(extend)
			if err != nil {
				return nil, err
			}
		}

		stackings := make([]int, cols)
		stackings[0] = 1
		rows := extended[0].Size()
		for i := 1; i < cols; i++ {
			stackings[i] = rows / extended[i].Size()
		}

		stacked := make([]*vector.Vector, cols)
		for i, stack := range stackings {
			stacked[i], err = extended[i].Stack(stack)
			if err != nil {
				return nil, err
			}
		}

		matrix, err := NewMatrix(stacked)
		if err != nil {
			return nil, err
		}

		return matrix.T(), nil
	}(vectors)

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}
