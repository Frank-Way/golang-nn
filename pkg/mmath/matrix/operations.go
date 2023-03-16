package matrix

import (
	"fmt"
	"math"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
	"sync"
)

type UnaryOperation func(a float64) float64
type BinaryOperation func(a, b float64) float64

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
	if matrix == nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec,
			fmt.Errorf("no second matrix provided for matrix multiplication: %v", matrix))
	}
	if m.rows*matrix.cols > ParallelThreshold {
		return m.matMulImplMulti(matrix)
	}
	return m.matMulImplSingle(matrix)
}

func (m *Matrix) matMulImplSingle(matrix *Matrix) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		N, M := m.rows, matrix.cols
		if m.cols != matrix.rows {
			return nil, fmt.Errorf("can't Mul matrices sized %dx%d and %dx%d", m.rows, m.cols, matrix.rows, matrix.cols)
		}

		rows := m.vectors
		cols := matrix.T().vectors

		values := make([][]float64, N)
		var value float64
		var rawRow []float64
		var err error
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
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) matMulImplMulti(matrix *Matrix) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		N, M := m.rows, matrix.cols
		if m.cols != matrix.rows {
			return nil, fmt.Errorf("can't Mul matrices sized %dx%d and %dx%d", m.rows, m.cols, matrix.rows, matrix.cols)
		}

		rows := m.vectors
		cols := matrix.T().vectors

		values := make([][]float64, N)
		var value float64
		var err error
		var rawRow []float64
		wg := sync.WaitGroup{}
		for i, row := range rows {
			wg.Add(1)
			rawRow = make([]float64, M)
			values[i] = rawRow
			go func(row *vector.Vector, arr []float64) {
				defer wg.Done()
				for j, col := range cols {
					value, err = row.MulScalar(col)
					if err != nil {
						panic(err)
					}
					arr[j] = value
				}
			}(row, rawRow)
		}

		wg.Wait()
		return NewMatrixRaw(values)
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func Add(a, b float64) float64 {
	return a + b
}

func Sub(a, b float64) float64 {
	return a - b
}

func Mul(a, b float64) float64 {
	return a * b
}

func Div(a, b float64) float64 {
	return a / b
}

func (m *Matrix) ApplyFunc(operation UnaryOperation) *Matrix {
	vectors := make([]*vector.Vector, m.rows)
	wrap := func(a float64) float64 {
		return operation(a)
	}

	for i, row := range m.vectors {
		vectors[i] = row.ApplyFunc(wrap)
	}

	matrix, err := NewMatrix(vectors)
	if err != nil {
		panic(err)
	}

	return matrix
}

func (m *Matrix) apply(operation BinaryOperation, provider func(row, col int) (float64, error)) (*Matrix, error) {
	values := make([][]float64, m.rows)
	for i := 0; i < m.rows; i++ {
		rawRow := make([]float64, m.cols)
		for j := 0; j < m.cols; j++ {
			a, _ := m.Get(i, j)
			b, err := provider(i, j)
			if err != nil {
				return nil, err
			}
			rawRow[j] = operation(a, b)
		}
		values[i] = rawRow
	}

	return NewMatrixRaw(values)
}

func (m *Matrix) ApplyFuncMat(matrix *Matrix, operation BinaryOperation) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if matrix == nil {
			return nil, fmt.Errorf("no second matrix provided: %v", matrix)
		} else if matrix.Rows() != m.rows || matrix.Cols() != m.cols {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, matrix.Rows(), matrix.Cols())
		}
		return m.apply(operation, func(i, j int) (float64, error) { return matrix.Get(i, j) })
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) ApplyFuncNum(number float64, operation BinaryOperation) *Matrix {
	matrix, err := m.apply(operation, func(i, j int) (float64, error) { return number, nil })
	if err != nil {
		panic(err)
	}

	return matrix
}

func (m *Matrix) ApplyFuncMatRow(row *Matrix, operation BinaryOperation) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if row == nil {
			return nil, fmt.Errorf("no row provided: %v", row)
		} else if row.rows != 1 {
			return nil, fmt.Errorf("matrix is not row: %dx%d", row.rows, row.cols)
		} else if row.cols != m.cols {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, row.rows, row.cols)
		}
		return m.apply(operation, func(i, j int) (float64, error) { return row.Get(0, j) })
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) ApplyFuncMatCol(col *Matrix, operation BinaryOperation) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if col == nil {
			return nil, fmt.Errorf("no col provided: %v", col)
		} else if col.cols != 1 {
			return nil, fmt.Errorf("matrix is not col: %dx%d", col.rows, col.cols)
		} else if col.rows != m.rows {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, col.rows, col.cols)
		}
		return m.apply(operation, func(i, j int) (float64, error) { return col.Get(i, 0) })
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) ApplyFuncVecRow(row *vector.Vector, operation BinaryOperation) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if row == nil {
			return nil, fmt.Errorf("no row provided: %v", row)
		} else if row.Size() != m.cols {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, row.Size(), 1)
		}
		return m.apply(operation, func(i, j int) (float64, error) { return row.Get(j) })
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) ApplyFuncVecCol(col *vector.Vector, operation BinaryOperation) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if col == nil {
			return nil, fmt.Errorf("no col provided: %v", col)
		} else if col.Size() != m.rows {
			return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, col.Size(), 1)
		}
		return m.apply(operation, func(i, j int) (float64, error) { return col.Get(i) })
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Add(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Add)
}

func (m *Matrix) Sub(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Sub)
}

func (m *Matrix) Mul(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Mul)
}

func (m *Matrix) Div(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Div)
}

func (m *Matrix) AddRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Add)
}

func (m *Matrix) SubRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Sub)
}

func (m *Matrix) MulRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Mul)
}

func (m *Matrix) DivRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Div)
}

func (m *Matrix) AddCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Add)
}

func (m *Matrix) SubCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Sub)
}

func (m *Matrix) MulCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Mul)
}

func (m *Matrix) DivCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Div)
}

func (m *Matrix) AddRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Add)
}

func (m *Matrix) SubRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Sub)
}

func (m *Matrix) MulRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Mul)
}

func (m *Matrix) DivRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Div)
}

func (m *Matrix) AddColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Add)
}

func (m *Matrix) SubColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Sub)
}

func (m *Matrix) MulColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Mul)
}

func (m *Matrix) DivColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Div)
}

func (m *Matrix) AddNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Add)
}

func (m *Matrix) SubNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Sub)
}

func (m *Matrix) MulNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Mul)
}

func (m *Matrix) DivNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Div)
}

type Axis uint8

const (
	Horizontal Axis = iota
	Vertical
)

func (m *Matrix) ReduceAxed(axis Axis, operation func(a, b float64) float64) (*vector.Vector, error) {
	res, err := func() (*vector.Vector, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Reduce(operation)
			}
			return vector.NewVector(values)
		case Vertical:
			return m.T().ReduceAxed(Horizontal, operation)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) ReduceAxedM(axis Axis, operation func(a, b float64) float64) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		switch axis {
		case Horizontal:
			values := make([]float64, m.rows)
			for i, row := range m.vectors {
				values[i] = row.Reduce(operation)
			}
			return NewMatrixRawFlat(m.rows, 1, values)
		case Vertical:
			values := make([]float64, m.cols)
			t := m.T()
			for j, col := range t.vectors {
				values[j] = col.Reduce(operation)
			}
			return NewMatrixRawFlat(1, m.cols, values)
		default:
			return nil, fmt.Errorf("unknown axis: %d", axis)
		}
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Reduce(operation func(a, b float64) float64) float64 {
	vec, _ := m.ReduceAxed(Horizontal, operation)
	return vec.Reduce(operation)
}

func (m *Matrix) SumAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, Add)
}

func (m *Matrix) SumAxedM(axis Axis) (*Matrix, error) {
	return m.ReduceAxedM(axis, Add)
}

func (m *Matrix) Sum() float64 {
	return m.Reduce(Add)
}

func (m *Matrix) MaxAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, math.Max)
}

func (m *Matrix) Max() float64 {
	return m.Reduce(math.Max)
}

func (m *Matrix) MinAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, math.Min)
}

func (m *Matrix) Min() float64 {
	return m.Reduce(math.Min)
}

func (m *Matrix) AvgAxed(axis Axis) (*vector.Vector, error) {
	sum, err := m.SumAxed(axis)
	if err != nil {
		return nil, err
	}

	divisor := m.cols
	if axis == Vertical {
		divisor = m.rows
	}

	return sum.DivNum(float64(divisor)), nil
}

func (m *Matrix) Avg() float64 {
	vec, _ := m.AvgAxed(Horizontal)
	return vec.Avg()
}

func (m *Matrix) Abs() *Matrix {
	return m.ApplyFunc(math.Abs)
}

func (m *Matrix) Exp() *Matrix {
	return m.ApplyFunc(math.Exp)
}

func (m *Matrix) Pow(scale float64) *Matrix {
	return m.ApplyFunc(func(a float64) float64 {
		return math.Pow(a, scale)
	})
}

func (m *Matrix) Sqr() *Matrix {
	return m.Pow(2)
}

func (m *Matrix) Sqrt() *Matrix {
	return m.Pow(1.0 / 2.0)
}

func (m *Matrix) Tanh() *Matrix {
	return m.ApplyFunc(math.Tanh)
}

func (m *Matrix) SubMatrix(rowsStart, rowsStop, rowsStep, colsStart, colsStop, colsStep int) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		l := rowsStop - rowsStart
		resLen := int(math.Ceil(float64(l) / float64(rowsStep)))
		if l < 1 {
			return nil, fmt.Errorf("wrong start and stop row indicies for Sub matrix: %d >= %d", rowsStart, rowsStop)
		} else if rowsStart < 0 {
			return nil, fmt.Errorf("negative start row index for Sub matrix: %d", rowsStart)
		} else if rowsStop < 1 {
			return nil, fmt.Errorf("negative or zero stop row index for Sub matrix: %d", rowsStop)
		} else if rowsStep < 1 {
			return nil, fmt.Errorf("negative or zero row step for Sub matrix: %d", rowsStep)
		} else if m.rows < rowsStop {
			return nil, fmt.Errorf("stop row index for Sub matrix out of matrix rows count: %d > %d", rowsStop, m.rows)
		} else if resLen < 1 {
			return nil, fmt.Errorf("zero resulting Sub matrix length with start %d stop %d step %d rows %d", rowsStart, rowsStop, rowsStep, m.rows)
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
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) HStack(matrices []*Matrix) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if matrices == nil || len(matrices) < 1 {
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
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) VStack(matrices []*Matrix) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if matrices == nil || len(matrices) < 1 {
			return nil, fmt.Errorf("no matrices provided for vertical stacking")
		}

		vectors := m.Copy().vectors
		for k := 0; k < len(matrices); k++ {
			vectors = append(vectors, matrices[k].Copy().vectors...)
		}

		return NewMatrix(vectors)
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func (m *Matrix) Equal(matrix *Matrix) bool {
	if m == nil || matrix == nil {
		if (m != nil && matrix == nil) || (m == nil && matrix != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if m.rows != matrix.rows {
		return false
	}

	for i := 0; i < m.rows; i++ {
		if !m.vectors[i].Equal(matrix.vectors[i]) {
			return false
		}
	}

	return true
}

func (m *Matrix) EqualApprox(matrix *Matrix) bool {
	if m == nil || matrix == nil {
		if (m != nil && matrix == nil) || (m == nil && matrix != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if m.rows != matrix.rows {
		return false
	}

	for i := 0; i < m.rows; i++ {
		if !m.vectors[i].EqualApprox(matrix.vectors[i]) {
			return false
		}
	}

	return true
}

func (m *Matrix) Order(indices []int) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
		if indices == nil {
			return nil, fmt.Errorf("no indices provided for ordering matrix: %v", indices)
		} else if len(indices) != m.rows {
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
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}

func CartesianProduct(vectors []*vector.Vector) (*Matrix, error) {
	res, err := func() (*Matrix, error) {
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
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrOperationExec, err)
	}

	return res, nil
}
