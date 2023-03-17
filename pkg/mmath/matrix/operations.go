package matrix

import (
	"fmt"
	"math"
	"nn/pkg/mmath/vector"
	"nn/pkg/wraperr"
	"sync"
)

// UnaryOperation is function to apply to one Matrix
type UnaryOperation func(a float64) float64

// BinaryOperation is function to apply to two Matrix
type BinaryOperation func(a, b float64) float64

// T return transposed Matrix.
//
// Example:
//     | 1 2 |.T() = | 1 3 5 |
//     | 3 4 |       | 2 4 6 |
//     | 5 6 |
func (m *Matrix) T() *Matrix {
	if m == nil {
		return nil
	}
	cols := make([]*vector.Vector, m.cols)

	for j := 0; j < m.cols; j++ {
		col, _ := m.GetCol(j)
		cols[j] = col
	}

	matrix, _ := NewMatrix(cols)
	return matrix
}

const ParallelThreshold = 64

// MatMul perform matrix multiplication. This matrix cols count must match given matrix rows count.
//
// Throws ErrExec error.
//
// Example:
//     | 1 2 3 |.MatMul(| 7 |) = | 1*7 + 2*8 + 3*9 | = |  7 + 16 + 27 | = |  50 |
//     | 4 5 6 |        | 8 |    | 4*7 + 5*8 + 6*9 |   | 28 + 40 + 54 |   | 122 |
//                      | 9 |
func (m *Matrix) MatMul(matrix *Matrix) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if matrix == nil {
		return nil, fmt.Errorf("no second matrix provided for matrix multiplication: %v", matrix)
	} else if m.cols != matrix.rows {
		return nil, fmt.Errorf("can't Mul matrices sized %dx%d and %dx%d", m.rows, m.cols, matrix.rows, matrix.cols)
	}

	if m.rows*matrix.cols > ParallelThreshold {
		return m.matMulImplMulti(matrix)
	}
	return m.matMulImplSingle(matrix)
}

func (m *Matrix) matMulImplSingle(matrix *Matrix) (mat *Matrix, err error) {
	N, M := m.rows, matrix.cols

	rows := m.vectors
	cols := matrix.T().vectors

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
}

func (m *Matrix) matMulImplMulti(matrix *Matrix) (mat *Matrix, err error) {
	N, M := m.rows, matrix.cols

	rows := m.vectors
	cols := matrix.T().vectors

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
}

// Add return a + b
func Add(a, b float64) float64 {
	return a + b
}

// Sub return a - b
func Sub(a, b float64) float64 {
	return a - b
}

// Mul return a * b
func Mul(a, b float64) float64 {
	return a * b
}

// Div return a / b
func Div(a, b float64) float64 {
	return a / b
}

// ApplyFunc applies given UnaryOperation to this Matrix
//
// Example:
//     | -1  2 -3 |.ApplyFunc(math.Abs) = | 1 2 3 |
//     |  4 -5  6 |                       | 4 5 6 |
func (m *Matrix) ApplyFunc(operation UnaryOperation) *Matrix {
	if m == nil {
		return nil
	}
	vectors := make([]*vector.Vector, m.rows)
	wrap := func(a float64) float64 {
		return operation(a)
	} // wrap to conform vector.UnaryOperation

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

// ApplyFuncMat applies given BinaryOperation to this and given Matrix. Matrices must have same shape.
//
// Throws ErrExec error.
//
// Example:
//     | 1 2 |.ApplyFuncMat(| 5 6 |, Add) = | 1+2 2+6 | = |  3  8 |
//     | 3 4 |              | 7 8 |         | 3+7 4+8 |   | 10 12 |
func (m *Matrix) ApplyFuncMat(matrix *Matrix, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if matrix == nil {
		return nil, fmt.Errorf("no second matrix provided: %v", matrix)
	} else if matrix.Rows() != m.rows || matrix.Cols() != m.cols {
		return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, matrix.Rows(), matrix.Cols())
	}

	return m.apply(operation, func(i, j int) (float64, error) { return matrix.Get(i, j) })
}

// ApplyFuncNum applies given BinaryOperation to this and given float
//
// Example:
//     | 1 2 |.ApplyFuncNum(5, Add) = | 6 7 |
//     | 3 4 |                        | 8 9 |
func (m *Matrix) ApplyFuncNum(number float64, operation BinaryOperation) *Matrix {
	if m == nil {
		return nil
	}

	matrix, err := m.apply(operation, func(i, j int) (float64, error) { return number, nil })
	if err != nil {
		panic(err)
	}

	return matrix
}

// ApplyFuncMatRow applies given BinaryOperation to this and row as Matrix (row.Rows() == 1).
// This and row's cols count must match.
//
// Throws ErrExec error.
//
// Example:
//     | 1 2 |.ApplyFuncMatRow(| 5 6 |, Add) = | 1+5 2+6 | = | 6  8 |
//     | 3 4 |                                 | 3+5 4+6 |   | 8 10 |
func (m *Matrix) ApplyFuncMatRow(row *Matrix, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if row == nil {
		return nil, fmt.Errorf("no row provided: %v", row)
	} else if row.rows != 1 {
		return nil, fmt.Errorf("matrix is not row: %dx%d", row.rows, row.cols)
	} else if row.cols != m.cols {
		return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, row.rows, row.cols)
	}

	return m.apply(operation, func(i, j int) (float64, error) { return row.Get(0, j) })
}

// ApplyFuncMatCol applies given BinaryOperation to this and col as Matrix (col.Cols() == 1).
// This and col's rows count must match.
//
// Throws ErrExec error.
//
// Example:
//     | 1 2 |.ApplyFuncMatRow(| 5 |, Add) = | 1+5 2+5 | = | 6  7 |
//     | 3 4 |                 | 6 |         | 3+6 4+6 |   | 9 10 |
func (m *Matrix) ApplyFuncMatCol(col *Matrix, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if col == nil {
		return nil, fmt.Errorf("no col provided: %v", col)
	} else if col.cols != 1 {
		return nil, fmt.Errorf("matrix is not col: %dx%d", col.rows, col.cols)
	} else if col.rows != m.rows {
		return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, col.rows, col.cols)
	}
	return m.apply(operation, func(i, j int) (float64, error) { return col.Get(i, 0) })
}

// ApplyFuncVecRow is same as ApplyFuncMatCol
func (m *Matrix) ApplyFuncVecRow(row *vector.Vector, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if row == nil {
		return nil, fmt.Errorf("no row provided: %v", row)
	} else if row.Size() != m.cols {
		return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, row.Size(), 1)
	}

	return m.apply(operation, func(i, j int) (float64, error) { return row.Get(j) })
}

// ApplyFuncVecCol is same as ApplyFuncMatCol
func (m *Matrix) ApplyFuncVecCol(col *vector.Vector, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if col == nil {
		return nil, fmt.Errorf("no col provided: %v", col)
	} else if col.Size() != m.rows {
		return nil, fmt.Errorf("matrix size mismatces: %dx%d != %dx%d", m.rows, m.cols, col.Size(), 1)
	}

	return m.apply(operation, func(i, j int) (float64, error) { return col.Get(i) })
}

// See ApplyFuncMat and Add
func (m *Matrix) Add(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Add)
}

// See ApplyFuncMat and Sub
func (m *Matrix) Sub(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Sub)
}

// See ApplyFuncMat and Mul
func (m *Matrix) Mul(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Mul)
}

// See ApplyFuncMat and Div
func (m *Matrix) Div(matrix *Matrix) (*Matrix, error) {
	return m.ApplyFuncMat(matrix, Div)
}

// See ApplyFuncVecRow and Add
func (m *Matrix) AddRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Add)
}

// See ApplyFuncVecRow and Sub
func (m *Matrix) SubRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Sub)
}

// See ApplyFuncVecRow and Mul
func (m *Matrix) MulRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Mul)
}

// See ApplyFuncVecRow and Div
func (m *Matrix) DivRow(row *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecRow(row, Div)
}

// See ApplyFuncVecCol and Add
func (m *Matrix) AddCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Add)
}

// See ApplyFuncVecCol and Sub
func (m *Matrix) SubCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Sub)
}

// See ApplyFuncVecCol and Mul
func (m *Matrix) MulCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Mul)
}

// See ApplyFuncVecCol and Div
func (m *Matrix) DivCol(col *vector.Vector) (*Matrix, error) {
	return m.ApplyFuncVecCol(col, Div)
}

// See ApplyFuncMatRow and Add
func (m *Matrix) AddRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Add)
}

// See ApplyFuncMatRow and Sub
func (m *Matrix) SubRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Sub)
}

// See ApplyFuncMatRow and Mul
func (m *Matrix) MulRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Mul)
}

// See ApplyFuncMatRow and Div
func (m *Matrix) DivRowM(row *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatRow(row, Div)
}

// See ApplyFuncMatCol and Add
func (m *Matrix) AddColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Add)
}

// See ApplyFuncMatCol and Sub
func (m *Matrix) SubColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Sub)
}

// See ApplyFuncMatCol and Mul
func (m *Matrix) MulColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Mul)
}

// See ApplyFuncMatCol and Div
func (m *Matrix) DivColM(col *Matrix) (*Matrix, error) {
	return m.ApplyFuncMatCol(col, Div)
}

// See ApplyFuncNum and Add
func (m *Matrix) AddNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Add)
}

// See ApplyFuncNum and Sub
func (m *Matrix) SubNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Sub)
}

// See ApplyFuncNum and Mul
func (m *Matrix) MulNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Mul)
}

// See ApplyFuncNum and Div
func (m *Matrix) DivNum(number float64) *Matrix {
	return m.ApplyFuncNum(number, Div)
}

// Axis represents axis of axed operations: horizontal and vertical
type Axis uint8

const (
	Horizontal Axis = iota
	Vertical
)

// ReduceAxed reduce Matrix by sequentially applying given BinaryOperation to all values in given axis resulting in
// a Vector.
//
// Throws ErrExec error.
//
// Example:
//     | 1 2 3 |.ReduceAxed(Horizontal, Add) = | 1+2+3 | = |  6 |
//     | 4 5 6 |                               | 4+5+6 |   | 15 |
//
//     | 1 2 3 |.ReduceAxed(Vertical, Add) = | 1+4 2+5 3+6 | = | 5 7 9 |
//     | 4 5 6 |
func (m *Matrix) ReduceAxed(axis Axis, operation BinaryOperation) (vec *vector.Vector, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	}

	switch axis {
	case Horizontal:
		values := make([]float64, m.rows)
		for i, row := range m.vectors {
			values[i] = row.Reduce(func(a, b float64) float64 {
				return operation(a, b)
			})
		}
		return vector.NewVector(values)
	case Vertical:
		return m.T().ReduceAxed(Horizontal, operation)
	default:
		return nil, fmt.Errorf("unknown axis: %d", axis)
	}
}

// ReduceAxedM same as ReduceAxed but result is Matrix (not Vector)
func (m *Matrix) ReduceAxedM(axis Axis, operation BinaryOperation) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	}

	wrap := func(a, b float64) float64 {
		return operation(a, b)
	}

	switch axis {
	case Horizontal:
		values := make([]float64, m.rows)
		for i, row := range m.vectors {
			values[i] = row.Reduce(wrap)
		}
		return NewMatrixRawFlat(m.rows, 1, values)
	case Vertical:
		values := make([]float64, m.cols)
		t := m.T()
		for j, col := range t.vectors {
			values[j] = col.Reduce(wrap)
		}
		return NewMatrixRawFlat(1, m.cols, values)
	default:
		return nil, fmt.Errorf("unknown axis: %d", axis)
	}
}

// Reduce applies given BinaryOperation to Horizontal and Vertical axes for ReduceAxed resulting in a single float value
func (m *Matrix) Reduce(operation BinaryOperation) float64 {
	vec, _ := m.ReduceAxed(Horizontal, operation)
	return vec.Reduce(func(a, b float64) float64 {
		return operation(a, b)
	})
}

// See ReduceAxed and Add
func (m *Matrix) SumAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, Add)
}

// See ReduceAxedM and Add
func (m *Matrix) SumAxedM(axis Axis) (*Matrix, error) {
	return m.ReduceAxedM(axis, Add)
}

// See Reduce and Add
func (m *Matrix) Sum() float64 {
	return m.Reduce(Add)
}

// See ReduceAxed
func (m *Matrix) MaxAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, math.Max)
}

// See Reduce
func (m *Matrix) Max() float64 {
	return m.Reduce(math.Max)
}

// See ReduceAxed
func (m *Matrix) MinAxed(axis Axis) (*vector.Vector, error) {
	return m.ReduceAxed(axis, math.Min)
}

// See Reduce
func (m *Matrix) Min() float64 {
	return m.Reduce(math.Min)
}

// See SumAxed and DivNum
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

// See AvgAxed
func (m *Matrix) Avg() float64 {
	vec, _ := m.AvgAxed(Horizontal)
	return vec.Avg()
}

// See ApplyFunc
func (m *Matrix) Abs() *Matrix {
	return m.ApplyFunc(math.Abs)
}

// See ApplyFunc
func (m *Matrix) Exp() *Matrix {
	return m.ApplyFunc(math.Exp)
}

// See ApplyFunc
func (m *Matrix) Pow(scale float64) *Matrix {
	return m.ApplyFunc(func(a float64) float64 {
		return math.Pow(a, scale)
	})
}

// See Pow
func (m *Matrix) Sqr() *Matrix {
	return m.Pow(2)
}

// See Pow
func (m *Matrix) Sqrt() *Matrix {
	return m.Pow(1.0 / 2.0)
}

// See ApplyFunc
func (m *Matrix) Tanh() *Matrix {
	return m.ApplyFunc(math.Tanh)
}

// SubMatrix returns sub matrix similar to vector.Slice. All inputs must be correct non-zero values (*Start < *Stop).
// Result must have at least 1 row and 1 col. *Start index is included, *Stop index is excluded.
//
// Throws ErrExec error.
//
// Example:
//     |  1  2  3  4  5  6  7  8 |.SubMatrix(0, 2, 1, 1, 6, 2) = |  2  4  6 |
//     |  9 10 11 12 13 14 15 16 |                               | 10 12 14 |
//     | 17 18 19 20 21 21 22 23 |
func (m *Matrix) SubMatrix(rowsStart, rowsStop, rowsStep, colsStart, colsStop, colsStep int) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	l := rowsStop - rowsStart
	resLen := int(math.Ceil(float64(l) / float64(rowsStep)))
	if m == nil {
		return nil, ErrNil
	} else if l < 1 {
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
}

// HStack stacks this and given matrices by Horizontal Axis. This and given matrices must have same rows count.
//
// Throws ErrExec error
//
// Example:
//     | 1 2 |.HStack([| 5 |, |  7  8  9 |]) = | 1 2 5  7  8  9 |
//     | 3 4 |         | 6 |  | 10 11 12 |     | 3 4 6 10 11 12 |
func (m *Matrix) HStack(matrices []*Matrix) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if len(matrices) < 1 {
		return nil, fmt.Errorf("no matrices provided for horizontal stacking")
	}

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
}

// VStack stacks this and given matrices by Vertical Axis. This and given matrices must have same cols count.
//
// Throws ErrExec error
//
// Example:
//     | 1 2 |.VStack([| 5 6 |, |  7  8 |]) = |  1  2 |
//     | 3 4 |                  |  9 10 |     |  3  4 |
//                              | 11 12 |     |  5  6 |
//                                            |  7  8 |
//                                            |  9 10 |
//                                            | 11 12 |
func (m *Matrix) VStack(matrices []*Matrix) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if len(matrices) < 1 {
		return nil, fmt.Errorf("no matrices provided for vertical stacking")
	}

	vectors := m.Copy().vectors
	for k := 0; k < len(matrices); k++ {
		vectors = append(vectors, matrices[k].Copy().vectors...)
	}

	return NewMatrix(vectors)
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

// Order preform row ordering for given indices (permutation of [0; rows).
//
// Throws ErrExec error.
//
// Example:
//     | 1 |.Order([2 1 0 3]) = | 3 |
//     | 2 |                    | 2 |
//     | 3 |                    | 1 |
//     | 4 |                    | 4 |
func (m *Matrix) Order(indices []int) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if m == nil {
		return nil, ErrNil
	} else if indices == nil {
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
}

// CartesianProduct perform cartesian multiplication. There is no restrictions to size of vectors.
// Result of product is Matrix with rows count == product of lengths of all vectors and cols count == count of vectors.
//
// Throws ErrExec error.
//
// Example:
//     CartesianProduct([| 1 |, | 3 |, | 4 |]) = | 1 3 4 |
//                       | 2 |         | 5 |     | 1 3 5 |
//                                     | 6 |     | 1 3 6 |
//                                               | 2 3 4 |
//                                               | 2 3 5 |
//                                               | 2 3 6 |
func CartesianProduct(vectors []*vector.Vector) (mat *Matrix, err error) {
	defer wraperr.WrapError(ErrExec, &err)

	if len(vectors) < 1 {
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

	mat, err = NewMatrix(stacked)
	if err != nil {
		return nil, err
	}

	return mat.T(), nil
}
