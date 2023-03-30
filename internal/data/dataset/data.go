// Package dataset provides functionality for Data and Dataset to store data for training neural network
package dataset

import (
	"fmt"
	"math/rand"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
)

// Data holds inputs and corresponding outputs
type Data struct {
	X *matrix.Matrix
	Y *matrix.Matrix
}

// NewData checks given Matrix and creates Data. Inputs and outputs must have same rows count.
//
// Throws ErrCreate error.
func NewData(x *matrix.Matrix, y *matrix.Matrix) (data *Data, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrCreate, &err)

	logger.Infof("create data from x %q and y %q", x.ShortString(), y.ShortString())
	if x == nil {
		return nil, fmt.Errorf("no inputs provided: %v", x)
	} else if y == nil {
		return nil, fmt.Errorf("no outputs provided: %v", y)
	} else if x.Rows() != y.Rows() {
		return nil, fmt.Errorf("rows count mismatches in inputs and outputs: %d != %d", x.Rows(), y.Rows())
	}

	data = &Data{X: x, Y: y}
	logger.Tracef("created data: %s", data.ShortString())
	return data, nil
}

// Copy return deep copy of Data
func (d *Data) Copy() *Data {
	logger.Tracef("copy data %q", d.ShortString())
	data, err := NewData(d.X.Copy(), d.Y.Copy())
	if err != nil {
		panic(err)
	}

	return data
}

func (d *Data) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"x": stringer(d.X),
		"y": stringer(d.Y),
	}

}

func (d *Data) String() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.String), utils.BaseFormat)
}

func (d *Data) PrettyString() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (d *Data) ShortString() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.ShortString), utils.ShortFormat)
}

// Shuffle mixes data. Shuffle is row-based. Inputs and outputs reordering using the same random permutation.
// Shuffle creates new Data, source Data stay untouched.
//
// Example:
//     {X: | 1 |, Y: | 4 |}.Shuffle() = {X: | 3 |, Y: | 6 |}, [2 0 1]
//         | 2 |     | 5 |                  | 1 |     | 4 |
//         | 3 |     | 6 |                  | 2 |     | 5 |
func (d *Data) Shuffle() (data *Data, perm []int) {
	logger.Infof("shuffle data: %s", d.ShortString())
	perm = rand.Perm(d.X.Rows())
	logger.Tracef("permutation: %v", perm)

	var err error
	var xOrdered, yOrdered *matrix.Matrix
	if xOrdered, err = d.X.Order(perm); err != nil {
		panic(err)
	}
	if yOrdered, err = d.Y.Order(perm); err != nil {
		panic(err)
	}

	if data, err = NewData(xOrdered, yOrdered); err != nil {
		panic(err)
	} else {
		logger.Infof("shuffled data: %s", data.ShortString())
		return data, perm
	}
}

func (d *Data) Equal(data *Data) bool {
	if d == nil || data == nil {
		if (d != nil && data == nil) || (d == nil && data != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if d.X.Rows() != data.X.Rows() {
		return false
	} else if !d.X.Equal(data.X) {
		return false
	}

	return d.X.Equal(data.X)
}

func (d *Data) EqualApprox(data *Data) bool {
	if d == nil || data == nil {
		if (d != nil && data == nil) || (d == nil && data != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	} else if d.X.Rows() != data.X.Rows() {
		return false
	} else if !d.X.EqualApprox(data.X) {
		return false
	}

	return d.Y.EqualApprox(data.Y)
}

// Split return two Data, first contains values in [0; pivot), second - in [pivot; Data.Rows()).
// Split is a row-based.
//
// Throws ErrSplit error.
//
// Example:
//     {X: |  1 |, Y: |  5 |}.Split(2) = [{X: | 1 |, Y: | 5 |}, {X: |  3 |, Y: |  7 |}]
//         |  2 |     |  6 |                  | 2 |     | 6 |       |  4 |     |  8 |
//         |  3 |     |  7 |                                        | 44 |     | 88 |
//         |  4 |     |  8 |
//         | 44 |     | 88 |
func (d *Data) Split(pivot int) (first *Data, second *Data, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrSplit, &err)

	logger.Infof("split data %q by pivot %d", d.ShortString(), pivot)
	if pivot < 1 {
		return nil, nil, fmt.Errorf("negative or zero split pivot for data")
	} else if d.X.Rows()-pivot < 1 {
		return nil, nil, fmt.Errorf("not enough values for split data sized %d with pivot %d", d.X.Rows(), pivot)
	}

	var firstX, firstY, secondX, secondY *matrix.Matrix

	if firstX, err = d.X.SubMatrix(0, pivot, 1, 0, d.X.Cols(), 1); err != nil {
		return nil, nil, fmt.Errorf("error getting first part from inputs: %w", err)
	}
	if secondX, err = d.X.SubMatrix(pivot, d.X.Rows(), 1, 0, d.X.Cols(), 1); err != nil {
		return nil, nil, fmt.Errorf("error getting second part from inputs: %w", err)
	}
	if firstY, err = d.Y.SubMatrix(0, pivot, 1, 0, d.Y.Cols(), 1); err != nil {
		return nil, nil, fmt.Errorf("error getting first part from outputs: %w", err)
	}
	if secondY, err = d.Y.SubMatrix(pivot, d.Y.Rows(), 1, 0, d.Y.Cols(), 1); err != nil {
		return nil, nil, fmt.Errorf("error getting second part from outputs: %w", err)
	}

	if first, err = NewData(firstX, firstY); err != nil {
		return nil, nil, fmt.Errorf("error getting data from first parts of inputs and outputs: %w", err)
	}
	if second, err = NewData(secondX, secondY); err != nil {
		return nil, nil, fmt.Errorf("error getting data from second parts of inputs and outputs: %w", err)
	}

	logger.Tracef("first part: %s", first.ShortString())
	logger.Tracef("second part: %s", second.ShortString())

	return first, second, nil
}

// Batches return closure over Data split in batches with given size. Last batch size may be less then <batchSize>.
// Generator may be overused any times causing no actual data-splitting logic (e.g. no cost). Second return value is
// max index of generator to use it in `for-loop`.
//
// Throws ErrSplit error.
func (d *Data) Batches(batchSize int) (generator func(i int) (*Data, error), count int, err error) {
	defer logger.CatchErr(&err)
	defer wraperr.WrapError(ErrSplit, &err)

	if batchSize < 1 {
		return nil, 0, fmt.Errorf("negative or zero batch size: %d", batchSize)
	}

	count = d.X.Rows() / batchSize
	if d.X.Rows()%batchSize != 0 {
		count++
	}

	batches := make([]*Data, count)
	var first, second *Data
	first = d
	for i := 0; i < count; i++ {
		if first.X.Rows() <= batchSize {
			batches[i] = first
			break
		}
		first, second, err = first.Split(batchSize)
		if err != nil {
			return nil, 0, fmt.Errorf("error getting %d'th batch: %w", i, err)
		}

		batches[i] = first
		first, second = second, nil
	}

	return func(i int) (*Data, error) {
		if i < 0 || i >= len(batches) {
			return nil, fmt.Errorf("wrong batch index for %d batches: %d", len(batches), i)
		}
		return batches[i], nil // closure over batches from outer scope
	}, count, nil
}
