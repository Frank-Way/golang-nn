package dataset

import (
	"fmt"
	"math/rand"
	"nn/internal/utils"
	"nn/pkg/mmath/matrix"
	"nn/pkg/wraperr"
	"strings"
)

type Data struct {
	X *matrix.Matrix
	Y *matrix.Matrix
}

func NewData(x *matrix.Matrix, y *matrix.Matrix) (*Data, error) {
	var err error
	if x == nil {
		err = fmt.Errorf("no inputs provided: %v", x)
	} else if y == nil {
		err = fmt.Errorf("no outputs provided: %v", y)
	} else if x.Rows() != y.Rows() {
		err = fmt.Errorf("rows count mismatches in inputs and outputs: %d != %d", x.Rows(), y.Rows())
	}
	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return &Data{X: x, Y: y}, nil
}

func (d *Data) Copy() *Data {
	data, err := NewData(d.X.Copy(), d.Y.Copy())
	if err != nil {
		panic(err)
	}

	return data
}

func (d *Data) String() string {
	return fmt.Sprintf("{X: %s, Y: %s}", d.X.String(), d.Y.String())
}

func (d *Data) PrettyString() string {
	var sb strings.Builder
	sb.WriteString(utils.PrettyString("X", d.X) + "\n")
	sb.WriteString(utils.PrettyString("Y", d.Y))
	return sb.String()
}

func (d *Data) ShortString() string {
	return fmt.Sprintf("{X: %dx%d, Y: %dx%d}", d.X.Rows(), d.X.Cols(), d.Y.Rows(), d.Y.Cols())
}

func (d *Data) Shuffle() (*Data, []int) {
	perm := rand.Perm(d.X.Rows())

	var err error
	var xOrdered, yOrdered *matrix.Matrix
	if xOrdered, err = d.X.Order(perm); err != nil {
		panic(err)
	}
	if yOrdered, err = d.Y.Order(perm); err != nil {
		panic(err)
	}

	if data, err := NewData(xOrdered, yOrdered); err != nil {
		panic(err)
	} else {
		return data, perm
	}
}

func (d *Data) Equal(data *Data) bool {
	if data == nil {
		return false
	} else if d.X.Rows() != data.X.Rows() {
		return false
	} else if !d.X.Equal(data.X) {
		return false
	}

	return d.Y.Equal(data.Y)
}

// Split return two *Data, first contains values in [0; pivot), second - in [pivot; Data.Rows()).
// Split is a row-based function.
// If fail, Split returns ErrSplit error wrapper.
func (d *Data) Split(pivot int) (*Data, *Data, error) {
	res1, res2, err := func(pivot int) (*Data, *Data, error) {
		if pivot < 1 {
			return nil, nil, fmt.Errorf("negative or zero split pivot for data")
		} else if d.X.Rows()-pivot < 1 {
			return nil, nil, fmt.Errorf("not enough values for split data sized %d with pivot %d", d.X.Rows(), pivot)
		}

		var firstX, firstY, secondX, secondY *matrix.Matrix
		var err error
		if firstX, err = d.X.SubMatrix(0, pivot, 1, 0, d.X.Cols(), 1); err != nil {
			return nil, nil, err
		}
		if secondX, err = d.X.SubMatrix(pivot, d.X.Rows(), 1, 0, d.X.Cols(), 1); err != nil {
			return nil, nil, err
		}
		if firstY, err = d.Y.SubMatrix(0, pivot, 1, 0, d.Y.Cols(), 1); err != nil {
			return nil, nil, err
		}
		if secondY, err = d.Y.SubMatrix(pivot, d.Y.Rows(), 1, 0, d.Y.Cols(), 1); err != nil {
			return nil, nil, err
		}

		var first, second *Data
		if first, err = NewData(firstX, firstY); err != nil {
			return nil, nil, err
		}
		if second, err = NewData(secondX, secondY); err != nil {
			return nil, nil, err
		}

		return first, second, nil
	}(pivot)

	if err != nil {
		return nil, nil, wraperr.NewWrapErr(ErrSplit, err)
	}

	return res1, res2, nil
}
