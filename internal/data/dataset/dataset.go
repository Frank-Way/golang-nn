package dataset

import (
	"fmt"
	"math"
	"nn/internal/utils"
	"nn/pkg/percent"
	"nn/pkg/wraperr"
)

type Dataset struct {
	Train *Data
	Tests *Data
	Valid *Data
}

func NewDataset(train *Data, tests *Data, valid *Data) (*Dataset, error) {
	var err error
	if train == nil {
		err = fmt.Errorf("no train data: %v", train)
	} else if tests == nil {
		err = fmt.Errorf("no tests data: %v", tests)
	} else if valid == nil {
		err = fmt.Errorf("no valid data: %v", valid)
	} else if train.X.Cols() != tests.X.Cols() {
		err = fmt.Errorf("cols count in train and tests input data mismatches: %d != %d", train.X.Cols(), tests.X.Cols())
	} else if tests.X.Cols() != valid.X.Cols() {
		err = fmt.Errorf("cols count in tests and valid input data mismatches: %d != %d", tests.X.Cols(), valid.X.Cols())
	} else if train.Y.Cols() != tests.Y.Cols() {
		err = fmt.Errorf("cols count in train and tests output data mismatches: %d != %d", train.Y.Cols(), tests.Y.Cols())
	} else if tests.Y.Cols() != valid.Y.Cols() {
		err = fmt.Errorf("cols count in tests and valid output data mismatches: %d != %d", tests.Y.Cols(), valid.Y.Cols())
	}
	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return &Dataset{Train: train, Tests: tests, Valid: valid}, nil
}

func NewDatasetSplit(data *Data, parameters *DataSplitParameters) (*Dataset, error) {
	res, err := func() (*Dataset, error) {
		if data == nil {
			return nil, fmt.Errorf("no data provided for splitting: %v", data)
		} else if parameters == nil {
			return nil, fmt.Errorf("no parameters provided for splitting: %v", parameters)
		}
		n := data.X.Rows()
		trainSize := parameters.TrainPercent.GetI(n)
		train, others, err := data.Split(trainSize)
		if err != nil {
			return nil, fmt.Errorf("error during splitting to get traing data: %w", err)
		}

		testSize := parameters.TestsPercent.GetI(n)
		tests, valid, err := others.Split(testSize)
		if err != nil {
			return nil, fmt.Errorf("error during splitting to get tests and valid data: %w", err)
		}

		validSize := parameters.ValidPercent.GetI(n)
		if math.Abs(float64(validSize-valid.X.Rows())) > percent.Percent10.GetF(float64(n)) {
			return nil,
				fmt.Errorf("desired and actual valid part sizes mismathes too much: %d != %d", validSize, valid.X.Rows())
		}

		return NewDataset(train, tests, valid)
	}()

	if err != nil {
		return nil, wraperr.NewWrapErr(ErrCreate, err)
	}

	return res, nil
}

func (d *Dataset) Copy() *Dataset {
	dataset, err := NewDataset(d.Train.Copy(), d.Tests.Copy(), d.Valid.Copy())
	if err != nil {
		panic(err)
	}

	return dataset
}

func (d *Dataset) Equal(dataset *Dataset) bool {
	if d == nil || dataset == nil {
		if (d != nil && dataset == nil) || (d == nil && dataset != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

	return d.Train.Equal(dataset.Train) && d.Tests.Equal(dataset.Tests) && d.Valid.Equal(dataset.Valid)
}

func (d *Dataset) EqualApprox(dataset *Dataset) bool {
	if d == nil || dataset == nil {
		if (d != nil && dataset == nil) || (d == nil && dataset != nil) {
			return false // non-nil != nil and nil != non-nil
		} else {
			return true // nil == nil
		}
	}

	return d.Train.EqualApprox(dataset.Train) && d.Tests.EqualApprox(dataset.Tests) && d.Valid.EqualApprox(dataset.Valid)
}

func (d *Dataset) toMap(stringer func(spStringer utils.SPStringer) string) map[string]string {
	return map[string]string{
		"Train": stringer(d.Train),
		"Tests": stringer(d.Tests),
		"Valid": stringer(d.Valid),
	}
}

func (d *Dataset) String() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.String), utils.BaseFormat)
}

func (d *Dataset) PrettyString() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.PrettyString), utils.PrettyFormat)
}

func (d *Dataset) ShortString() string {
	if d == nil {
		return "<nil>"
	}
	return utils.FormatObject(d.toMap(utils.ShortString), utils.ShortFormat)
}
