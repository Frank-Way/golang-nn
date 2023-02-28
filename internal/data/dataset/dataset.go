package dataset

import (
	"fmt"
	"math"
	"nn/internal/utils"
	"nn/pkg/wraperr"
	"strings"
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

type DataSplitParameters struct {
	TrainPercent Percent
	TestsPercent Percent
	ValidPercent Percent
}

var DefaultDataSplitParameters = &DataSplitParameters{
	TrainPercent: Percent60,
	TestsPercent: Percent30,
	ValidPercent: Percent10,
}

func (p *DataSplitParameters) Copy() *DataSplitParameters {
	return &DataSplitParameters{
		TrainPercent: p.TrainPercent,
		TestsPercent: p.TestsPercent,
		ValidPercent: p.ValidPercent,
	}
}

func NewDatasetSplit(data *Data, parameters *DataSplitParameters) (*Dataset, error) {
	res, err := func(data *Data, parameters *DataSplitParameters) (*Dataset, error) {
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
		if math.Abs(float64(validSize-valid.X.Rows())) > Percent10.GetF(float64(n)) {
			return nil,
				fmt.Errorf("desired and actual valid part sizes mismathes too much: %d != %d", validSize, valid.X.Rows())
		}

		return NewDataset(train, tests, valid)
	}(data, parameters)

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
	if dataset == nil {
		return false
	}
	return d.Train.Equal(dataset.Train) && d.Tests.Equal(dataset.Tests) && d.Valid.Equal(dataset.Valid)
}

func (d *Dataset) String() string {
	return fmt.Sprintf("{Train: %s, Tests: %v, Valid: %v}", d.Train.String(), d.Tests.String(), d.Valid.String())
}

func (d *Dataset) PrettyString() string {
	var sb strings.Builder
	sb.WriteString(utils.PrettyString("Train", d.Train) + "\n")
	sb.WriteString(utils.PrettyString("Tests", d.Tests) + "\n")
	sb.WriteString(utils.PrettyString("Valid", d.Valid))

	return sb.String()
}

func (d *Dataset) ShortString() string {
	return fmt.Sprintf("{Train: %s, Tests: %v, Valid: %v}",
		d.Train.ShortString(), d.Tests.ShortString(), d.Valid.ShortString())
}