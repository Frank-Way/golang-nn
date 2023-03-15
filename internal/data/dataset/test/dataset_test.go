package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"nn/internal/testutils"
	"nn/internal/testutils/fabrics"
	"nn/pkg/percent"
	"testing"
)

func TestNewDataset(t *testing.T) {
	tests := []struct {
		testutils.Base
		train         fabrics.DataParameters
		tests_        fabrics.DataParameters
		valid         fabrics.DataParameters
		nilCheckTrain bool
		nilCheckTests bool
		nilCheckValid bool
	}{
		{
			Base:   testutils.Base{Name: "valid parameters"},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:          testutils.Base{Name: "nil train", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckTrain: true,
		},
		{
			Base:          testutils.Base{Name: "nil tests", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckTests: true,
		},
		{
			Base:          testutils.Base{Name: "nil valid", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckValid: true,
		},
		{
			Base:   testutils.Base{Name: "train and tests inputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 2}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   testutils.Base{Name: "train and tests outputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 2}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   testutils.Base{Name: "tests and valid inputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 2}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   testutils.Base{Name: "tests and valid outputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 2}},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			train := fabrics.NewData(t, test.train)
			tests_ := fabrics.NewData(t, test.tests_)
			valid := fabrics.NewData(t, test.valid)
			if test.nilCheckTrain {
				train = nil
			}
			if test.nilCheckTests {
				tests_ = nil
			}
			if test.nilCheckValid {
				valid = nil
			}

			_, err := dataset.NewDataset(train, tests_, valid)
			if test.Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestNewDatasetSplit(t *testing.T) {
	tests := []struct {
		testutils.Base
		single        fabrics.DataParameters
		params        *dataset.DataSplitParameters
		expectedTrain int
		expectedTests int
		expectedValid int
	}{
		{
			Base:          testutils.Base{Name: "enough values, default parameters"},
			single:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params:        dataset.DefaultDataSplitParameters,
			expectedTrain: 60,
			expectedTests: 30,
			expectedValid: 10,
		},
		{
			Base:   testutils.Base{Name: "not enough values", Err: dataset.ErrCreate},
			single: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params: &dataset.DataSplitParameters{TrainPercent: percent.Percent50, TestsPercent: percent.Percent50, ValidPercent: percent.Percent50},
		},
		{
			Base:   testutils.Base{Name: "too many values", Err: dataset.ErrCreate},
			single: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params: &dataset.DataSplitParameters{TrainPercent: percent.Percent10, TestsPercent: percent.Percent10, ValidPercent: percent.Percent10},
		},
	}

	for i, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			data := fabrics.NewData(t, test.single)

			ds, err := dataset.NewDatasetSplit(data, test.params)
			if test.Err == nil {
				require.NoError(t, err)
				require.Equal(t, test.expectedTrain, ds.Train.X.Rows())
				require.Equal(t, test.expectedTrain, ds.Train.Y.Rows())
				require.Equal(t, test.expectedTests, ds.Tests.X.Rows())
				require.Equal(t, test.expectedTests, ds.Tests.Y.Rows())
				require.Equal(t, test.expectedValid, ds.Valid.X.Rows())
				require.Equal(t, test.expectedValid, ds.Valid.Y.Rows())
				for i1 := 0; i1 < test.expectedTrain; i1++ {
					expectedX, err := data.X.GetRow(i1)
					require.NoError(t, err)
					actualX, err := ds.Train.X.GetRow(i1)
					require.NoError(t, err)
					require.True(t, expectedX.Equal(actualX))

					expectedY, err := data.Y.GetRow(i)
					require.NoError(t, err)
					actualY, err := ds.Train.Y.GetRow(i)
					require.NoError(t, err)
					require.True(t, expectedY.Equal(actualY))
				}
				for i2 := 0; i2 < test.expectedTests; i2++ {
					expectedX, err := data.X.GetRow(test.expectedTrain + i2)
					require.NoError(t, err)
					actualX, err := ds.Tests.X.GetRow(i2)
					require.NoError(t, err)
					require.True(t, expectedX.Equal(actualX))

					expectedY, err := data.Y.GetRow(test.expectedTrain + i2)
					require.NoError(t, err)
					actualY, err := ds.Tests.Y.GetRow(i2)
					require.NoError(t, err)
					require.True(t, expectedY.Equal(actualY))
				}
				for i3 := 0; i3 < test.expectedValid; i3++ {
					expectedX, err := data.X.GetRow(test.expectedTrain + test.expectedTests + i3)
					require.NoError(t, err)
					actualX, err := ds.Valid.X.GetRow(i3)
					require.NoError(t, err)
					require.True(t, expectedX.Equal(actualX))

					expectedY, err := data.Y.GetRow(test.expectedTrain + test.expectedTests + i3)
					require.NoError(t, err)
					actualY, err := ds.Valid.Y.GetRow(i3)
					require.NoError(t, err)
					require.True(t, expectedY.Equal(actualY))
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, test.Err)
			}
		})
	}
}

func TestDataset(t *testing.T) {
	ds := fabrics.NewDataset(t, fabrics.DatasetParameters{
		Single: &fabrics.DataParameters{
			X: fabrics.MatrixParameters{Rows: 100, Cols: 1},
			Y: fabrics.MatrixParameters{Rows: 100, Cols: 1},
		},
	})

	cp := ds.Copy()
	require.True(t, ds != cp)
	require.True(t, ds.Equal(ds))
	require.True(t, ds.Equal(cp))
	require.True(t, cp.Equal(ds))
}

func TestDataset_Strings(t *testing.T) {
	ds := fabrics.NewDataset(t, fabrics.DatasetParameters{
		Single: &fabrics.DataParameters{
			X: fabrics.MatrixParameters{Rows: 10, Cols: 3},
			Y: fabrics.MatrixParameters{Rows: 10, Cols: 2},
		},
	})

	t.Log("ShortString():\n" + ds.ShortString())
	t.Log("String():\n" + ds.String())
	t.Log("PrettyString():\n" + ds.PrettyString())
}
