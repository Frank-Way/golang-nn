package test

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"nn/internal/test/utils"
	"nn/internal/test/utils/fabrics"
	"nn/pkg/percent"
	"testing"
)

func TestNewDataset(t *testing.T) {
	tests := []struct {
		utils.Base
		train         fabrics.DataParameters
		tests_        fabrics.DataParameters
		valid         fabrics.DataParameters
		nilCheckTrain bool
		nilCheckTests bool
		nilCheckValid bool
	}{
		{
			Base:   utils.Base{Name: "valid parameters"},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:          utils.Base{Name: "nil train", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckTrain: true,
		},
		{
			Base:          utils.Base{Name: "nil tests", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckTests: true,
		},
		{
			Base:          utils.Base{Name: "nil valid", Err: dataset.ErrCreate},
			train:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:         fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
			nilCheckValid: true,
		},
		{
			Base:   utils.Base{Name: "train and tests inputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 2}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   utils.Base{Name: "train and tests outputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 2}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   utils.Base{Name: "tests and valid inputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 2}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 1}},
		},
		{
			Base:   utils.Base{Name: "tests and valid outputs size mismatch", Err: dataset.ErrCreate},
			train:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 3, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 3, Cols: 1}},
			tests_: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 2, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 2, Cols: 1}},
			valid:  fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 5, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 5, Cols: 2}},
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			train := fabrics.NewData(t, tests[i].train)
			tests_ := fabrics.NewData(t, tests[i].tests_)
			valid := fabrics.NewData(t, tests[i].valid)
			if tests[i].nilCheckTrain {
				train = nil
			}
			if tests[i].nilCheckTests {
				tests_ = nil
			}
			if tests[i].nilCheckValid {
				valid = nil
			}

			_, err := dataset.NewDataset(train, tests_, valid)
			if tests[i].Err == nil {
				require.NoError(t, err)
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
			}
		})
	}
}

func TestNewDatasetSplit(t *testing.T) {
	tests := []struct {
		utils.Base
		single        fabrics.DataParameters
		params        *dataset.DataSplitParameters
		expectedTrain int
		expectedTests int
		expectedValid int
	}{
		{
			Base:          utils.Base{Name: "enough values, default parameters"},
			single:        fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params:        dataset.DefaultDataSplitParameters,
			expectedTrain: 60,
			expectedTests: 30,
			expectedValid: 10,
		},
		{
			Base:   utils.Base{Name: "not enough values", Err: dataset.ErrCreate},
			single: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params: &dataset.DataSplitParameters{TrainPercent: percent.Percent50, TestsPercent: percent.Percent50, ValidPercent: percent.Percent50},
		},
		{
			Base:   utils.Base{Name: "too many values", Err: dataset.ErrCreate},
			single: fabrics.DataParameters{X: fabrics.MatrixParameters{Rows: 100, Cols: 1}, Y: fabrics.MatrixParameters{Rows: 100, Cols: 1}},
			params: &dataset.DataSplitParameters{TrainPercent: percent.Percent10, TestsPercent: percent.Percent10, ValidPercent: percent.Percent10},
		},
	}

	for i := range tests {
		t.Run(tests[i].Name, func(t *testing.T) {
			data := fabrics.NewData(t, tests[i].single)

			ds, err := dataset.NewDatasetSplit(data, tests[i].params)
			if tests[i].Err == nil {
				require.NoError(t, err)
				require.Equal(t, tests[i].expectedTrain, ds.Train.X.Rows())
				require.Equal(t, tests[i].expectedTrain, ds.Train.Y.Rows())
				require.Equal(t, tests[i].expectedTests, ds.Tests.X.Rows())
				require.Equal(t, tests[i].expectedTests, ds.Tests.Y.Rows())
				require.Equal(t, tests[i].expectedValid, ds.Valid.X.Rows())
				require.Equal(t, tests[i].expectedValid, ds.Valid.Y.Rows())
				for i1 := 0; i1 < tests[i].expectedTrain; i1++ {
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
				for i2 := 0; i2 < tests[i].expectedTests; i2++ {
					expectedX, err := data.X.GetRow(tests[i].expectedTrain + i2)
					require.NoError(t, err)
					actualX, err := ds.Tests.X.GetRow(i2)
					require.NoError(t, err)
					require.True(t, expectedX.Equal(actualX))

					expectedY, err := data.Y.GetRow(tests[i].expectedTrain + i2)
					require.NoError(t, err)
					actualY, err := ds.Tests.Y.GetRow(i2)
					require.NoError(t, err)
					require.True(t, expectedY.Equal(actualY))
				}
				for i3 := 0; i3 < tests[i].expectedValid; i3++ {
					expectedX, err := data.X.GetRow(tests[i].expectedTrain + tests[i].expectedTests + i3)
					require.NoError(t, err)
					actualX, err := ds.Valid.X.GetRow(i3)
					require.NoError(t, err)
					require.True(t, expectedX.Equal(actualX))

					expectedY, err := data.Y.GetRow(tests[i].expectedTrain + tests[i].expectedTests + i3)
					require.NoError(t, err)
					actualY, err := ds.Valid.Y.GetRow(i3)
					require.NoError(t, err)
					require.True(t, expectedY.Equal(actualY))
				}
			} else {
				require.Error(t, err)
				require.ErrorIs(t, err, tests[i].Err)
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
