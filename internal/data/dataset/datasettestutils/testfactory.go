package datasettestutils

import (
	"github.com/stretchr/testify/require"
	"nn/internal/data/dataset"
	"nn/internal/testutils/testfactories"
	"testing"
)

type DataParameters struct {
	X testfactories.MatrixParameters
	Y testfactories.MatrixParameters
}

func NewData(t *testing.T, parameters DataParameters) *dataset.Data {
	x := testfactories.NewMatrix(t, parameters.X)
	y := testfactories.NewMatrix(t, parameters.Y)
	data, err := dataset.NewData(x, y)
	require.NoError(t, err)

	return data
}

type DatasetParameters struct {
	Train  DataParameters
	Tests  DataParameters
	Valid  DataParameters
	Single *DataParameters
}

func NewDataset(t *testing.T, parameters DatasetParameters) *dataset.Dataset {
	if parameters.Single != nil {
		data := NewData(t, *parameters.Single)
		ds, err := dataset.NewDatasetSplit(data, dataset.DefaultDataSplitParameters)
		require.NoError(t, err)

		return ds
	}
	train := NewData(t, parameters.Train)
	tests := NewData(t, parameters.Tests)
	valid := NewData(t, parameters.Valid)

	ds, err := dataset.NewDataset(train, tests, valid)
	require.NoError(t, err)

	return ds
}
