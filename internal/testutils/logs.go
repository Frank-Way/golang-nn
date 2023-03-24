package testutils

import (
	"nn/pkg/mylog"
	"os"
)

func SetupLogger() {
	mylog.Setup(mylog.LeveledWriter{
		Level:  mylog.Trace,
		Writer: os.Stdout,
	})
}
