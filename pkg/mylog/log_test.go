package mylog

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"io/ioutil"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewLogger(t *testing.T) {
	Reset()
	Setup(LeveledWriter{Level: Info, Writer: os.Stderr})
	loggers := make([]Logger, 2)

	for i, name := range []string{"view-1", "view-2"} {
		loggers[i] = NewLogger(name)
		require.NotNil(t, loggers[i])
	}
}

func send(logger Logger, test logtest) {
	var wg sync.WaitGroup
	wg.Add(test.errorMsgCnt)
	wg.Add(test.warnMsgCnt)
	wg.Add(test.infoMsgCnt)
	wg.Add(test.debugMsgCnt)
	wg.Add(test.traceMsgCnt)

	go func() {
		for i := 0; i < test.errorMsgCnt; i++ {
			time.Sleep(time.Duration(5) * time.Millisecond)
			logger.Error(fmt.Sprintf("%d'th Error message", i))
			wg.Done()
		}
	}()
	go func() {
		for i := 0; i < test.warnMsgCnt; i++ {
			time.Sleep(time.Duration(4) * time.Millisecond)
			logger.Warn(fmt.Sprintf("%d'th Warn message", i))
			wg.Done()
		}
	}()
	go func() {
		for i := 0; i < test.infoMsgCnt; i++ {
			time.Sleep(time.Duration(3) * time.Millisecond)
			logger.Info(fmt.Sprintf("%d'th Info message", i))
			wg.Done()
		}
	}()
	go func() {
		for i := 0; i < test.debugMsgCnt; i++ {
			time.Sleep(time.Duration(2) * time.Millisecond)
			logger.Debug(fmt.Sprintf("%d'th Debug message", i))
			wg.Done()
		}
	}()
	go func() {
		for i := 0; i < test.traceMsgCnt; i++ {
			time.Sleep(time.Millisecond)
			logger.Trace(fmt.Sprintf("%d'th Trace message", i))
			wg.Done()
		}
	}()

	wg.Wait()
}

func anyHasSuffix(suffix string, values []string) bool {
	for _, value := range values {
		if strings.HasSuffix(value, suffix) {
			return true
		}
	}
	return false
}

func check(t *testing.T, loggerName string, filename string, test logtest, level Level) {
	content := getContent(t, filename)
	for _, lvl := range []Level{Error, Warn, Info, Debug, Trace} {
		if lvl > level {
			continue
		}
		cnt := 0
		if lvl == Error {
			cnt = test.errorMsgCnt
		} else if lvl == Warn {
			cnt = test.warnMsgCnt
		} else if lvl == Info {
			cnt = test.infoMsgCnt
		} else if lvl == Debug {
			cnt = test.debugMsgCnt
		} else if lvl == Trace {
			cnt = test.traceMsgCnt
		}

		for i := 0; i < cnt; i++ {
			suffix := fmt.Sprintf("[%s] %s: %d'th %s message", lvlMap[lvl], loggerName, i, lvlMap[lvl])
			require.True(t, anyHasSuffix(suffix, content))
		}
	}
}

type logtest struct {
	name        string
	loggersCnt  int
	errorMsgCnt int
	warnMsgCnt  int
	infoMsgCnt  int
	debugMsgCnt int
	traceMsgCnt int
}

var tests = []logtest{
	{name: "3 loggers, 3 messages of every type", loggersCnt: 3, errorMsgCnt: 3, warnMsgCnt: 3, infoMsgCnt: 3, debugMsgCnt: 3, traceMsgCnt: 3},
}

func getContent(t *testing.T, filename string) []string {
	bytes, err := ioutil.ReadFile(filename)
	require.NoError(t, err)
	s := string(bytes)
	content := strings.Split(s, "\n")
	return content
}

func run(t *testing.T, tests []logtest, levels []Level, files []*os.File) {
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var wg sync.WaitGroup
			for i := 0; i < test.loggersCnt; i++ {
				wg.Add(1)
				go func(i int) {
					send(NewLogger(fmt.Sprintf("logger-%d", i)), test)
					wg.Done()
				}(i)
			}

			wg.Wait()
			for j, file := range files {
				for i := 0; i < test.loggersCnt; i++ {
					check(t, fmt.Sprintf("logger-%d", i), file.Name(), test, levels[j])
				}
			}
		})
	}
}

func TestLogger_Stderr(t *testing.T) {
	Reset()
	Setup(LeveledWriter{Level: Info, Writer: os.Stderr})
	run(t, tests, nil, nil)
}

func TestLogger_TwoDiffFilesDiffLevels(t *testing.T) {
	file1, err := ioutil.TempFile("", "tmp-logfile1-&.txt")
	require.NoError(t, err)
	defer os.Remove(file1.Name())
	file2, err := ioutil.TempFile("", "tmp-logfile2-&.txt")
	require.NoError(t, err)
	defer os.Remove(file2.Name())
	Reset()
	Setup(LeveledWriter{Level: Warn, Writer: file1},
		LeveledWriter{Level: Debug, Writer: file2})
	run(t, tests, []Level{Warn, Debug}, []*os.File{file1, file2})
}

func TestLogger_TwoWritersOnSameFileDiffLevels(t *testing.T) {
	file, err := ioutil.TempFile("", "tmp-logfile-&.txt")
	require.NoError(t, err)
	defer os.Remove(file.Name())
	Reset()
	Setup(LeveledWriter{Level: Warn, Writer: file},
		LeveledWriter{Level: Debug, Writer: file})
	run(t, tests, []Level{Warn, Debug}, []*os.File{file, file})
}

func TestLogger_StderrAndFileDiffLevels(t *testing.T) {
	file, err := ioutil.TempFile("", "tmp-logfile-&.txt")
	require.NoError(t, err)
	defer os.Remove(file.Name())
	Reset()
	Setup(LeveledWriter{Level: Info, Writer: os.Stderr},
		LeveledWriter{Level: Trace, Writer: file})
	run(t, tests, []Level{Trace}, []*os.File{file})
}

func TestLogger_CatchErr(t *testing.T) {
	file, err := ioutil.TempFile("", "tmp-logfile-&.txt")
	require.NoError(t, err)
	defer os.Remove(file.Name())
	Reset()
	Setup(LeveledWriter{Level: Info, Writer: file})
	logger := NewLogger("logger-1")
	f := func(i int) (err error) {
		defer logger.CatchErr(&err)
		logger.Infof("got %d", i)
		if i < 0 {
			return fmt.Errorf("negative input: %d", i)
		}
		logger.Infof("processing %d", i)
		return nil
	}

	require.Error(t, f(-2))
	require.NoError(t, f(2))

	content := getContent(t, file.Name())
	expectedSuffixes := []string{
		"[Info] logger-1: got -2",
		"[Error] logger-1: negative input: -2",
		"[Info] logger-1: got 2",
		"[Info] logger-1: processing 2",
		"",
	}
	require.Equal(t, len(content), len(expectedSuffixes))
	for i := 0; i < len(content); i++ {
		require.True(t, strings.HasSuffix(content[i], expectedSuffixes[i]))
	}
}

func TestLogger_CatchWrappedErr(t *testing.T) {
	file, err := ioutil.TempFile("", "tmp-logfile-&.txt")
	require.NoError(t, err)
	defer os.Remove(file.Name())
	Reset()
	Setup(LeveledWriter{Level: Info, Writer: file})
	logger := NewLogger("logger-1")
	f := func(i int) (err error) {
		defer logger.CatchErr(&err)
		defer func(err *error) {
			if *err != nil {
				*err = fmt.Errorf("wrap: %w", *err)
			}
		}(&err)
		logger.Infof("got %d", i)
		if i < 0 {
			return fmt.Errorf("negative input: %d", i)
		}
		logger.Infof("processing %d", i)
		return nil
	}

	require.Error(t, f(-2))
	require.NoError(t, f(2))

	content := getContent(t, file.Name())
	expectedSuffixes := []string{
		"[Info] logger-1: got -2",
		"[Error] logger-1: wrap: negative input: -2",
		"[Info] logger-1: got 2",
		"[Info] logger-1: processing 2",
		"",
	}
	require.Equal(t, len(content), len(expectedSuffixes))
	for i := 0; i < len(content); i++ {
		t.Log(content[i])
		require.True(t, strings.HasSuffix(content[i], expectedSuffixes[i]))
	}
}

type lvltest struct {
	lvl      Level
	expected bool
}

func TestLogger_IsEnabled(t *testing.T) {
	tests := []struct {
		name    string
		writers []LeveledWriter
		checks  []lvltest
	}{
		{
			name: "info logger",
			writers: []LeveledWriter{
				{Level: Info, Writer: os.Stderr},
			},
			checks: []lvltest{
				{expected: true},
				{lvl: Error, expected: true},
				{lvl: Warn, expected: true},
				{lvl: Info, expected: true},
				{lvl: Debug, expected: false},
				{lvl: Trace, expected: false},
			},
		},
		{
			name: "warn, debug logger",
			writers: []LeveledWriter{
				{Level: Warn, Writer: os.Stderr},
				{Level: Debug, Writer: os.Stderr},
			},
			checks: []lvltest{
				{expected: true},
				{lvl: Error, expected: true},
				{lvl: Warn, expected: true},
				{lvl: Info, expected: true},
				{lvl: Debug, expected: true},
				{lvl: Trace, expected: false},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			Reset()
			Setup(test.writers...)

			logger := NewLogger("losstestutils-logger")

			for _, check := range test.checks {
				require.Equal(t, check.expected, logger.IsEnabled(check.lvl))
			}
		})
	}
}
