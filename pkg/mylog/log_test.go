package mylog

import (
	"fmt"
	"github.com/stretchr/testify/require"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestNewStandardLogger(t *testing.T) {
	logger := NewStandardLogger("test-logger")
	casted, ok := logger.(*StandardLogger)
	require.True(t, ok)
	require.Equal(t, "test-logger", casted.name)
	require.Equal(t, 0, len(casted.loggers))
	require.False(t, casted.DisableStdErr())
	require.False(t, casted.DisableFile("test-file"))
	require.False(t, casted.DisableWriter("test-writer"))
}

func TestStandardLogger_WriteToFile(t *testing.T) {
	file, f := getGlobalLogFile(t)
	defer f(file.Name())
	logger := newStandardLogger("test-logger")
	require.NoError(t, logger.EnableFile(Trace, file.Name()))
	logger.Info("info message1")
	logger.Error("error message1")
	logger.Trace("trace message1")
	logger.Info("info message2")
	logger.Debug("debug message1")
}

func send(logger Logger, count int, message string, level Level, timeout int, wg *sync.WaitGroup) {
	for i := 0; i < count; i++ {
		logger.Log(level, message)
		time.Sleep(time.Duration(timeout) * time.Millisecond)
	}
	wg.Done()
}

type stderrS struct {
}

type fileS struct {
	name string
}

type writerS struct {
	name   string
	writer io.Writer
}

type test struct {
	logger   *StandardLogger
	lvl      Level
	stderr   *stderrS
	file     *fileS
	writer   *writerS
	message  string
	count    int
	timeout  int
	msgLevel Level
}

func (t *test) setup(tt *testing.T) {
	if t.stderr != nil {
		t.logger.EnableStdErr(t.lvl)
	}
	if t.file != nil {
		require.NoError(tt, t.logger.EnableFile(t.lvl, t.file.name))
	}
	if t.writer != nil {
		require.NoError(tt, t.logger.EnableWriter(t.lvl, t.writer.name, t.writer.writer))
	}
}

func (t *test) launch(wg *sync.WaitGroup) {
	go send(t.logger, t.count, t.message, t.msgLevel, t.timeout, wg)
}

func expectedSuffix(name string, level Level, msg string) string {
	return fmt.Sprintf("%s [%s] %s", name, mapLevel[level], msg)
}

func (t *test) check(content []string) bool {
	expectedSuf := expectedSuffix(t.logger.name, t.msgLevel, t.message)
	expectedCnt := t.count
	if t.msgLevel > t.lvl {
		expectedCnt = 0
	}
	cnt := 0
	for _, line := range content {
		if strings.HasSuffix(line, expectedSuf) {
			cnt++
		}
	}

	return cnt == expectedCnt
}

func getTests(file *os.File) []*test {
	name := ""
	if file != nil {
		name = file.Name()
	}
	return []*test{
		{
			logger:   newStandardLogger("test-logger1"),
			lvl:      Trace,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 1 logger",
			count:    5,
			timeout:  20,
			msgLevel: Error,
		},
		{
			logger:   newStandardLogger("test-logger2"),
			lvl:      Debug,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 2 logger",
			count:    10,
			timeout:  20,
			msgLevel: Trace,
		},
		{
			logger:   newStandardLogger("test-logger3"),
			lvl:      Info,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 3 logger",
			count:    5,
			timeout:  30,
			msgLevel: Info,
		},
		{
			logger:   newStandardLogger("test-logger4"),
			lvl:      Warn,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 4 logger",
			count:    5,
			timeout:  20,
			msgLevel: Error,
		},
		{
			logger:   newStandardLogger("test-logger5"),
			lvl:      Error,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 5 logger",
			count:    5,
			timeout:  10,
			msgLevel: Info,
		},
		{
			logger:   newStandardLogger("test-logger6"),
			lvl:      0,
			stderr:   &stderrS{},
			file:     &fileS{name: name},
			writer:   &writerS{writer: file},
			message:  "message to 6 logger",
			count:    5,
			timeout:  10,
			msgLevel: Info,
		},
	}
}

func setupTests(t *testing.T, tests []*test) {
	for i := range tests {
		tests[i].setup(t)
	}
}

func launchTests(wg *sync.WaitGroup, tests []*test) {
	for i := range tests {
		tests[i].launch(wg)
	}
}

func checkTests(content []string, tests []*test) bool {
	for i := range tests {
		if !tests[i].check(content) {
			return false
		}
	}
	return true
}

func contains(item int, values []int) bool {
	for _, value := range values {
		if item == value {
			return true
		}
	}
	return false
}

func disableStderr(tests []*test, exceptions ...int) {
	for i, test := range tests {
		if contains(i, exceptions) {
			continue
		}
		test.stderr = nil
	}
}

func disableFile(tests []*test, exceptions ...int) {
	for i, test := range tests {
		if contains(i, exceptions) {
			continue
		}
		test.file = nil
	}
}

func disableWriter(tests []*test, exceptions ...int) {
	for i, test := range tests {
		if contains(i, exceptions) {
			continue
		}
		test.writer = nil
	}
}

func getContent(t *testing.T, filename string) []string {
	bytes, err := ioutil.ReadFile(filename)
	require.NoError(t, err)
	s := string(bytes)
	content := strings.Split(s, "\n")
	return content
}

func getLocalLogFile(t *testing.T) (*os.File, func(s string)) {
	file, err := os.OpenFile("tmp.txt", os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	require.NoError(t, err)
	return file, func(s string) {}
}

func getGlobalLogFile(t *testing.T) (*os.File, func(s string)) {
	file, err := ioutil.TempFile("", "tmp-logfile.*.txt")
	require.NoError(t, err)
	return file, func(s string) {
		os.Remove(s)
	}
}

func TestStandardLogger_MultipleLoggersToOneStderr(t *testing.T) {
	tests := getTests(nil)
	disableFile(tests)
	disableWriter(tests)
	setupTests(t, tests)

	wg := &sync.WaitGroup{}
	wg.Add(len(tests))

	launchTests(wg, tests)

	wg.Wait()
}

func TestStandardLogger_MultipleLoggersToOneFileAsWriter(t *testing.T) {
	file, f := getGlobalLogFile(t)
	defer f(file.Name())

	tests := getTests(file)
	disableStderr(tests)
	disableFile(tests)
	setupTests(t, tests)

	wg := &sync.WaitGroup{}
	wg.Add(len(tests))

	launchTests(wg, tests)

	wg.Wait()
	require.True(t, checkTests(getContent(t, file.Name()), tests))
}

func TestStandardLogger_MultipleLoggersToOneFile(t *testing.T) {
	file, f := getGlobalLogFile(t)
	defer f(file.Name())

	tests := getTests(file)
	disableStderr(tests)
	disableWriter(tests)
	setupTests(t, tests)

	wg := &sync.WaitGroup{}
	wg.Add(len(tests))

	launchTests(wg, tests)

	wg.Wait()
	require.True(t, checkTests(getContent(t, file.Name()), tests))
}
