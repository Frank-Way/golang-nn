package mylog

import (
	"fmt"
	"io"
	"log"
	"os"
)

var _ Logger = (*StandardLogger)(nil)

type leveledLogger struct {
	level  Level
	logger *log.Logger
}

var mapLevel = map[Level]string{
	Error: "Error",
	Warn:  "Warn",
	Info:  "Info",
	Debug: "Debug",
	Trace: "Trace",
}

var flags = log.LstdFlags | log.Lmsgprefix | log.Lmicroseconds

func (l *leveledLogger) log(level Level, message string) {
	if level > l.level {
		return
	}

	l.logger.Print(fmt.Sprintf(" [%s] %s", mapLevel[level], message))
}

type StandardLogger struct {
	name    string
	loggers map[string]*leveledLogger
}

func (l *StandardLogger) Log(level Level, message string) {
	for _, logger := range l.loggers {
		logger.log(level, message)
	}
}

func (l *StandardLogger) Error(message string) {
	l.Log(Error, message)
}

func (l *StandardLogger) Warn(message string) {
	l.Log(Warn, message)
}

func (l *StandardLogger) Info(message string) {
	l.Log(Info, message)
}

func (l *StandardLogger) Debug(message string) {
	l.Log(Debug, message)
}

func (l *StandardLogger) Trace(message string) {
	l.Log(Trace, message)
}

func NewStandardLogger(name string) Logger {
	return newStandardLogger(name)
}

func newStandardLogger(name string) *StandardLogger {
	return &StandardLogger{
		name:    name,
		loggers: make(map[string]*leveledLogger),
	}
}

func stderrKey() string {
	return "!STDERR"
}

func (l *StandardLogger) delete(key string) bool {
	if _, ok := l.loggers[key]; !ok {
		return false
	}

	delete(l.loggers, key)
	return true
}

func (l *StandardLogger) EnableStdErr(level Level) {
	key := stderrKey()
	if _, ok := l.loggers[key]; ok {
		return
	}

	l.loggers[key] = &leveledLogger{
		level:  level,
		logger: log.New(os.Stderr, l.name, flags),
	}
}

func (l *StandardLogger) DisableStdErr() bool {
	return l.delete(stderrKey())
}

func fileKey(name string) string {
	return fmt.Sprintf("!FILE#%s", name)
}

func (l *StandardLogger) EnableFile(level Level, name string) error {
	key := fileKey(name)
	if _, ok := l.loggers[key]; ok {
		return nil
	}
	file, err := os.OpenFile(name, os.O_CREATE|os.O_APPEND|os.O_RDWR, 0644)
	if err != nil {
		return err
	}

	l.loggers[key] = &leveledLogger{
		level:  level,
		logger: log.New(file, l.name, flags),
	}
	return nil
}

func (l *StandardLogger) DisableFile(name string) bool {
	return l.delete(fileKey(name))
}

func writerKey(name string) string {
	return fmt.Sprintf("!WRITER#%s", name)
}

func (l *StandardLogger) EnableWriter(level Level, name string, writer io.Writer) error {
	key := writerKey(name)
	if _, ok := l.loggers[key]; ok {
		return nil
	}

	l.loggers[key] = &leveledLogger{
		level:  level,
		logger: log.New(writer, l.name, flags),
	}
	return nil
}

func (l *StandardLogger) DisableWriter(name string) bool {
	return l.delete(writerKey(name))
}
