package mylog

import (
	"fmt"
	"io"
	"log"
)

var lvlMap = map[Level]string{
	Error: "Error",
	Warn:  "Warn",
	Info:  "Info",
	Debug: "Debug",
	Trace: "Trace",
}

var flags = log.LstdFlags | log.Lmsgprefix | log.Lmicroseconds

type LeveledWriter struct {
	Level  Level
	Writer io.Writer
}

type leveledLogger struct {
	level  Level
	logger *log.Logger
}

type logsWrapper struct {
	loggers []*leveledLogger
}

func (l *leveledLogger) log(lvl Level, msg string) {
	if lvl > l.level {
		return
	}

	l.logger.Println(fmt.Sprintf("[%s] %s", lvlMap[lvl], msg))
}

func (w *logsWrapper) log(lvl Level, msg string) {
	for _, logger := range w.loggers {
		logger.log(lvl, msg)
	}
}

var _ Logger = (*logsView)(nil)

type logsView struct {
	name string
	logs *logsWrapper
}

func (v *logsView) Log(lvl Level, msg string) {
	if v.logs != nil {
		v.logs.log(lvl, fmt.Sprintf("%s: %s", v.name, msg))
	}
}

func (v *logsView) Logf(lvl Level, format string, args ...interface{}) {
	v.Log(lvl, fmt.Sprintf(format, args...))
}

func (v *logsView) CatchErr(err *error) {
	if *err != nil {
		v.Error((*err).Error())
	}
}

func (v *logsView) Error(msg string) {
	v.Log(Error, msg)
}

func (v *logsView) Warn(msg string) {
	v.Log(Warn, msg)
}

func (v *logsView) Info(msg string) {
	v.Log(Info, msg)
}

func (v *logsView) Debug(msg string) {
	v.Log(Debug, msg)
}

func (v *logsView) Trace(msg string) {
	v.Log(Trace, msg)
}

func (v *logsView) Errorf(format string, args ...interface{}) {
	v.Logf(Error, format, args...)
}

func (v *logsView) Warnf(format string, args ...interface{}) {
	v.Logf(Warn, format, args...)
}

func (v *logsView) Infof(format string, args ...interface{}) {
	v.Logf(Info, format, args...)
}

func (v *logsView) Debugf(format string, args ...interface{}) {
	v.Logf(Debug, format, args...)
}

func (v *logsView) Tracef(format string, args ...interface{}) {
	v.Logf(Trace, format, args...)
}

var singleton *logsWrapper

func Setup(writers ...LeveledWriter) {
	if singleton == nil {
		loggers := make([]*leveledLogger, len(writers))
		for i, writer := range writers {
			loggers[i] = &leveledLogger{
				level:  writer.Level,
				logger: log.New(writer.Writer, "", flags),
			}
		}
		singleton = &logsWrapper{
			loggers: loggers,
		}
	}
}

func Reset() {
	singleton = nil
}

func NewLogger(name string) Logger {
	return newLogView(name)
}

func newLogView(name string) *logsView {
	return &logsView{
		name: name,
		logs: singleton,
	}
}
