package mylog

import (
	"fmt"
	"io"
	"log"
	"strings"
)

var lvlMap = map[Level]string{
	Error: "Error",
	Warn:  "Warn",
	Info:  "Info",
	Debug: "Debug",
	Trace: "Trace",
}

var flags = log.LstdFlags | log.Lmsgprefix | log.Lmicroseconds

// LeveledWriter represent io.Writer which will be attempted to send logs to with specified Level
type LeveledWriter struct {
	Level  Level
	Writer io.Writer
}

// leveledLogger represent LeveledWriter with io.Writer wrapped with log.Logger. It handles level-based logic.
// Here logger has no prefix, e.g. it has no name.
type leveledLogger struct {
	level  Level
	logger *log.Logger
}

// logsWrapper is wrap on slice of leveledLogger. It delegate calls to under laying leveledLogger's. Such wrap allows
//  to use several loggers writing to different destinations with different logging levels.
type logsWrapper struct {
	loggers []*leveledLogger
}

func (l *leveledLogger) log(lvl Level, msg string) {
	if lvl > l.level {
		return
	}
	if strings.Contains(msg, "\n") {
		for _, s := range strings.Split(msg, "\n") {
			l.logger.Println(fmt.Sprintf("[%s] %s", lvlMap[lvl], s))
		}
	} else {
		l.logger.Println(fmt.Sprintf("[%s] %s", lvlMap[lvl], msg))
	}
}

func (w *logsWrapper) log(lvl Level, msg string) {
	for _, logger := range w.loggers {
		logger.log(lvl, msg)
	}
}

var _ Logger = (*logsView)(nil)

// logsView represent a named view on logsWrapper. It allows to give a name to logger. It implements Logger API.
type logsView struct {
	name string
	logs **logsWrapper
}

// IsEnabled iterating over all set up loggers to check it levels
func (v *logsView) IsEnabled(lvl Level) bool {
	if *v.logs != nil {
		for _, logger := range (*v.logs).loggers {
			if lvl <= logger.level {
				return true
			}
		}
	}
	return false
}

func (v *logsView) Log(lvl Level, msg string) {
	if *v.logs != nil && msg != "" {
		(*v.logs).log(lvl, fmt.Sprintf("%s: %s", v.name, msg))
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

// singleton is one logsWrapper used by all logsView
var singleton *logsWrapper

// Setup allows to set up logs configuration
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

// Reset allows to reset logs configuration
func Reset() {
	singleton = nil
}

// NewLogger returns Logger implementation
func NewLogger(name string) Logger {
	return newLogView(name)
}

// newLogView returns named view on singleton
func newLogView(name string) *logsView {
	return &logsView{
		name: name,
		logs: &singleton,
	}
}
