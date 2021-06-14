# bonz version
VERSION = 0.0

# Customize below to fit your system

# Install paths
PREFIX := /usr/local
MANPREFIX := $(PREFIX)/share/man

# Depencies includes and libs
INCS := `pkg-config --cflags sdl2 jack fftw3f`
LIBS := `pkg-config --libs sdl2 jack fftw3f`

# Flags
CFLAGS ?= -std=c99 -pedantic -march=native -D_XOPEN_SOURCE=500 -D_POSIX_C_SOURCE=200112L
CFLAGS += -g -W -pthread
CFLAGS += $(INCS) -DVERSION=\"$(VERSION)\"

LDFLAGS += $(LIBS) -ldl
