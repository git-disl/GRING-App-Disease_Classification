UNAME := $(shell uname)

all: GRING_plugin.so bootstrap

GRING_plugin.so:
	go build -buildmode=c-shared -o $@ peer.go

bootstrap:
	go build -o $@ bootstrap.go

clean:
	rm -f peer bootstrap GRING_plugin.so GRING_plugin.h
