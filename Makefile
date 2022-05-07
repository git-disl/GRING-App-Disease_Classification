UNAME := $(shell uname)

all: libp2p_peer.so bootstrap

libp2p_peer.so:
	go build -buildmode=c-shared -o $@ peer.go

bootstrap:
	go build -o $@ bootstrap.go

clean:
	rm -f peer bootstrap libp2p_peer.so libp2p_peer.h
