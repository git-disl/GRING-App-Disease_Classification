package main

import (
	"bufio"
	"errors"
	"fmt"
	"github.com/theued/GRING"
	"github.com/theued/GRING/dual"
	"io"
	"os"
	"strings"
	"time"
	"net"
	"go.uber.org/zap"
	"strconv"
	"runtime"
)

// printedLength is the total prefix length of a public key associated to a chat users ID.
const printedLength = 8

var (
        node *GRING.Node
	overlay *dual.Protocol
	events dual.Events
	max_peers=100
	max_conn=1000
	discoverInterval = 30 //sec
)

type chatMessage struct {
	Opcode byte
	Contents []byte
}

func (m chatMessage) Marshal() []byte {
	return append([]byte{m.Opcode}, m.Contents...)
}

func unmarshalChatMessage(buf []byte) (chatMessage, error) {
	return chatMessage{Opcode: buf[0], Contents: buf[1:]}, nil
}

// check panics if err is not nil.
func check(err error) {
	if err != nil {
		panic(err)
	}
}

func init_p2p(host string, port int, publicIP string){
        var err error

	//fmt.Printf("host : %s port : %d \n",host,port)

        logger, _ := zap.NewProduction()
	// Create a new configured node.
	node, err = GRING.NewNode(
		GRING.WithNodeBindHost(net.ParseIP(host)),
		GRING.WithNodeBindPort(uint16(port)),
		GRING.WithNodeAddress(publicIP),
		GRING.WithNodeMaxRecvMessageSize(1<<24), //16MB
		GRING.WithNodeMaxInboundConnections(uint(max_conn)),
                GRING.WithNodeMaxOutboundConnections(uint(max_conn)),
		GRING.WithNodeLogger(logger),
	)
	check(err)

	// Register the chatMessage Go type to the node with an associated unmarshal function.
        node.RegisterMessage(chatMessage{}, unmarshalChatMessage)

	// Instantiate dual.
	events = dual.Events{
	        OnPeerAdmitted_bc: func(id GRING.ID) {
	                fmt.Printf("bootstrap : a new peer %s(%s).\n", id.Address, id.ID.String()[:printedLength])
	        },
		OnPeerEvicted: func(id GRING.ID) {
			//fmt.Printf("Forgotten a peer %s(%s).\n", id.Address, id.ID.String()[:printedLength])
		},
	}

	overlay = dual.New(dual.WithProtocolEvents(events),
	                   dual.WithProtocolMaxNeighborsBC(max_peers),)

	// Bind dual to the node.
	node.Bind(overlay.Protocol())

        //fmt.Printf("start listen\n")
	// Have the node start listening for new peers.
	check(node.Listen())

	// Print out the nodes ID and a help message comprised of commands.
	//help(node)

        //go startPeriodicDiscover()

        fmt.Printf("init done\n")
}

func startPeriodicDiscover() {
        var len_peers int

	for {
	    time.Sleep(time.Duration(discoverInterval) * time.Second)
            len_peers = len(overlay.Table_bc().KEntries(max_peers))
	    if len_peers < max_peers {
	        ids := overlay.DiscoverRandom(false)
	        var str []string
	        for _, id := range ids {
		    str = append(str, fmt.Sprintf("%s(%s)", id.Address, id.ID.String()[:printedLength]))
	        }
/*
	        if len(ids) > 0 {
		    fmt.Printf("Discovered %d peer(s): [%v]\n", len(ids), strings.Join(str, ", "))
	        } else {
		    fmt.Printf("Did not discover any peers.\n")
	        }
*/
	    }
	}
}

func Input() {
	r := bufio.NewReader(os.Stdin)

	for {
		buf, _, err := r.ReadLine()
		if err != nil {
			if errors.Is(err, io.EOF) {
				return
			}

			check(err)
		}

		line := string(buf)
		if len(line) == 0 {
			continue
		}

                fmt.Printf(line)
		chat(line)
	}
}

// help prints out the users ID and commands available.
func help(node *GRING.Node) {
	fmt.Printf("Your ID is %s(%s). Type '/discover' to attempt to discover new "+
		"peers, or '/peers' to list out all peers you are connected to.\n",
		node.ID().Address,
		node.ID().ID.String()[:printedLength],
	)
}

// discover uses Kademlia to discover new peers from nodes we already are aware of.
func discover(overlay *dual.Protocol) {
	ids := overlay.DiscoverRandom(false)

	var str []string
	for _, id := range ids {
		str = append(str, fmt.Sprintf("%s(%s)", id.Address, id.ID.String()[:printedLength]))
	}
	if len(ids) > 0 {
		fmt.Printf("Discovered %d peer(s): [%v]\n", len(ids), strings.Join(str, ", "))
	} else {
		fmt.Printf("Did not discover any peers.\n")
	}
}

func chat(line string) {
	switch line {
	case "/discover":
		discover(overlay)
		return
	case "/peers_bc":
		overlay.KPeers_bc(max_peers) // show backbone overlay routing table
		return
	default:
	}

	if strings.HasPrefix(line, "/") {
		help(node)
		return
	}

}

func main() {
        // args[0] = host IP 
        // args[1] = host Port 
	// args[2] = host Public Address(IP:Port) if host IP is private and behind NAT
        runtime.GOMAXPROCS(runtime.NumCPU())
        args := os.Args[1:]
        port, err := strconv.Atoi(args[1])
	if err != nil {
	    // Add code here to handle the error!
	}

	if len(args) > 2 {
            init_p2p(string(args[0]), port, string(args[2]))
        }else{
            init_p2p(string(args[0]), port, "")
        }

        // block here
        Input() // simulation many bench on the same physical node causes error on stdin
}


