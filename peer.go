package main

/*
typedef void (*convert) ();

static inline void call_c_func(convert ptr, char* data) {
        (ptr)(data);
}

static inline void call_c_func2(convert ptr, int num) {
        (ptr)(num);
}
*/
import "C"

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"github.com/git-disl/GRING"
	"github.com/git-disl/GRING/dual"
	"io"
	"os"
	"strings"
	"time"
	"net"
	"unsafe"
	"go.uber.org/zap"
	"sort"
	"encoding/json"
	"strconv"
	"math"
        //"math/rand"
	"sync"
	"runtime"
	"encoding/binary"
)

// application opcode
const (
	//to server
	OP_RECV                      = 0x00
	//OP_CLIENT_WAKE_UP            = 0x01
	OP_CLIENT_READY              = 0x02
	OP_CLIENT_UPDATE             = 0x03
	OP_CLIENT_EVAL               = 0x04
	//to client
	OP_INIT                      = 0x05
	OP_REQUEST_UPDATE            = 0x06
	OP_STOP_AND_EVAL             = 0x07

	OP_GLOBAL_MODEL              = 0x08
)

// printedLength is the total prefix length of a public key associated to a user ID.
const printedLength = 8

type userMessage struct {
	Opcode byte
	Contents []byte
}

func (m userMessage) Marshal() []byte {
	return append([]byte{m.Opcode}, m.Contents...)
}

func unmarshalUserMessage(buf []byte) (userMessage, error) {
	return userMessage{Opcode: buf[0], Contents: buf[1:]}, nil
}

type UserInfo struct {
    TrainTime uint32
    NumLeave uint32
    TrustValue uint32
    UserID GRING.ID
}

func (m UserInfo) Marshal() []byte {
    var dst []byte
    dst = append(dst, make([]byte, 12)...)
    binary.BigEndian.PutUint32(dst[:4], m.TrainTime)
    binary.BigEndian.PutUint32(dst[4:8], m.NumLeave)
    binary.BigEndian.PutUint32(dst[8:], m.TrustValue)
    dst = append(dst, m.UserID.Marshal()...)

    return dst
}

func UnmarshalUserInfo(buf []byte) (UserInfo, error) {
    traintime := binary.BigEndian.Uint32(buf[:4])
    numleave := binary.BigEndian.Uint32(buf[4:8])
    trustvalue := binary.BigEndian.Uint32(buf[8:12])
    userid, err := GRING.UnmarshalID(buf[12:])
    if err != nil {
        return UserInfo{}, io.ErrUnexpectedEOF
    }
    //buf = buf[40+userid.Size():]

    return UserInfo{TrainTime: traintime, NumLeave: numleave, TrustValue: trustvalue, UserID: userid}, nil
}

type ops struct {
        on_set_num_client C.convert
        on_joined C.convert
	//server
        on_wakeup_initiator C.convert
        on_wakeup_subleader C.convert
        on_clientready_initiator C.convert
        on_clientready_subleader C.convert
        on_clientupdate_initiator C.convert
        on_clientupdatedone_initiator C.convert
        on_clientupdate_subleader C.convert
        on_clienteval_initiator C.convert
        on_clientevaldone_initiator C.convert
        on_clienteval_subleader C.convert
	on_trainmymodel C.convert
	on_globalmodel C.convert
	on_trainnextround C.convert

	//peers
        on_init_worker C.convert
        on_init_subleader C.convert
        on_requestupdate_worker C.convert
        on_requestupdate_subleader C.convert
        on_stopandeval_worker C.convert
        on_stopandeval_subleader C.convert
	on_reportclientupdate C.convert
	on_reportclienteval C.convert

	//publisher
	on_reqglobalmodel C.convert
	on_clientupdatedone_publisher C.convert
	on_clientevaldone_publisher C.convert
}

var (
        node *GRING.Node
	overlay *dual.Protocol
	events dual.Events
	callbacks ops
	max_peers=50
        //group_size=9 // group_size-1 = degree of tree = # of children
        group_size=3 // group_size-1 = degree of tree = # of children
	max_conn=1000
	discoverInterval = 60 //sec

	mutex_client_ready *sync.Mutex
	mutex_client_update *sync.Mutex
	mutex_client_eval *sync.Mutex
        num_client_ready = 0
        num_client_update  = 0
        num_client_eval = 0

        //for aggregation
	report = false

        // TODO : under the multi project, publisher and subscriber role should be recorded by each project
	isPublisher = false

        //publisher
        initiator string // record current round initiator address
	starttime time.Time
	my_pjt_id string

	//subscriber
        subscribed_pjt map[string]string

	//for user info test
	mytraintime = 0 //seconds // for debug
	mynumleave = 0
	mytrustvalue = 0
	current_round = 0

	//for straggler timer
	// TODO : should be calculated. disse time + processing time + aggregation time at each sub-tree
	timerDurationReady = 300 // 5min
	timerDurationUpdate = 1200 // 20min
	timerDurationEval = 1200 // 20min
	isFirstReportReady = true
	isFirstReportUpdate = true
	isFirstReportEval = true
	mutexReportReady *sync.Mutex
	mutexReportUpdate *sync.Mutex
	mutexReportEval *sync.Mutex
	timerReportChecker *time.Timer
	doneTimer chan bool
	resetTimer chan bool
	toleranceRate = 100 //%
	toleranceReduceStep = 10 //%

	transfertime = 27 //seconds // for debug
)

//export ResetNumClientReady
func ResetNumClientReady(){
	mutex_client_ready.Lock()
	num_client_ready = 0
	mutex_client_ready.Unlock()
}

//export ResetNumClientUpdate
func ResetNumClientUpdate(){
	mutex_client_update.Lock()
	num_client_update = 0
	mutex_client_update.Unlock()
}

//export ResetNumClientEval
func ResetNumClientEval(){
	mutex_client_eval.Lock()
	num_client_eval = 0
	mutex_client_eval.Unlock()
}

//export IncreaseNumClientReady
func IncreaseNumClientReady(){
	mutex_client_ready.Lock()
        num_client_ready++
	fmt.Printf("call IncreaseClientReady: %d\n",num_client_ready)

	//if me is the last one. I report
	if num_client_ready == (overlay.GetTotMembers()+1) {
            // reset the value after reporting
	    num := num_client_ready
	    num_client_ready = 0
	    mutex_client_ready.Unlock()

            usermsg := userMessage{Opcode: OP_CLIENT_READY, Contents: nil}
            overlay.Report_GR(node.ID(), usermsg.Marshal(), num)

	    // done timer
	    doneTimer <- true
        }else{
	    mutex_client_ready.Unlock()
	}
}

//export IncreaseNumClientUpdateInitiator
func IncreaseNumClientUpdateInitiator(){
	mutex_client_update.Lock()
        num_client_update++
	fmt.Printf("call IncreaseClientUpdateInitiator: %d , tot_members: %d\n",num_client_update, overlay.GetTotMembers())

	//if me is the last one. I finalize the round
	if num_client_update == (overlay.GetTotMembers()+1) {
           // reset the value after reporting
	   num_client_update = 0
	   mutex_client_update.Unlock()

	   ptr := unsafe.Pointer(nil)
           C.call_c_func(callbacks.on_clientupdatedone_initiator, (*C.char)(ptr))

	   // done timer
	   doneTimer <- true
        }else{
	   mutex_client_update.Unlock()
	}
}

//export IncreaseNumClientUpdate
func IncreaseNumClientUpdate(){
	mutex_client_update.Lock()
        num_client_update++
	fmt.Printf("call IncreaseClientUpdate: %d\n",num_client_update)

	//if me is the last one. I report
	if num_client_update == (overlay.GetTotMembers()+1) {
           // reset the value after reporting
	   num := num_client_update
	   num_client_update = 0
	   mutex_client_update.Unlock()

           // python code will aggreagte and report
           C.call_c_func2(callbacks.on_reportclientupdate, (C.int)(num))

	   // done timer
	   doneTimer <- true
        }else{
	   mutex_client_update.Unlock()
	}
}

//export IncreaseNumClientEvalInitiator
func IncreaseNumClientEvalInitiator(){
	mutex_client_eval.Lock()
        num_client_eval++
	fmt.Printf("call IncreaseClientEval: %d\n",num_client_eval)

	//if me is the last one. I report
	if num_client_eval == (overlay.GetTotMembers()+1) {
            // reset the value after reporting
	    num_client_eval = 0
	    mutex_client_eval.Unlock()

            fmt.Printf("call client eval done initiator\n")
            ptr := unsafe.Pointer(nil)
	    // watch out nested lock on_clientevaldone_initiator
            C.call_c_func(callbacks.on_clientevaldone_initiator, (*C.char)(ptr))

            // done timer
	    doneTimer <- true
        }else{
	    mutex_client_eval.Unlock()
	}
}

//export IncreaseNumClientEval
func IncreaseNumClientEval(){
	mutex_client_eval.Lock()
        num_client_eval++
	fmt.Printf("call IncreaseClientEval: %d\n",num_client_eval)

	//if me is the last one. I report
	if num_client_eval == (overlay.GetTotMembers()+1) {
            // reset the value after reporting
	    num := num_client_eval
	    num_client_eval = 0
	    mutex_client_eval.Unlock()

            // python code will aggreagte and report
            C.call_c_func2(callbacks.on_reportclienteval, (C.int)(num))

            // done timer
	    doneTimer <- true
        }else{
	    mutex_client_eval.Unlock()
	}
}

//export Register_callback
func Register_callback(name *C.char, fn C.convert) {
	//fmt.Printf("register callback : %s\n",C.GoString(name))

	if C.GoString(name) == "on_set_num_client" {
	    //fmt.Printf("on_set_num_client registered\n")
	    callbacks.on_set_num_client = fn
	}

	if C.GoString(name) == "on_joined" {
	    //fmt.Printf("on_joined registered\n")
	    callbacks.on_joined = fn
	}
	//server handler
	if C.GoString(name) == "on_wakeup_initiator" {
	    //fmt.Printf("on_wakeup_initiator registered\n")
	    callbacks.on_wakeup_initiator = fn
	}
	if C.GoString(name) == "on_wakeup_subleader" {
	    //fmt.Printf("on_wakeup_subleader registered\n")
	    callbacks.on_wakeup_subleader = fn
	}
	if C.GoString(name) == "on_clientready_initiator" {
	    //fmt.Printf("on_clientready_initiator registered\n")
	    callbacks.on_clientready_initiator = fn
	}
	if C.GoString(name) == "on_clientready_subleader" {
	    //fmt.Printf("on_clientready_subleader registered\n")
	    callbacks.on_clientready_subleader = fn
	}
	if C.GoString(name) == "on_clientupdate_initiator" {
	    //fmt.Printf("on_clientupdate_initiator registered\n")
	    callbacks.on_clientupdate_initiator = fn
	}
	if C.GoString(name) == "on_clientupdatedone_initiator" {
	    //fmt.Printf("on_clientupdatedone_initiator registered\n")
	    callbacks.on_clientupdatedone_initiator = fn
	}
	if C.GoString(name) == "on_clientupdate_subleader" {
	    //fmt.Printf("on_clientupdate_subleader registered\n")
	    callbacks.on_clientupdate_subleader = fn
	}
	if C.GoString(name) == "on_clienteval_initiator" {
	    //fmt.Printf("on_clienteval_initiator registered\n")
	    callbacks.on_clienteval_initiator = fn
	}
	if C.GoString(name) == "on_clientevaldone_initiator" {
	    //fmt.Printf("on_clientevaldone_initiator registered\n")
	    callbacks.on_clientevaldone_initiator = fn
	}
	if C.GoString(name) == "on_clienteval_subleader" {
	    //fmt.Printf("on_clienteval_subleader registered\n")
	    callbacks.on_clienteval_subleader = fn
	}
	//client handler
	if C.GoString(name) == "on_init_worker" {
	    //fmt.Printf("on_init_worker registered\n")
	    callbacks.on_init_worker = fn
	}
	if C.GoString(name) == "on_init_subleader" {
	    //fmt.Printf("on_init_subleader registered\n")
	    callbacks.on_init_subleader = fn
	}
        if C.GoString(name) == "on_request_update_worker" {
	    //fmt.Printf("on_request_update_worker registered\n")
	    callbacks.on_requestupdate_worker = fn
	}
	if C.GoString(name) == "on_request_update_subleader" {
	    //fmt.Printf("on_request_update_subleader registered\n")
	    callbacks.on_requestupdate_subleader = fn
	}
	if C.GoString(name) == "on_stop_and_eval_worker" {
	    //fmt.Printf("on_stop_and_eval_worker registered\n")
	    callbacks.on_stopandeval_worker = fn
	}
	if C.GoString(name) == "on_stop_and_eval_subleader" {
	    //fmt.Printf("on_stop_and_eval_subleader registered\n")
	    callbacks.on_stopandeval_subleader = fn
	}

        if C.GoString(name) == "on_global_model" {
	    //fmt.Printf("on_global_model registered\n")
	    callbacks.on_globalmodel = fn
	}

        if C.GoString(name) == "on_report_client_update" {
	    //fmt.Printf("on_report_client_update\n")
	    callbacks.on_reportclientupdate = fn
	}

        if C.GoString(name) == "on_train_next_round" {
	    //fmt.Printf("on_train_next_round\n")
	    callbacks.on_trainnextround = fn
	}

        if C.GoString(name) == "on_report_client_eval" {
	    //fmt.Printf("on_report_client_eval\n")
	    callbacks.on_reportclienteval = fn
	}

        if C.GoString(name) == "on_train_my_model" {
	    //fmt.Printf("on_train_my_model\n")
	    callbacks.on_trainmymodel = fn
	}

	//publisher
	if C.GoString(name) == "on_reqglobalmodel" {
	    //fmt.Printf("on_reqglobalmodel registered\n")
	    callbacks.on_reqglobalmodel = fn
	}
	if C.GoString(name) == "on_clientupdatedone_publisher" {
	    //fmt.Printf("on_clientupdatedone_publisher registered\n")
	    callbacks.on_clientupdatedone_publisher = fn
	}
	if C.GoString(name) == "on_clientevaldone_publisher" {
	    //fmt.Printf("on_clientevaldone_publisher registered\n")
	    callbacks.on_clientevaldone_publisher = fn
	}

}

//export RecordMyTrainTime
func RecordMyTrainTime(time C.int) {
    fmt.Printf("My TrainTime : %d seconds \n",time)
}

//export Broadcast_BC
func Broadcast_BC(src *C.char, size C.int, opcode byte) {
	data := C.GoBytes(unsafe.Pointer(src), C.int(size))
	overlay.Broadcast_BC(node.ID(), data, opcode)
}

//export Multicast_GR
func Multicast_GR(src *C.char, size C.int, opcode byte) {
    data := C.GoBytes(unsafe.Pointer(src), C.int(size))
    overlay.Multicast_GR(node.ID(), data, opcode)
}

//export Fedcomp_GR 
func Fedcomp_GR(src *C.char, size C.int, opcode byte) {
    data := C.GoBytes(unsafe.Pointer(src), C.int(size))
    usermsg := userMessage{Opcode: opcode, Contents: data}
    overlay.Fedcomp_GR(node.ID(), usermsg.Marshal())
}

//export Report_GR
func Report_GR(src *C.char, size C.int, opcode byte, aggregation int) {
    data := C.GoBytes(unsafe.Pointer(src), C.int(size))
    usermsg := userMessage{Opcode: opcode, Contents: data}
    overlay.Report_GR(node.ID(), usermsg.Marshal(), aggregation)
}

// user provides sort function for perf-aware node arrangement in grouping
//TODO : use interface for user defined Subscriber struct
//func sort_by_metric(list map[string]interface{}) []Subscriber {
func sort_by_metric(list map[string]dual.Subscriber) []dual.Subscriber {
        arr := make([]dual.Subscriber, 0)
        for _, tx := range list {
            //arr = append(arr, tx.(Subscriber))
            arr = append(arr, tx)
            //fmt.Printf("sublist[%s] traintime:%d numleave:%d\n",tx.UserInfo.(UserInfo).UserID.ID.String()[:8],tx.UserInfo.(UserInfo).TrainTime, tx.UserInfo.(UserInfo).NumLeave)
        }
	if current_round > 0 {
            sort.Slice(arr, func(i, j int) bool {
            //if arr[i].UserInfo.(UserInfo).TrainTime != arr[j].UserInfo.(UserInfo).TrainTime {
                //return arr[i].UserInfo.(UserInfo).TrainTime >= arr[j].UserInfo.(UserInfo).TrainTime //best case, descending order
                return arr[i].UserInfo.(UserInfo).TrainTime <= arr[j].UserInfo.(UserInfo).TrainTime //worst vase, ascending order
            //}
            })
	}

        /*
        sort.Slice(arr, func(i, j int) bool {
            if arr[i].Id != arr[j].Id {
                return arr[i].Id < arr[j].Id
            }
            return arr[i].Priority < arr[j].Priority
        })
	*/
        return arr
}

func gossip_FedComp(data []byte, avoid []string) {
     uuid := dual.CreateUUID()
     gossip_BC_FedComp(uuid, data, avoid)
     //gossip_BC_FedComp_FirstN(uuid, data, avoid)
}

func gossip_BC_FedComp_FirstN(uuid string, data []byte, avoid []string) {
     euuid, _ := json.Marshal(uuid)
     peers := overlay.Table_bc().KEntries(max_peers)
     if len(avoid) > 0 {
          peers = overlay.Remove(overlay.Table_bc().KEntries(max_peers),avoid)
     }

     len_peers := len(peers)
     N := math.Sqrt(float64(len_peers))

     for k, id := range peers {
	 if k < int(N) {
           ctx, cancel := context.WithCancel(context.Background())
           //ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
           err := node.SendMessage(ctx, id.Address, dual.GossipMessage{Opcode: dual.OP_FED_COMP_PULL_REQ, UUID: euuid, Contents: data })
	   /*
           fmt.Printf("Send Pull Req message to %s(%s)\n",
                      id.Address,
                      id.ID.String()[:printedLength],
           )
	   */
           cancel()
           if err != nil {
               fmt.Printf("Failed to send message to %s(%s). Skipping... [error: %s]\n",
                         id.Address,
                          id.ID.String()[:printedLength],
                          err,
               )
               continue
           }
	 }else{
             ctx, cancel := context.WithCancel(context.Background())
             //ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
             err := node.SendMessage(ctx, id.Address, dual.GossipMessage{Opcode: dual.OP_FED_COMP_PUSH, UUID: euuid, Contents: data })
	     /*
             fmt.Printf("Send Push message to %s(%s)\n",
                   id.Address,
                   id.ID.String()[:printedLength],
             )
	     */
             cancel()
             if err != nil {
                 fmt.Printf("Failed to send message to %s(%s). Skipping... [error: %s]\n",
                         id.Address,
                         id.ID.String()[:printedLength],
                         err,
                 )
                 continue
             }
	 }
      }
}

func gossip_BC_FedComp(uuid string, data []byte, avoid []string) {
     euuid, _ := json.Marshal(uuid)
     __gossip_BC_FedComp(euuid, data, avoid)
}

func __gossip_BC_FedComp(euuid []byte, data []byte, avoid []string) {
     peers := overlay.Table_bc().KEntries(max_peers)
     if len(avoid) > 0 {
          peers = overlay.Remove(overlay.Table_bc().KEntries(max_peers),avoid)
     }

     len_peers := len(peers)
     //push := math.Sqrt(float64(len_peers))
     push := len_peers/2
     if push == 0 {
         push = 1
     }
     pushlist, pulllist := overlay.FindRandMember(push, len_peers, peers)

     list := make([]string,0)
     for _, id := range pushlist {
	 list = append(list,id.ID.String())
     }
     list = append(list,avoid...)

     listdata, _ := json.Marshal(list)
     length := len(listdata)

     for _, id := range pushlist {
	 overlay.Sendmsg(euuid, data, length, listdata, id, dual.OP_FED_COMP_PUSH, dual.NOOP, GRING.ID{})
     }

     for _, id := range pulllist {
	 overlay.Sendmsg(euuid, data, length, listdata, id, dual.OP_FED_COMP_PULL_REQ, dual.NOOP, GRING.ID{})
     }
}

func sendAD() {
     // TODO : select project to work on

     uuid := dual.CreateUUID() // ad ID. Will be assigned to GossipMessage.
     fmt.Printf("\nAD created. ID:%s \n",uuid[:printedLength])

     epjtid, _ := json.Marshal(my_pjt_id) //string to []byte
     admsg := dual.ADMessage{PjtID: epjtid, PubID:node.ID() }

     fmt.Printf("Broadcast AD message, msg.Type:%v\n",dual.MSG_TYPE_ADMSG)
     list := make([]string,0)
     overlay.Gossip_BC(uuid, admsg.Marshal(), list, dual.MSG_TYPE_ADMSG, GRING.ID{})
}

func on_group_done(p2pmsg dual.P2pMessage) {
     fmt.Printf("Receive OP_JOIN_GROUP value : %d\n",p2pmsg.Aggregation)
     duration := time.Since(starttime)
     fmt.Printf("DONE Grouping %d peers time:%s\n", p2pmsg.Aggregation, duration)
     current_round = current_round + 1
}

func on_recv_gossip_msg(msg dual.GossipMessage) {
    //fmt.Printf("on_recv_gossip_msg msg.Type:%v \n",msg.Type)

    if msg.Type == dual.MSG_TYPE_ADMSG && isPublisher == false{
        //fmt.Printf("on_recv_gossip_msg : msg type: %v dual.MSG_TYPE_ADMSG \n", msg.Type)

        admsg, err := dual.UnmarshalADMessage(msg.Contents)
        if err != nil {
            fmt.Printf("on_recv_gossip_msg : not ok. Fail to unmarshal AD msg\n",)
            return
        }
        fmt.Printf("AD received. ADID:%s PjtID:%s, PubID:%s\n",msg.UUID[:printedLength], admsg.PjtID[:printedLength], admsg.PubID.ID.String()[:printedLength])

        _, found := subscribed_pjt[string(admsg.PjtID)]
	if found {
	    return
	}else{
	    subscribed_pjt[string(admsg.PjtID)] = admsg.PubID.ID.String()
	}

        // TODO : this delay is for test. overlay.setItem setUUID should process multiple to avoid conflict between AD and SUBS 
	//time.Sleep(time.Duration(2) * time.Second)

	//rnum := rand.Intn(30)
	//time.Sleep(time.Duration(rnum) * time.Second)

        fmt.Printf("Join project. Gossip Subs msg. SubID:%s, SubAddr:%s len(Addr):%d \n",node.ID().ID.String()[:printedLength], node.ID().Address, len(node.ID().Address))

        respmsg := dual.SubsMessage{SubID: node.ID(), Host: node.ID().Host, Port: node.ID().Port}
        uuid := dual.CreateUUID()
	// option 1 : relay message
        //ctx, cancel := context.WithCancel(context.Background())
        //closest := overlay.Table_bc().FindClosest(admsg.PubID.ID, 1) // find one node that is the closest to the target(Pub)
        //err = node.SendMessage(ctx, closest[0].Address, dual.RelayMessage{UUID: euuid, TargetID: admsg.PubID, Contents: respmsg.Marshal() })

        // option 2 : gossip broadcast
        list := make([]string,0)
        overlay.Gossip_BC(uuid, respmsg.Marshal(), list, dual.MSG_TYPE_SUBSMSG, admsg.PubID) // set targetID to publisher so that publisher can receive msg

        return
    }

    if msg.Type == dual.MSG_TYPE_SUBSMSG && isPublisher == true{
        //fmt.Printf("on_recv_gossip_msg : msg type: %v dual.MSG_TYPE_SUBSMSG \n", msg.Type)

	// check if I am the target
	if msg.TargetID.ID.String() != node.ID().ID.String() {
	    //fmt.Printf("I am not the target me:%s target:%s\n",node.ID().ID.String()[:printedLength], msg.TargetID.ID.String()[:printedLength])
            return
        }
	//fmt.Printf("I am the target me:%s target:%s\n",node.ID().ID.String()[:printedLength], msg.TargetID.ID.String()[:printedLength])

        subsmsg, err := dual.UnmarshalSubsMessage(msg.Contents)
        if err != nil {
            fmt.Printf("on_recv_gossip_msg : not ok. Fail unmarshal subs msg\n",)
            return
        }
        //fmt.Printf("recv subscriber SubID:%s, SubAddr:%s\n",subsmsg.SubID.ID.String()[:printedLength], subsmsg.Addr)

	addr := net.JoinHostPort(subsmsg.Host.String(), strconv.FormatUint(uint64(subsmsg.Port), 10))
        pjt, found := overlay.PubSub().GetProject(my_pjt_id)
	if found {
            pjt.AddSubscriber(subsmsg.SubID.ID.String(), dual.Subscriber{subsmsg.SubID.ID.String(), addr, UserInfo{}})
            fmt.Printf("new subscriber %s(%s) total: %d\n",addr, subsmsg.SubID.ID.String()[:printedLength],pjt.Size())
	}else{
            fmt.Printf("Recv new subscriber %s(%s) but no project exists\n",addr, subsmsg.SubID.ID.String()[:printedLength])
	}

        return
    }
}

func myfedcomputationpush(msg dual.GossipMessage, ctx GRING.HandlerContext) {
    //fmt.Printf("recv OP_FED_COMP_PUSH RECEIVE ITEM count:%d\n",msg.Count)

    list := make([]string,0)
    if msg.Count != 0{
        err := json.Unmarshal(msg.List, &list)
        if err != nil {
            fmt.Printf("Error on decode process: %v\n", err)
        }
	//fmt.Println(list)
    }

    // send push and pull req
    list = append(list,ctx.ID().ID.String())
    __gossip_BC_FedComp(msg.UUID, msg.Contents, list)
    //go gossip_BC_FedComp_FirstN(msg.UUID, msg.Contents, ctx.ID())

    if overlay.GetID() == dual.ID_INITIATOR || overlay.GetID() == dual.ID_SUBLEADER {
        //fmt.Printf("I am an initiator or a subleader. wait for workers.\n")
    }else{ // worker
        //fmt.Printf("I am a workers. I report.\n")
        usermsg := userMessage{Contents: msg.Contents}
        go overlay.Report_GR(node.ID(), usermsg.Marshal(), 1)
    }
}

func myfedcomputationpullreq(msg dual.GossipMessage, ctx GRING.HandlerContext) {
    //fmt.Printf("recv OP_FED_COMP_PULL_REQ count:%d\n",msg.Count)

    //go func() {
        list := make([]string,0)
        if msg.Count != 0{
            err := json.Unmarshal(msg.List, &list)
            if err != nil {
                fmt.Printf("Error on decode process: %v\n", err)
            }
	    //fmt.Println(list)
        }

        maxtry := 5
        i := 0
        for i < maxtry {
            // RequestMessage to download data
            dmsg, rst := overlay.RequestDownload(msg.UUID, list)

            if rst {
                //fmt.Printf("Got Download RECEIVE ITEM\n")
                // save item for pull request
                overlay.SetItem(dmsg)
                overlay.SetUUID(dmsg.UUID)

                // do fed
                if overlay.GetID() == dual.ID_INITIATOR || overlay.GetID() == dual.ID_SUBLEADER {
                    //fmt.Printf("I am an initiator or a subleader. wait for workers.\n")
                }else{ // worker
                    // report back dummy data
                    //fmt.Printf("I am a workers. I report.\n")
                    usermsg := userMessage{Contents: msg.Contents}
                    go overlay.Report_GR(node.ID(), usermsg.Marshal(), 1)
                }
	        return
            }else{
                //fmt.Printf("fail to download.\n")
	        time.Sleep(300 * time.Millisecond)
            }
            i++
         }
    //}()
}

func on_fedcomp(p2pmsg dual.P2pMessage) {
        msg, err := unmarshalUserMessage(p2pmsg.Contents)
	if err != nil {
	    fmt.Printf("myfedcomp : not ok. Fail recv app msg\n",)
	    return
	}

	fmt.Printf("myfedcomp : recv app msg opcode : %v\n", msg.Opcode)
	ptr := unsafe.Pointer(&msg.Contents[0])

        // relay the msg
        if overlay.GetID() == dual.ID_INITIATOR {
            //fmt.Printf("OP_FED_COMP. I am initiator\n")

	    if msg.Opcode == OP_GLOBAL_MODEL {
	        fmt.Printf("GO INIT : recv OP_GLOBAL_MODEL\n")
                C.call_c_func(callbacks.on_globalmodel, (*C.char)(ptr) )
		// no relay this message. only initiator receives this.
		return
	    }
	}

	if overlay.GetID() == dual.ID_SUBLEADER {
            //fmt.Printf("OP_FED_COMP. I am subleader\n")

            if msg.Opcode == OP_INIT {
                fmt.Printf("GO SUB : recv OP_INIT\n")
                //pass the opcode down to sub groups if any
                overlay.Multicast_GR(node.ID(), p2pmsg.Contents, dual.OP_FED_COMP)

                // generate local dataset in python
                C.call_c_func(callbacks.on_init_subleader, (*C.char)(ptr) )
                return
            }

            if msg.Opcode == OP_REQUEST_UPDATE {
                fmt.Printf("GO SUB : recv OP_REQUEST_UPDATE\n")
                // pass the opcode down
                overlay.Multicast_GR(node.ID(), p2pmsg.Contents, dual.OP_FED_COMP)

                // train my model
                C.call_c_func(callbacks.on_requestupdate_subleader, (*C.char)(ptr) )
                return
            }

            if msg.Opcode == OP_STOP_AND_EVAL {
                fmt.Printf("GO SUB : recv OP_STOP_AND_EVAL\n")
                // pass the opcode down
                overlay.Multicast_GR(node.ID(), p2pmsg.Contents, dual.OP_FED_COMP)

                //python code makes my eval ready
                C.call_c_func(callbacks.on_stopandeval_subleader, (*C.char)(ptr) )
                fmt.Printf("GO WORKER : recv OP_INIT\n")
                return
            }
        }

        if overlay.GetID() == dual.ID_WORKER {

            if msg.Opcode == OP_INIT {
	        fmt.Printf("GO WORKER : recv OP_INIT\n")
	        C.call_c_func(callbacks.on_init_worker, (*C.char)(ptr) )
		return
	    }

            if msg.Opcode == OP_REQUEST_UPDATE {
	        fmt.Printf("GO WORKER : recv OP_REQUEST_UPDATE\n")
		// train my model
                C.call_c_func(callbacks.on_requestupdate_worker, (*C.char)(ptr) )
		return
            }

            if msg.Opcode == OP_STOP_AND_EVAL {
	        fmt.Printf("GO WORKER : recv OP_STOP_AND_EVAL\n")
                C.call_c_func(callbacks.on_stopandeval_worker, (*C.char)(ptr) )
		return
            }

	}

        fmt.Printf("on_fedcomp : Unknown message\n")
}

func on_report(p2pmsg dual.P2pMessage) {
        ptr := unsafe.Pointer(nil)
        msg, err := unmarshalUserMessage(p2pmsg.Contents)
	if err != nil {
	    fmt.Printf("on_report : not ok. Fail recv app msg\n",)
	    return
	}

	fmt.Printf("on_report : recv app msg opcode : %v\n", msg.Opcode)
	if len(msg.Contents) != 0 {
            ptr = unsafe.Pointer(&msg.Contents[0])
	}else{
	    fmt.Printf("on_report : no contents\n")
	}

        if overlay.GetID() == dual.ID_PUBLISHER {

            if msg.Opcode == OP_CLIENT_UPDATE {
	        fmt.Printf("GO PUB : recv OP_CLIENT_UPDATE\n")

                fmt.Printf("call client update done publisher\n")
                C.call_c_func(callbacks.on_clientupdatedone_publisher, (*C.char)(ptr))

                duration := time.Since(starttime)
                fmt.Printf("DONE round %d training. time:%s\n",current_round, duration)
	    }

            if msg.Opcode == OP_CLIENT_EVAL {
	        fmt.Printf("GO PUB : recv OP_CLIENT_EVAL\n")

                fmt.Printf("call client eval done publisher\n")
                C.call_c_func(callbacks.on_clientevaldone_publisher, (*C.char)(ptr))

	        // set project state commited
                pjt, found := overlay.PubSub().GetProject(my_pjt_id)
                if found {
                    pjt.SetPjtState(dual.STATE_COMMITTED)
                }
                duration := time.Since(starttime)
                fmt.Printf("DONE round %d evaluation. time:%s\n",current_round, duration)
	    }

            return
	}

        // FL server
        if overlay.GetID() == dual.ID_INITIATOR {

            if msg.Opcode == OP_CLIENT_READY {
	        fmt.Printf("GO INIT : recv OP_CLIENT_READY\n")

		mutex_client_ready.Lock()
                value := p2pmsg.Aggregation
	        //fmt.Printf("OP_REPORT value : %d\n",value)
	        num_client_ready += int(value)
                fmt.Printf("FL Server receives OP_REPORT num_client_ready current: %d, total expected : %d\n",num_client_ready,  overlay.GetTotMembers())
	        if num_client_ready == overlay.GetTotMembers()+1 {
                    // reset the value after reporting
		    num_client_ready = 0
		    mutex_client_ready.Unlock()

                    fmt.Printf("call on_train_next_round\n")
                    C.call_c_func(callbacks.on_trainnextround, (*C.char)(ptr))

		    // train my model
                    C.call_c_func(callbacks.on_trainmymodel, (*C.char)(ptr))

		    // done timer
		    doneTimer <- true

		    return
	        }else{
		    mutex_client_ready.Unlock()
		}

		mutexReportReady.Lock()
		if isFirstReportReady {
		    isFirstReportReady = false
		    mutexReportReady.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationReady) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO INIT : start timerReady\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO INIT : restart timerReady")
				    timerReportChecker.Reset(time.Duration(timerDurationReady) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO INIT : timerReady is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO INIT : TimerReady fired at", t)
		                    // checking report
                                    fmt.Printf("GO INIT : checkReport num_client_ready current: %d, total expected : %d\n",num_client_ready,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_ready.Lock()
                                    if num_client_ready < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_ready.Unlock()
                                        // if yes(under), reduce teh tolerance rate
					toleranceRate -= toleranceReduceStep
	                                // then continue the loop and wait for next timer. next timer will check.
				        fmt.Println("GO INIT : restart timerReady")
				        timerReportChecker.Reset(time.Duration(timerDurationReady) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer.
                                        // reset the value after reporting
                                        num_client_ready = 0
                                        mutex_client_ready.Unlock()

                                        fmt.Printf("GO INIT : call on_train_next_round\n")
                                        ptr := unsafe.Pointer(nil)
                                        C.call_c_func(callbacks.on_trainnextround, (*C.char)(ptr))

                                        // train my model
                                        C.call_c_func(callbacks.on_trainmymodel, (*C.char)(ptr))

	                                fmt.Printf("GO INIT : timerReady is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
		    mutexReportReady.Unlock()
	            resetTimer <- true
		} // isFirstReportReady

		return
            }

            if msg.Opcode == OP_CLIENT_UPDATE {
	        fmt.Printf("GO INIT : recv OP_CLIENT_UPDATE\n")
		// python code gathers updates
                C.call_c_func(callbacks.on_clientupdate_initiator, (*C.char)(ptr))

                value := p2pmsg.Aggregation
		mutex_client_update.Lock()
	        num_client_update += int(value)
                fmt.Printf("GO INIT : FL Server receives OP_REPORT num_client_update current: %d, total expected : %d\n", num_client_update, overlay.GetTotMembers())

	        if num_client_update == (overlay.GetTotMembers()+1) { // we need to report to publisher, so we count me as well.
                    // reset the value after reporting
		    num_client_update = 0
		    mutex_client_update.Unlock()

                    fmt.Printf("GO INIT : call client update done initiator\n")
		    // watch out nested lock on_clientupdatedone_initiator 
                    C.call_c_func(callbacks.on_clientupdatedone_initiator, (*C.char)(ptr))

		    // done timer
		    doneTimer <- true
	        }else{
		    mutex_client_update.Unlock()
		}

		mutexReportUpdate.Lock()
		if isFirstReportUpdate {
		    isFirstReportUpdate = false
		    mutexReportUpdate.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationUpdate) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO INIT : start timerUpdate\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO INIT : restart timerUpdate")
				    timerReportChecker.Reset(time.Duration(timerDurationUpdate) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO INIT : timerUpdate is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO INIT : TimerUpdate fired at", t)
		                    // checking report
                                    fmt.Printf("GO INIT : checkReportUpdate: num_client_update current: %d, total expected : %d\n",num_client_update,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_update.Lock()
                                    if num_client_update < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_update.Unlock()
                                        // if yes(under), reduce teh tolerance rate
					toleranceRate -= toleranceReduceStep

	                                // then continue the loop and wait for next timer. next timer will check.
		                        fmt.Println("GO INIT : restart timerUpdate")
				        timerReportChecker.Reset(time.Duration(timerDurationUpdate) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer.
                                        // reset the value after reporting
                                        num_client_update = 0
                                        mutex_client_update.Unlock()

                                        ptr := unsafe.Pointer(nil)
                                        fmt.Printf("GO INIT : call client update done initiator\n")
		                        // watch out nested lock on_clientupdatedone_initiator 
                                        C.call_c_func(callbacks.on_clientupdatedone_initiator, (*C.char)(ptr))

	                                fmt.Printf("GO INIT : timerUpdate is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
		    mutexReportUpdate.Unlock()
	            resetTimer <- true
		} // isFirstReportUpdate

		return
            }

            if msg.Opcode == OP_CLIENT_EVAL {
	        fmt.Printf("GO INIT : recv OP_CLIENT_EVAL\n")
		// python code gathers updates
                C.call_c_func(callbacks.on_clienteval_initiator, (*C.char)(ptr) )

                value := p2pmsg.Aggregation
		mutex_client_eval.Lock()
	        num_client_eval += int(value)
                fmt.Printf("GO INIT : FL Server receives OP_REPORT num_client_eval current: %d, total expected : %d\n",num_client_eval, overlay.GetTotMembers())

	        if num_client_eval == (overlay.GetTotMembers()+1) {
                    // reset the value after reporting
		    num_client_eval = 0
		    mutex_client_eval.Unlock()

	            fmt.Printf("GO INIT : call client eval done initiator\n")
		    // python code will aggreagte and report
		    // watch out nested lock on_clientevaldone_initiator
                    C.call_c_func(callbacks.on_clientevaldone_initiator, (*C.char)(ptr))

		    // done timer
		    doneTimer <- true
	        }else{
		    mutex_client_eval.Unlock()
		}

                mutexReportEval.Lock()
		if isFirstReportEval {
		    isFirstReportEval = false
                    mutexReportEval.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationEval) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO INIT : start timerEval\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO INIT : restart timerEval")
				    timerReportChecker.Reset(time.Duration(timerDurationEval) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO INIT : timerEval is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO INIT : TimerEval fired at", t)
		                    // checking report
                                    fmt.Printf("GO INIT : checkReportUpdate: num_client_eval current: %d, total expected : %d\n",num_client_eval,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_eval.Lock()
                                    if num_client_eval < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_eval.Unlock()
                                        // if yes(under), reduce teh tolerance rate
	                                // then continue the loop and wait for next timer. next timer will check.
					toleranceRate -= toleranceReduceStep
				        fmt.Println("GO INIT : restart timerEval")
				        timerReportChecker.Reset(time.Duration(timerDurationEval) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer.
                                        // reset the value after reporting
                                        num_client_eval = 0
                                        mutex_client_eval.Unlock()

                                        ptr := unsafe.Pointer(nil)
	                                fmt.Printf("GO INIT : call client eval done initiator\n")
		                        // python code will aggreagte and report
		                        // watch out nested lock on_clientevaldone_initiator
                                        C.call_c_func(callbacks.on_clientevaldone_initiator, (*C.char)(ptr))

                                        fmt.Printf("GO INIT : timerEval is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
                    mutexReportEval.Unlock()
	            resetTimer <- true
		} // isFirstReportEval

		return
            }
        }

	if overlay.GetID() == dual.ID_SUBLEADER {
            //fmt.Printf("Receive OP_REPORT as an sub-leader.\n")

            if msg.Opcode == OP_CLIENT_READY {
                fmt.Printf("GO SUB : recv OP_CLIENT_READY\n")

                mutex_client_ready.Lock()
                value := p2pmsg.Aggregation
                num_client_ready += int(value)
                fmt.Printf("GO SUB : Sub-leader receives OP_REPORT num_client_ready current: %d, total expected : %d\n",num_client_ready, overlay.GetTotMembers())
                if num_client_ready == (overlay.GetTotMembers()+1) {
                    // reset the value after reporting
		    num := num_client_ready
                    num_client_ready = 0
                    mutex_client_ready.Unlock()

                    usermsg := userMessage{Opcode: OP_CLIENT_READY, Contents: nil}
                    overlay.Report_GR(node.ID(), usermsg.Marshal(), num)
                    fmt.Printf("GO SUB : Send report to upper leader %s opcode: OP_REPORT value : %d\n",overlay.GetLeader(),num)

		    // done timer
		    doneTimer <- true
                }else{
                    mutex_client_ready.Unlock()
                }

		mutexReportReady.Lock()
		if isFirstReportReady {
		    isFirstReportReady = false
		    mutexReportReady.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationReady) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO SUB : start timerReady\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO SUB : restart timerReady")
				    timerReportChecker.Reset(time.Duration(timerDurationReady) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO SUB : timerReady is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO SUB : TimerReady fired at", t)
		                    // checking report
                                    fmt.Printf("GO SUB : checkReport num_client_ready current: %d, total expected : %d\n",num_client_ready,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_ready.Lock()
                                    if num_client_ready < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_ready.Unlock()
                                        // if yes(under), reduce teh tolerance rate
	                                // then continue the loop and wait for next timer fire. next timer will check.
					toleranceRate -= toleranceReduceStep
				        fmt.Println("GO SUB : restart timerReady")
				        timerReportChecker.Reset(time.Duration(timerDurationReady) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer
                                        // reset the value after reporting
		                        num := num_client_ready
		                        num_client_ready = 0
		                        mutex_client_ready.Unlock()

		                        usermsg := userMessage{Opcode: OP_CLIENT_READY, Contents: nil}
		                        overlay.Report_GR(node.ID(), usermsg.Marshal(), num)
				        fmt.Printf("Send report to upper leader %s opcode: OP_REPORT value : %d\n",overlay.GetLeader(),num)

	                                fmt.Printf("GO SUB : timerReady is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
		    mutexReportReady.Unlock()
	            resetTimer <- true
		} // isFirstReportReady

                return
            }//OP_CLIENT_READY

	    if msg.Opcode == OP_CLIENT_UPDATE {
                fmt.Printf("GO SUB : recv OP_CLIENT_UPDATE\n")
                // python code gathers updates
                C.call_c_func(callbacks.on_clientupdate_subleader, (*C.char)(ptr) )

                mutex_client_update.Lock()
                value := p2pmsg.Aggregation
                num_client_update += int(value)
                fmt.Printf("GO SUB : Sub-leader receives OP_REPORT num_client_update current: %d, total expected : %d\n",num_client_update, overlay.GetTotMembers())

                if num_client_update == (overlay.GetTotMembers()+1) {
                    // reset the value after reporting
		    num := num_client_update
                    num_client_update = 0
                    mutex_client_update.Unlock()

                    fmt.Printf("GO SUB : Send report to upper leader %s opcode: OP_REPORT value : %d\n",overlay.GetLeader(),num)
                    // python code will aggreagte and report as well. 
                    C.call_c_func2(callbacks.on_reportclientupdate, (C.int)(num))

		    // done timer
		    doneTimer <- true
                }else{
                    mutex_client_update.Unlock()
                }

		mutexReportUpdate.Lock()
		if isFirstReportUpdate {
		    isFirstReportUpdate = false
		    mutexReportUpdate.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationUpdate) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO SUB : start timerUpdate\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO SUB : restart timerUpdate")
				    timerReportChecker.Reset(time.Duration(timerDurationUpdate) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO SUB : timerUpdate is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO SUB : TimerUpdate fired at", t)
		                    // checking report
                                    fmt.Printf("GO SUB : checkReport num_client_update current: %d, total expected : %d\n",num_client_update,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_update.Lock()
                                    if num_client_update < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_update.Unlock()
                                        // if yes(under), reduce teh tolerance rate
	                                // then continue the loop and wait for next timer. next timer will check.
					toleranceRate -= toleranceReduceStep

				        fmt.Println("GO SUB : restart timerUpdate")
				        timerReportChecker.Reset(time.Duration(timerDurationUpdate) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer.
                                        // reset the value after reporting
                                        num := num_client_update
					num_client_update = 0
					mutex_client_update.Unlock()

                                        fmt.Printf("Send report to upper leader %s opcode: OP_REPORT value : %d\n",overlay.GetLeader(),num)
	                                // python code will aggreagte and report as well. 
	                                C.call_c_func2(callbacks.on_reportclientupdate, (C.int)(num))

                                        fmt.Printf("GO SUB : timerUpdate is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
		    mutexReportUpdate.Unlock()
	            resetTimer <- true
		} // isFirstReportUpdate

                return
            }

	    if msg.Opcode == OP_CLIENT_EVAL {
                fmt.Printf("GO SUB : recv OP_CLIENT_EVAL\n")
                // python code gathers eval
                C.call_c_func(callbacks.on_clienteval_subleader, (*C.char)(ptr) )

                mutex_client_eval.Lock()
                value := p2pmsg.Aggregation
                num_client_eval += int(value)
                fmt.Printf("GO SUB : Sub-leader receives OP_REPORT num_client_eval current: %d, total expected : %d\n",num_client_eval, overlay.GetTotMembers())

                if num_client_eval == (overlay.GetTotMembers()+1) {
                    // reset the value after reporting
		    num := num_client_eval
                    num_client_eval = 0
                    mutex_client_eval.Unlock()

                    fmt.Printf("GO SUB : call report client eval\n")
                    // python code will aggreagte and report
                    C.call_c_func2(callbacks.on_reportclienteval, (C.int)(num))

		    // done timer
		    doneTimer <- true
                }else{
                    mutex_client_eval.Unlock()
                }

                mutexReportEval.Lock()
		if isFirstReportEval {
		    isFirstReportEval = false
                    mutexReportEval.Unlock()
		    // start the timer
		    timerReportChecker = time.NewTimer(time.Duration(timerDurationEval) * time.Second)
                    doneTimer = make(chan bool)
                    resetTimer = make(chan bool)
		    toleranceRate = 100

                    go func () {
                        fmt.Printf("GO SUB : start timerEval\n")
                        for {
	                    select {
	                        case <-resetTimer:
				    //if !timerReportChecker.C.Stop() {
				    //    <-timerReportChecker.C
			            //}t
				    fmt.Println("GO SUB : restart timerEval")
				    timerReportChecker.Reset(time.Duration(timerDurationEval) * time.Second)
	                        case <-doneTimer:
	                            fmt.Printf("GO SUB : timerEval is done\n")
		                    return
	                        case t := <-timerReportChecker.C:
	                            fmt.Println("GO SUB : TimerEval fired at", t)
		                    // checking report
                                    fmt.Printf("GO SUB : checkReport: num_client_eval current: %d, total expected : %d\n",num_client_eval,  overlay.GetTotMembers() * toleranceRate/100)
                                    // check if it is under current rate. It starts with 100% (1.0)
                                    mutex_client_eval.Lock()
                                    if num_client_eval < (overlay.GetTotMembers() * toleranceRate/100) {
                                        mutex_client_eval.Unlock()
                                        // if yes(under), reduce teh tolerance rate
	                                // then continue the loop and wait for next timer. next timer will check.
					toleranceRate -= toleranceReduceStep

				        fmt.Println("GO SUB : restart timerEval")
				        timerReportChecker.Reset(time.Duration(timerDurationEval) * time.Second)
                                    }else{
                                        // if no(enough), we report. and quit the timer.
                                        // reset the value after reporting
                                        num_client_eval = 0
                                        mutex_client_eval.Unlock()

                                        ptr := unsafe.Pointer(nil)
	                                fmt.Printf("GO SUB : call client eval done initiator\n")
		                        // python code will aggreagte and report
		                        // watch out nested lock on_clientevaldone_initiator
                                        C.call_c_func(callbacks.on_clientevaldone_initiator, (*C.char)(ptr))

	                                fmt.Printf("GO SUB : timerEval is done\n")
		                        return
                                    }
	                    }//select
                        }//for
                    }()// go func

		}else {
                    mutexReportEval.Unlock()
	            resetTimer <- true
		} // isFirstReportEval

                return
            }
	}

        fmt.Printf("on_report : Unknown message\n")
}

/*
// handle handles and prints out valid messages from peers.
func handle(ctx GRING.HandlerContext) error {
	if ctx.IsRequest() {
		return nil
	}

	obj, err := ctx.DecodeMessage()
	if err != nil {
	        fmt.Printf("Decode fail recv msg from %s(%s)\n", ctx.ID().Address, ctx.ID().ID.String()[:printedLength])
		return nil
	}
	switch msg := obj.(type) {
            case userMessage:
                msg, ok := obj.(userMessage)
	        if !ok {
	            fmt.Printf("not ok fail recv app msg from %s(%s)\n", ctx.ID().Address, ctx.ID().ID.String()[:printedLength])
		    return nil
	        }

	        fmt.Printf("recv app msg from %s(%s) opcode : %v\n", ctx.ID().Address, ctx.ID().ID.String()[:printedLength], msg.Opcode)

	    case dual.P2pMessage:
                msg, ok := obj.(dual.P2pMessage)
	        if !ok {
	            fmt.Printf("not ok fail recv p2p msg from %s(%s)\n", ctx.ID().Address, ctx.ID().ID.String()[:printedLength])
	            return nil
	        }

	        fmt.Printf("recv app msg from %s(%s) opcode : %v\n", ctx.ID().Address, ctx.ID().ID.String()[:printedLength], msg.Opcode)
	}//switch

	return nil
}
*/


// check panics if err is not nil.
func check(err error) {
	if err != nil {
		panic(err)
	}
}

//export Init_p2p
func Init_p2p(host *C.char, port int, isPub int, serveraddr *C.char, traintime int, numleave int) {
    mytraintime = traintime
    mynumleave = numleave

    init_p2p(C.GoString(host), port, C.GoString(serveraddr))
    if isPub == 1 {
	isPublisher = true
        overlay.SetID(dual.ID_PUBLISHER)
    }else{
	isPublisher = false
        overlay.SetID(dual.ID_NOBODY)
    }
}

func init_p2p(host string, port int, serveraddr string){
        var err error

	mutex_client_ready = new(sync.Mutex)
	mutex_client_update = new(sync.Mutex)
	mutex_client_eval = new(sync.Mutex)
	mutexReportReady = new(sync.Mutex)
	mutexReportUpdate = new(sync.Mutex)
	mutexReportEval = new(sync.Mutex)

	report = false

	subscribed_pjt = make(map[string]string,0)

	//fmt.Printf("host : %s port : %d \n",host,port)

        logger, _ := zap.NewProduction()
	// Create a new configured node.
	node, err = GRING.NewNode(
		GRING.WithNodeBindHost(net.ParseIP(host)),
		GRING.WithNodeBindPort(uint16(port)),
		GRING.WithNodeMaxRecvMessageSize(1<<24), //16MB
		//GRING.WithNodeMaxRecvMessageSize(1<<28), //256MB
		GRING.WithNodeMaxInboundConnections(uint(max_conn)),
                GRING.WithNodeMaxOutboundConnections(uint(max_conn)),
		GRING.WithNodeLogger(logger),
	)
	check(err)

	// Register the userMessage Go type to the node with an associated unmarshal function.
        node.RegisterMessage(userMessage{}, unmarshalUserMessage)

	// Register a message handler to the node.
	// we don't use it for now.
	//node.Handle(handle)

	// Instantiate dual.
	events = dual.Events{
	        OnPeerAdmitted_bc: func(id GRING.ID) {
		    /*
		        if isPublisher {
		            //fmt.Printf("[peer_bc]Learned about a new peer %s(%s).\n", id.Address, id.ID.String()[:printedLength])
                            pjt, found := overlay.PubSub().GetProject(my_pjt_id)
	                    if found {
                                pjt.AddSubscriber(id.ID.String(), dual.Subscriber{id.ID.String(), id.Address, UserInfo{}})
                                fmt.Printf("new subscriber %s(%s) total: %d \n",id.Address, id.ID.String()[:printedLength],pjt.Size())
	                    }else{
			        fmt.Printf("No project found. Create project first\n")
	                        pjtid, pjt := overlay.PubSub().NewProject(node.ID())
		                my_pjt_id = pjtid
		                fmt.Printf("project[%s] created. subscribers : %d\n",my_pjt_id[:printedLength], pjt.Size())

                                pjt.AddSubscriber(id.ID.String(), dual.Subscriber{id.ID.String(), id.Address, UserInfo{}})
                                fmt.Printf("new subscriber %s(%s) total: %d \n",id.Address, id.ID.String()[:printedLength],pjt.Size())
			    }
			}
			*/
	        },
	        OnPeerAdmitted_gr: func(id GRING.ID) {
	                //fmt.Printf("[peer_gr]Learned about a new peer %s(%s).\n", id.Address, id.ID.String()[:printedLength])
		},
		OnPeerEvicted: func(id GRING.ID) {
			//fmt.Printf("Forgotten a peer %s(%s).\n", id.Address, id.ID.String()[:printedLength])
		},
		/*
                // define your own group functions otherwise default dual group functions will be used
                OnRequestGroup: on_request_group_fl_server,
                OnRequestGroupSub: on_request_group_subleader,
                OnRequestJoin: on_request_join_worker,
                OnJoinGroup: on_join_group,
                */
		OnSortByMetric: sort_by_metric,
		OnQueryGroupInfo: on_query_group_info,
		OnAggregateGroupInfo: on_aggregate_group_info,
                OnGroupDone: on_group_done, // if you want to do something after grouping is done, this is what is called when it is done.

                OnFedComputation: on_fedcomp,
                OnReport: on_report,

		// gossip msg handler
                OnRecvGossipMsg: on_recv_gossip_msg,
	}

	overlay = dual.New(dual.WithProtocolEvents(events),
	                   dual.WithProtocolMaxNeighborsBC(max_peers),)

	// Bind dual to the node.
	node.Bind(overlay.Protocol())

	//overlay.PubSub().RegisterMessage(UserInfo{}, UnmarshalUserInfo)

        //fmt.Printf("start listen\n")
	// Have the node start listening for new peers.
	check(node.Listen())

	// Print out the nodes ID and a help message comprised of commands.
	//help(node)

        go startPeriodicDiscover(serveraddr)

        fmt.Printf("init done\n")
}

//export Bootstrapping
func Bootstrapping(serveraddr *C.char) {
        bootstrapping(C.GoString(serveraddr))
}

func bootstrapping(serveraddr string) {
        //fmt.Printf("start bootstrap %s\n",serveraddr)
	// Ping nodes to initially bootstrap and discover peers from.
	bootstrap(serveraddr)

        //fmt.Printf("start discover\n")
	// Attempt to discover peers if we are bootstrapped to any nodes.
	discover(overlay, true)

	//if len(overlay.Table_bc().Peers()) > 1 {
	if len(overlay.Table_bc().KEntries(max_peers)) > 1 {
            overlay.Table_bc().DeleteByAddress(serveraddr)
	}
	//overlay.KPeers_bc(max_peers) // show backbone overlay routing table

        overlay.Table_bc().DeleteByAddress(node.ID().Address)
}

// bootstrap pings and dials an array of network addresses which we may interact with and  discover peers from.
func bootstrap(addr string) {
                fmt.Printf("run bootstrap on %s\n", addr)
                ctx, cancel := context.WithCancel(context.Background())
		_, err := node.Ping(ctx, addr, dual.OP_NEW_CONN)
		cancel()

		if err != nil {
			fmt.Printf("Failed to ping bootstrap node (%s). Skipping... [error: %s]\n", addr, err)
		}
}

// discover uses Kademlia to discover new peers from nodes we already are aware of.
func discover(overlay *dual.Protocol, isBootstrap bool) {
	ids := overlay.DiscoverRandom(isBootstrap)

	var str []string
	for _, id := range ids {
		str = append(str, fmt.Sprintf("%s(%s)", id.Address, id.ID.String()[:printedLength]))
	}

	overlay.KPeers_bc(max_peers) // show backbone overlay routing table

	/*
	if len(ids) > 0 {
		fmt.Printf("Discovered %d peer(s): [%v]\n", len(ids), strings.Join(str, ", "))
	} else {
		fmt.Printf("Did not discover any peers.\n")
	}
	*/
}

func startPeriodicDiscover(serveraddr string) {
        var len_peers int

	for {
	    time.Sleep(time.Duration(discoverInterval) * time.Second)
            len_peers = len(overlay.Table_bc().KEntries(max_peers))
	    if len_peers < max_peers {
	        ids := overlay.DiscoverLocal(false)
	        var str []string
	        for _, id := range ids {
		    str = append(str, fmt.Sprintf("%s(%s)", id.Address, id.ID.String()[:printedLength]))
	        }

		//overlay.KPeers_bc(max_peers) // show backbone overlay routing table

		/*
	        if len(ids) > 0 {
		    fmt.Printf("Discovered %d peer(s): [%v]\n", len(ids), strings.Join(str, ", "))
	        } else {
		    fmt.Printf("Did not discover any peers.\n")
	        }
		*/
	    }else{
		if len(overlay.Table_bc().KEntries(max_peers)) > 1 {
                    overlay.Table_bc().DeleteByAddress(serveraddr)

		    //overlay.KPeers_bc(max_peers) // show backbone overlay routing table
	        }
	    }
        }
}

//export Input
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
		command(line)
	}
}

func command(line string) {
	switch line {
	// common
	case "/discover":
	        fmt.Printf("\n")
		discover(overlay, false)
		return
	case "/peers_bc":
	        fmt.Printf("\n")
		overlay.KPeers_bc(max_peers) // show backbone overlay routing table
		return
	case "/tot_peers":
	        fmt.Printf("\n")
		fmt.Printf("tot_peers : %d\n",overlay.Table_bc().NumEntries())
		return
	case "/peers_gr": // show members of his group. Note it doesn't show leader
	        fmt.Printf("\n")
		overlay.Peers_gr()
		return
	//publisher command
        case "/newpjt": //create new project
	        fmt.Printf("\n")
	        pjtid, pjt := overlay.PubSub().NewProject(node.ID())
		my_pjt_id = pjtid
		fmt.Printf("project[%s] created. subscribers : %d\n",my_pjt_id[:printedLength], pjt.Size())
                return
        case "/subs": // # of subscribers
	        fmt.Printf("\n")
		pjt, found := overlay.PubSub().GetProject(my_pjt_id)
	        if found {
		    fmt.Printf("project[%s] subscribers : %d\n",pjt.GetPjtID()[:printedLength], pjt.Size())
		}
                for k, v := range pjt.GetSubList() {
		    fmt.Printf("%s(%s) \n",k[:printedLength], v.Addr)
		}
                return
        case "/sendAD": // gossip broadcast AD
	        fmt.Printf("\n")
                sendAD()
                return
        case "/group": //forming group
	        fmt.Printf("\n")
                RequestGroup()
                return
        case "/starttrain": // start the training
	        fmt.Printf("\n")
                globalModel()
                return
	case "/pjtstate": // show project state
                fmt.Printf("\n")
		pjt, found := overlay.PubSub().GetProject(my_pjt_id)
	        if found {
		    state := pjt.GetPjtState()
		    if state == dual.STATE_PUBLISHED {
		        fmt.Printf("project[%s] state : PUBLISHED\n",pjt.GetPjtID()[:printedLength])
		    }
		    if state == dual.STATE_ACTIVE {
		        fmt.Printf("project[%s] state : ACTIVE\n",pjt.GetPjtID()[:printedLength])
		    }
		    if state == dual.STATE_COMMITTED {
		        fmt.Printf("project[%s] state : COMMITTED\n",pjt.GetPjtID()[:printedLength])
		    }
		}
	default:
	}

	if strings.HasPrefix(line, "/") {
		help(node)
		return
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

func globalModel(){
        ptr := unsafe.Pointer(nil)
        // request global model to application first
        // this callback calls SendGlobalModel() with global model
        C.call_c_func(callbacks.on_reqglobalmodel, (*C.char)(ptr) )
        // runtime measurement for one round training
        starttime = time.Now()
}

//export SendGlobalModel
func SendGlobalModel(src *C.char, size C.int) {
        // data from application
        userdata := C.GoBytes(unsafe.Pointer(src), C.int(size))

        // send global model to initiator
        usermsg := userMessage{Opcode: OP_GLOBAL_MODEL, Contents: userdata}
        overlay.Fedcomp_GR(node.ID(), usermsg.Marshal())
}

// group forming
func RequestGroup() {
        // delete existing(e.g., previous) routing information in table_gr
        peers := overlay.Table_gr().KEntries(group_size)
        for _, id := range peers {
            fmt.Printf("delete existing group members : %s\n",id.ID.String()[:8])
            overlay.RemoveConn(overlay.Table_gr(), id, false)
        }

        pjt, found := overlay.PubSub().GetProject(my_pjt_id)
        if ! found {
            fmt.Printf("project does not exist\n")
	    return
	}
        fmt.Printf("project[%s] subscribers : %d\n",my_pjt_id, pjt.Size())
	//fmt.Print(pjt.GetSubList())
        data, _ := json.Marshal(pjt.GetSubList()) // we send map to initiator.

        // change project state to active
	pjt.SetPjtState(dual.STATE_ACTIVE)

        // sort Subscribers by metric and return array
        arr := sort_by_metric(pjt.GetSubList())
	i := 0
        for _, tx := range arr {
	    if i > 50 {
		break
	    }
            fmt.Printf("sublist[%s] traintime:%d numleave:%d\n",tx.Key[:8], tx.UserInfo.(UserInfo).TrainTime, tx.UserInfo.(UserInfo).NumLeave)
	    i = i + 1
        }

        // for now, pick the first node from sub list for initiator
        initiator = arr[0].Addr

	// calculate optimal group size
	//size := overlay.GridSearchGroupSize(pjt.Size()) //TODO: for now grid search returns 3
	size := group_size
	overlay.SetMaxNeighborsGR(size)

        ctx, cancel := context.WithCancel(context.Background())
        starttime = time.Now()
        // send sub_list and global model to initiator
        err := node.SendMessage(ctx, string(initiator), dual.P2pMessage{Aggregation: uint32(size), Opcode: byte(dual.OP_REQUEST_GROUP), Contents: data }, dual.OP_GROUP_MULTICAST)
        fmt.Printf("Send message to %s opcode: OP_REQUEST_GROUP\n",
                   initiator,
        )
        cancel()
        if err != nil {
            fmt.Printf("Failed to send message to %s(%s). Skipping... [error: %s]\n",
                       initiator, err,)
	}
}

func on_query_group_info() []byte {

    fmt.Printf("on_query_group_info: mytraintime %d mynumleave %d\n",mytraintime, mynumleave)

    userdata := UserInfo{TrainTime: uint32(mytraintime), NumLeave: uint32(mynumleave), TrustValue: uint32(mytrustvalue), UserID: node.ID()}.Marshal()

    return userdata

}

func on_aggregate_group_info(pool [][]byte) { // only Publisher get this callback

    //userinfoArr := make([]UserInfo, 0)

    pjt, found := overlay.PubSub().GetProject(my_pjt_id)
    if ! found {
        fmt.Printf("project does not exist\n")
        return
    }

    for _, s := range pool {
	userinfo, err := UnmarshalUserInfo(s)
	if err != nil {
            fmt.Printf("on_aggregate_group_info: err unmarshal\n")
	}
        //fmt.Printf("userinfo: %d %d\n",userinfo.TrainTime, userinfo.NumLeave)
	//userinfoArr = append(userinfoArr,userinfo)

	//update the sublist with the information from the group
        updateSubList(pjt.GetSubList(), userinfo)
    }

    fmt.Printf("on_aggregate_group_info: return aggregated info\n")
}

func updateSubList(sublist map[string]dual.Subscriber, userinfo UserInfo) {
    //fmt.Printf("userinfo[%s]: %d %d\n",userinfo.UserID.ID.String()[:8], userinfo.TrainTime, userinfo.NumLeave)
    sub, found := sublist[userinfo.UserID.ID.String()]
    if found {
	sub.UserInfo = UserInfo{TrainTime:userinfo.TrainTime, NumLeave: userinfo.NumLeave, UserID:userinfo.UserID}
	sublist[userinfo.UserID.ID.String()] = sub
        //fmt.Printf("sublist[%s] updated traintime:%d numleave:%d\n",sub.UserInfo.(UserInfo).UserID.ID.String()[:8],sub.UserInfo.(UserInfo).TrainTime, sub.UserInfo.(UserInfo).NumLeave)
    }
}

func main() {
        runtime.GOMAXPROCS(runtime.NumCPU())
        args := os.Args[1:]
        port, err := strconv.Atoi(args[1])
	if err != nil {
	    // Add code here to handle the error!
	}

        if len(args) >= 3 {
            init_p2p(string(args[0]), port, string(args[2]))
            bootstrapping(string(args[2]))

	}else{
            init_p2p(string(args[0]), port, "")
	}

        // block here
	// If you use run.sh script to launch many peer nodes, you should comment below line
        //Input() // simulation many bench on the same physical node causes error on stdin.
	// Instead, use this to block the process.
	for{}
}

