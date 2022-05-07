# Introduction

This repository contains implementation of G-RING Application: Federated Learning on G-RING for Disease Classification with MedMNIST.

# Installation

1. Install Go
This repository is valided with, but not limited to, go1.16
```bash
wget https://golang.org/dl/go1.16.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.16.linux-amd64.tar.gz
```


# Running application

## Run bootstrap node

```bash
go run bootstrap.go 100.10.10.1 9999
```

## Run Publisher node

publisher.sh takes three commandline arguments: publisher IP, port and bootstrap address
```bash
python3 publisher.sh 100.10.10.2 8000 100.10.10.1:9999
```
## Run Peer nodes

peer.sh takes three commandline arguments: publisher IP, port and bootstrap address
```bash
python3 peer.sh 100.10.10.3 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.4 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.5 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.6 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.7 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.8 8000 100.10.10.1:9999
python3 peer.sh 100.10.10.9 8000 100.10.10.1:9999
```

## Run commands on Publisher node

Publisher creates a project for FL task
```bash
/newpjt
```

Publisher sends AD messages to recruit members
```bash
/sendAD
```

After subscribers are joined(as many as you want),
Publisher forms a group on application overlay
```bash
/group
```

Start FL training rounds
```bash
/starttrain
```


# Commandline commands

## Common
```bash
/peers_bc : print all peers in the backbone routing table
/peers_gr : print all peers in the application routing table
/discover : discover peers on backbone overlay
```
## Publisher
```bash
/newpjt : create a new project
/pjtstate : show project status : PUBLISHED, ACTIVE, COMMITED.
/subs : print the number of subscribers(i.e. project members) of the project
/sendAD : broadcast advertisement messages to backbone overlay to recruit project members
/group : form a group on appliction overlay for the project
```

## Peers
```bash
all common commands
```

# Setting parameters

## On Python-Go plug-in (peer.go)
This plug-in file is provided by G-RING to support seamless data transfer and control between Python(e.g. ML application) and Go(e.g., underlying network).

"group size" is to decide the degree of the tree on application overlay
```bash
// group\_size-1 = degree of tree = # of children
// in this toy example, we create a binary tree on application overlay
group\_size=3
```

## On Python Application (publisher.py peer.py)
publisher.py and peer.py are application codes for FL project implemented on G-RING

```bash
// total number of FL training rounds
NUM_GLOBAL_ROUNDS = 3

# number of local training at each peer node
NUM_LOCAL_EPOCHS = 1 
```

```bash
//training variables
lr = 0.001
```

# Supported Platforms

Tested environment:

OS : Ubuntu 18.04
Hardware : Ethernet

# Dataset requirement
This FL application on G-RING will download MedMNIST dataset.
No manual dataset download is required.

We use PathMNIST dataset. For more details about dataset, please refer to original source.
https://medmnist.com

# Status
The code is provided as is, without warranty or support.
