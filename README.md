# Pre-requisite:
## Download ubuntu-16.04 server version at following official website
```
http://releases.ubuntu.com/16.04.5/
ubuntu-16.04.5-server-amd64.iso
```
## Installation for python 2.7 in Ubuntu 16.04 or MacOS (If using vbox only need 1G RAM and 1 core CPU)
```
sudo apt-get install -y git
```
## Auto start recommendation engine
```
sh env-installation.sh
```
## Manually Start recommendation engine
```
cd chooosie-deals-intelligent-system
source lambda_prototype/bin/activate
python ms-mlengine-local.py
```
## Use WEB browser access port 9080
```
localhost:9080
127.0.0.1:9080
x.x.x.x : 9080
```
