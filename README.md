# Pre-requisite:

## Installation for python 2.7 in Ubuntu 16.04 or MacOS (If using vbox only need 1G RAM and 1 core CPU)
```
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev python-tk git curl vim
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
sudo python get-pip.py
sudo pip install virtualenv 

sh env-installation.sh

```
## Start recommendation engine
```
python ms-mlengine-local.py
```
## Use WEB browser access localhost:8080
```
localhost:8080
127.0.0.1:8080
```
