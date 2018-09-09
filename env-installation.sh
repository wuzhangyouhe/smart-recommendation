#!/bin/bash

virtualenv lambda_prototype
source lambda_prototype/bin/activate
sudo apt-get install -y build-essential libssl-dev libffi-dev python-dev python-tk tcpflow git tshark curl vim
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
sudo python get-pip.py
sudo pip install numpy pandas SciPy
 
sudo pip install scikit-learn 
sudo pip install matplotlib
 
sudo pip install csvkit 
sudo pip install pydotplus 
sudo pip install graphviz
 
sudo pip install Ipython
sudo pip install numpy pandas SciPy
sudo pip install scikit-learn 
sudo pip install matplotlib
sudo pip install csvkit 
sudo pip install pydotplus 
sudo pip install graphviz
sudo pip install Ipython
sudo pip install tensorrec
sudo pip install flask flask-jsonpify flask-sqlalchemy flask-restful
