#!/bin/bash

virtualenv lambda_prototype
source lambda_prototype/bin/activate
apt-get install -y build-essential libssl-dev libffi-dev python-dev python-tk tcpflow git curl vim
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python get-pip.py
pip install numpy pandas SciPy
 
pip install scikit-learn 
pip install matplotlib
 
pip install csvkit 
pip install pydotplus 
pip install graphviz
 
pip install Ipython
pip install numpy pandas SciPy
pip install scikit-learn 
pip install matplotlib
pip install csvkit 
pip install pydotplus 
pip install graphviz
pip install Ipython
pip install tensorrec
pip install flask flask-jsonpify flask-sqlalchemy flask-restful
