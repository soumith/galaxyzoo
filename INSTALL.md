Install Notes
=============
Install on Ubuntu or OSX
------------------------

* Install torch using
```
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-all | bash
```

* Then install the rest of the needed packages:
```
luarocks install cutorch
luarocks install cunn
luarocks install csvigo
luarocks install nnx
luarocks install torchffi
luarocks install env
luarocks install graphicsmagick
apt-get install libgraphicsmagick-dev
```

* Download the data from: http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
* Place the data files into a subfolder "data".
* Then run the script dataprep.sh
