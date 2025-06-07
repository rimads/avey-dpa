conda install gxx_linux-64=12.4.0 -y
git clone https://github.com/state-spaces/mamba.git ./mamba-temp
cd mamba-temp
pip install .
cd ..
rm -rf ./mamba-temp
