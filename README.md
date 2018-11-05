# Xirtam
Xirtam is a 2.5D robot motion planning simulator. It allows both the training and simulation of motion planning. During the training phase, foot placement and world region images are generated and exported for later use to train TimTamNet.

# TimTamNet
TimTamNet is a fully convolutional neural network designed to learn the mapping between foot placements and the hidden environment map. Significant effort was put into designing this network to be extremely small (933 learnable parameters) for efficient storage (130kB uncompressed) and inference (<1ms on laptop hardware).

# Requirements
- Python3.6
- `pip3 install -r requirements.txt`

# Running
_Note: Additional settings can be found and manipulated in `xirtam/core/settings.py`!_

For usage help:
```bash
python3 -m xirtam --help
```
To run a visual simulation:
```bash
python3 -m xirtam -s
```
To run a training simulation:
```bash
python3 -m xirtam -t
```
To generate a random map:
```bash
python3 -m xirtam -s -g
```
To split generated data into training and test sets (necessary for training):
```bash
python3 xirtam/neural/split_util.py -r path/to/robot_dir/
```
To utilise a trained TimTamNet model:
```bash
python3 -m xirtam -s -n model.hdf5
```
To train a TimTamNet model:
```bash
python3 xirtam/neural/train.py -r path/to/robot_dir/
```
To debug and view the results of a TimTamNet model:
```bash
jupyter notebook xirtam/neural/debug.ipynb
```

# Thesis
[Efficient Mapping in Partially Observable Environments using Deep Learning](https://drive.google.com/open?id=1SKtxHuCVcvMXvnQkKLPDFwzsXWYMOy91)
