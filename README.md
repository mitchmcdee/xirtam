# Xirtam
Xirtam is a 2.5D robot motion planning simulator. It allows both the training and simulation of motion planning. During the training phase, foot placement and world region images are generated and exported for later use to train TimTamNet.

# TimTamNet
TimTamNet is a fully convolutional neural network designed to learn the mapping between foot placements and the hidden environment map. Significant effort was put into designing this network to be extremely small (2.7k learnable parameters) for efficient storage (52kB uncompressed) and inference (<1ms on laptop hardware).

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
To specify a specific robot file:
```bash
python3 -m xirtam -s -r example.robot
```
To specify a specific world/motion file:
```bash
python3 -m xirtam -s -w example.world -m example.motion
```
To utilise a trained TimTamNet model:
```bash
python3 -m xirtam -s -n model.hdf5
```
To train a TimTamNet model:
```bash
python3 xirtam/neural/train.py -r ./path/to/images/
```
To debug and view the results of a TimTamNet model:
```bash
jupyter notebook xirtam/neural/debug.ipynb
```

# Thesis
Motion Planning in Partially Observable Environments using Fully Convolutional Networks (TODO(mitch): add link to thesis pdf here)
