# Quick Start

### Set up a new virtual environment
```bash
conda create -n guide python=3.8 -y
conda activate guide
```

### Install dependency packpages
```bash
guide_path="path/to/guide"
cd ${guide_path}
pip3 install --upgrade pip
pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip3 install -r requirement.txt
bash tools/install.sh
```

### Prepare the data
Download the [NuScenes dataset](https://www.nuscenes.org/nuscenes#download), put it in /path/to/nuscenes, create symbolic links.
```bash
cd ${guide_path}
mkdir data
ln -s path/to/nuscenes ./data/nuscenes
```

Pack the meta-information and labels of the dataset, and generate the required pkl files to data/infos. If you want a different range, you can modify roi_size in tools/data_converter/nuscenes_converter.py.
```bash
sh scripts/create_data.sh
```

### Generate anchors by K-means
Generated anchors are saved to data/kmeans and can be visualized in vis/kmeans.
```bash
sh scripts/kmeans.sh
```

### Generate instance GT occupancy in occ3d
```bash
sh scripts/gen_occ_gt.sh
```

### Download pre-trained weights
Download the required backbone [pre-trained weights](https://download.pytorch.org/models/resnet50-19c8e357.pth).
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

### Download the required weights
Download our trained weights [guide_gs32.pth](https://github.com/CN-ADLab/GUIDE/releases/download/v1.0/guide_gs32.pth), put it in ./ckpt.


### Commence testing
```bash

# test
sh scripts/test.sh
```
