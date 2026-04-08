source /opt/conda/bin/activate guide
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../
pip3 install spconv-cu116
pip3 install open3d
pip3 install localagg_prob_bin/