source ~/anaconda3/etc/profile.d/conda.sh
conda activate differentiable_rendering

rm -R build
rm cyfusion.cpp
rm cyfusion.cpython-37m-x86_64-linux-gnu.so
rm cyfusion.cpython-35m-x86_64-linux-gnu.so

mkdir build
cd build

cmake ..
make

cd ..
python setup.py build_ext --inplace
#cp -R ~/projects/voxel_fusion /home/marko/.conda/envs/differentiable_rendering/lib/python3.5/site-packages
#cp -R ~/voxel_fusion /cluster/home/markomih/.virtualenvs/py36/lib/python3.6/site-packages
cp -R ~/voxel_fusion /cluster/home/markomih/.local/lib64/python3.6/site-packages
