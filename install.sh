rm -R build
rm cyfusion.cpp
rm cyfusion.cpython-37m-x86_64-linux-gnu.so

mkdir build
cd build

cmake ..
make

cd ..
python setup.py build_ext --inplace
sudo cp -R ~/projects/voxel_fusion /home/marko/.conda/envs/differentiable_rendering/lib/python3.7/site-packages/
