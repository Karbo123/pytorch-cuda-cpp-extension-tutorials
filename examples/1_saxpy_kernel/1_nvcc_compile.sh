nvcc main.cu -o main.so -DTORCH_EXTENSION_NAME=main \
-std=c++14 --expt-relaxed-constexpr -shared -Xcompiler=-fPIC -O3 \
-D_GLIBCXX_USE_CXX11_ABI=`python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"` \
`python -c "from torch.utils import cpp_extension; print(' '.join([f'-I{s}' for s in cpp_extension.include_paths()]))"` \
`python -c "from distutils.sysconfig import get_python_inc; print('-I'+get_python_inc())"` \
`python -c "from torch.utils import cpp_extension; print('-L'+cpp_extension.TORCH_LIB_PATH)"` -ltorch_python
