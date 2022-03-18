# Contents
This contains two examples of OCP solution with casadi Opti class.
As a general documentation reference for casadi, you can refer to [the official doc](https://web.casadi.org/docs/) and look at the many [examples](https://github.com/casadi/casadi/tree/master/docs/examples)

In particular:
- ```cartpole.py``` uses a custom made problem with analytic dynamics, so it's a standalone example
- ```doublependulm.py``` uses ```pinocchio.casadi``` module (requires ```pinocchio >= 3.0```) to discretize the integrator dynamics with the algorithms implemented in ```pinocchio```

Similarly as shown in the code the functions in ```pinocchio.casadi``` can be used to impose costs or constraints.


# Set up

This package relies on Pinocchio-3x (internal repo @Inria), which in turns rely on several packages.

## APT packages

Some packages rely on robotpkg APT repo, that can be set up following [Pinocchio installation procedure](https://stack-of-tasks.github.io/pinocchio/download.html). If following this procedure, don't forget to also set up your environment variables.

You need the following packages:
```
sudo apt install cmake  cmake-curses-gui  ccache  doxygen  libgmp10-dev  libmpfr-dev  robotpkg-py38-casadi libsdformat-dev graphviz libboost-timer-dev robotpkg-py38-pinocchio
```

## HPP-FCL devel branch

You need to recompile [HPP-FCL](https://github.com/humanoid-path-planner/hpp-fcl/). Clone the repo, then execute:
```
cd hpp-fcl
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$HOME/compil/pinocchio
make -sj8 install
```

Add the following to your environment:
```
export PINOCCHIO_INSTALL_DIR=$HOME/compil/pinocchio
export PATH=${PINOCCHIO_INSTALL_DIR}/bin:${PATH}
export PKG_CONFIG_PATH=${PINOCCHIO_INSTALL_DIR}/lib/pkgconfig:${PKG_CONFIG_PATH}
export PKG_CONFIG_PATH=${PINOCCHIO_INSTALL_DIR}/lib64/pkgconfig:${PKG_CONFIG_PATH}
export LD_LIBRARY_PATH=${PINOCCHIO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${PINOCCHIO_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH}
export PYTHON_LOCAL_PATH=lib/python3/dist-packages
export PYTHONPATH=${PINOCCHIO_INSTALL_DIR}/${PYTHON_LOCAL_PATH}:${PYTHONPATH}
```

Check by loading the hpp-fcl module in python, it should point to your compilation folder:
```
python -c "import hppfcl; print(hppfcl.__file__)"
```

## Compile Pinocchio-3x

Clone the Inria repo of Pinocchio
```
git clone --recursive git@gitlab.inria.fr:jucarpen/pinocchio.git -b topic/template-instantiation
```

Compile it with the following options:
```
cd pinocchio
mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDINGS_WITH_BOOST_MPFR_SUPPORT=OFF -DBUILD_WITH_CASADI_SUPPORT=ON -DBUILD_WITH_COLLISION_SUPPORT=ON -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=$PINOCCHIO_INSTALL_DIR
make -sj8 install
```

### Viewer

You might consider to install Gepetto viewer 
```
sudo apt install robotpkg-py38-qt5-hpp-gui
```

Don't forget to start the viewer before executing the scripts needing it (otherwise you will typically get the following error "AttributeError: 'GepettoVisualizer' object has no attribute 'viewer'"):
```
gepetto-viewer
```

Alternatively, you will have to comment out the display command in the example scripts.



## Git commits

The installation procedure as been check with HPP-FCL commit 39fde9cb and Pinocchio commit 74deb9ab.

## Check

You can run
```python -c "import pinocchio.casadi as cpin; print(cpin.__file__)"```


