## Introduction
**[NURBS-Diff](https://www.sciencedirect.com/science/article/abs/pii/S0010448522000045)** is a nerual network differentiable programming module for NURBS, and **[LNLib](https://github.com/BIMCoderLiang/LNLib)** is a C++ NURBS Algorithms Library on Github. **NURBS-Diff-with-LNLib** is a simplified reproduction for NURBS-Diff by using LNLib.

## Run NURBS-Diff-with-LNLib
- Download Libtorch from https://pytorch.org/get-started/locally/. (For example: local-download folder is C:/Code/CodeReference/)
- Reset Libtorch Path in CMakeLists.txt (from **src** folder) up to your Libtorch download path.
- Run build.bat to construct C++ solution by CMake.

## Visualization
We use [VTK](https://vtk.org/) to visualize NURBS fitting curve and surface.

<img src="assets/nurbs curve fitting.jpg" width=800 height=400>
<img src="assets/nurbs surface fitting.png" width=800 height=400>

## Contributing
Welcome join this project including discussions in **Issues** and make **Pull requests**.

## Author

- **NURBS-Diff** is work done at Integrated Design and Engineering Analysis Lab, Iowa State University under Prof. Adarsh Krishnamurthy. Collaborators : Aditya Balu (baditya@iastate.edu), Harshil Shah (harshil@iastate.edu)
</br>

- **LNLib** & **NURBS-Diff-with-LNLib** are created by Yuqing Liang (bim.frankliang@foxmail.com),</br> 微信公众号：**BIMCoder**

## License
The source code is published under [GNU General Public License v3.0](https://www.gnu.org/licenses/), the license is available [here](LICENSE).

## Primary Reference
- [NURBS-Diff Article](https://www.sciencedirect.com/science/article/abs/pii/S0010448522000045)
- [NURBS-Diff open source codes on Github (BSD-3-Clause)](https://github.com/anjanadev96/NURBS_Diff)
- [LNLib on Github (LGPL-2.1)](https://github.com/BIMCoderLiang/LNLib)