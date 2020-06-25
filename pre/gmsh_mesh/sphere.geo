// Gmsh project created on Mon Jun 10 08:51:57 2019
SetFactory("OpenCASCADE");
//+
Sphere(1) = {0, 0, 0, 1, -Pi/2, Pi/2, 2*Pi};
//+
Box(2) = {0, 0, 0, 1, 1, 1};
//+
BooleanIntersection{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Sphere(2) = {0, 0, 0, 0.2, -Pi/2, Pi/2, 2*Pi};
//+
BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Physical Surface(1) = {5};
//+
Physical Surface(2) = {1};
//+
Physical Volume(3) = {1};
//+
Characteristic Length {6, 5, 4, 2, 3, 1} = 0.01;
