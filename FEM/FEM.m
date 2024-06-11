mesh = Mesh2D('mesh_0064.msh');

feMap = FEMap(mesh);

A = assembleDiffusion(mesh, feMap);