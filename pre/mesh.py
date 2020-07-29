#--------------------------------------------------------------
# Mesh generation module of a radially symmetric Stefan problem
#--------------------------------------------------------------
#--------------------------
# Additional modules import
#--------------------------
import dolfin
import mshr
import os
#===========================

#-------------------------
# Mesh generation routines
#-------------------------
# 1d:
def mesh1d(left,right,meshres):
    mesh = dolfin.IntervalMesh(meshres,left,right)
    # Construct of the facet markers:
    boundary = (dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0),{})
    #boundary[0].set_all(0)
    for f in dolfin.facets(mesh):
        if dolfin.near(f.midpoint()[0], left):
            boundary[0][f] = 1 # left
            boundary[1]['inner'] = 1
        elif dolfin.near(f.midpoint()[0], right):
            boundary[0][f] = 2 # right
            boundary[1]['outer'] = 2
    # Definition of measures and normal vector:
    n = dolfin.FacetNormal(mesh)
    dx = dolfin.Measure("dx", mesh)
    ds = dolfin.Measure("ds", subdomain_data = boundary[0])
    return (mesh,boundary,n,dx,ds)
#-------------------------
# 2d:
def mesh2d(inner,outer,*meshres,stefan=True):
    origin = dolfin.Point(0.,0.)
    if stefan:
        geometry=mshr.Circle(origin, outer, 2*meshres[0]) - mshr.Circle(origin, inner, int(0.5*meshres[0]))
        mesh=mshr.generate_mesh(geometry, meshres[0])
        mesh.init()
        # Construct of the facet markers:
        boundary = (dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0),{})
        for f in dolfin.facets(mesh):
            if f.midpoint().distance(origin) <= inner and f.exterior():
                boundary[0][f] = 1     # inner radius
                boundary[1]['inner'] = 1
            elif f.midpoint().distance(origin) >= (inner+outer)/2  and f.exterior():
                boundary[0][f] = 2     # outer radius
                boundary[1]['outer'] = 2
        # Definition of measures and normal vector:
        n = dolfin.FacetNormal(mesh)
        dx = dolfin.Measure("dx", mesh)
        ds = dolfin.Measure("ds", domain=mesh, subdomain_data = boundary[0])
    else:
        width = inner
        height = outer
        mesh = dolfin.RectangleMesh(origin,Point(width,height),meshres[0],meshres[1])
        mesh.init()
        boundary = (dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0),{})
        for f in dolfin.facets(mesh):
            if dolfin.near(f.midpoint()[1],0.):
                boundary[0][f] = 1 # bottom
                boundary[1]['bottom'] = 1
            elif dolfin.near(f.midpoint()[1],height):
                boundary[0][f] = 2 # top
                boundary[1]['top'] = 2
            elif dolfin.near(f.midpoint()[0],0.):
                boundary[0][f] = 3 # left
                boundary[1]['left'] = 3
            elif dolfin.near(f.midpoint()[0],width):
                boundary[0][f] = 4 # right
                boundary[1]['right'] = 4
        # Definition of measures and normal vector:
        n = dolfin.FacetNormal(mesh)
        dx = dolfin.Measure("dx", mesh)
        ds = dolfin.Measure("ds", subdomain_data = boundary[0])
    return (mesh,boundary,n,dx,ds)
#-------------------------
# 3d:
def mesh3d(width,depth,height,nx,ny,nz):
    mesh = dolfin.BoxMesh(Point(0.,0.,0.),Point(width,depth,height),nx,ny,nz)
    boundary = (dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0),{})
    for f in dolfin.facets(mesh):
        if dolfin.near(f.midpoint()[2],0.):
            boundary[0][f] = 1 # bottom
            boundary[1]['bottom'] = 1
        elif dolfin.near(f.midpoint()[2],height):
            boundary[0][f] = 2 # top
            boundary[1]['top'] = 2
        elif dolfin.near(f.midpoint()[0],0.):
            boundary[0][f] = 3 # left
            boundary[1]['left'] = 3
        elif dolfin.near(f.midpoint()[0],width):
            boundary[0][f] = 4 # right
            boundary[1]['right'] = 4
        elif dolfin.near(f.midpoint()[1],0):
            boundary[0][f] = 5 # front
            boundary[1]['front'] = 5
        elif dolfin.near(f.midpoint()[1],depth):
            boundary[0][f] = 6 # back
            boundary[1]['back'] = 6
    # Definition of measures and normal vector:
    n = dolfin.FacetNormal(mesh)
    dx = dolfin.Measure("dx", mesh)
    ds = dolfin.Measure("ds", subdomain_data = boundary[0])
    mesh_xdmf = dolfin.XDMFFile(mpi_comm_world(),"data/mesh_3D.xdmf")
    mesh_xdmf.write(boundaries[0])
    return (mesh,boundary,n,dx,ds)
#-------------------------
# mesh from gmesh file:
def meshxml(name): # reads mesh from .xml file, cf. Notes
    # IN: str of name of the .xml file (do that in serial):
    mesh = dolfin.Mesh(name+'.xml')
    mesh.init()
    hdf = dolfin.HDF5File(mesh.mpi_comm(),name+"_hdf.h5", "w")
    hdf.write(mesh, "/mesh")
    xdmf = dolfin.XDMFFile(mesh.mpi_comm(),name+"_xdmf.xdmf")
    xdmf.write(mesh)
    xdmf.close()
    domains=mesh.domains()
    if os.path.isfile(name+"_physical_region.xml"):
        domains.init(mesh.topology().dim() - 1)
    if os.path.isfile(name+"_facet_region.xml"):
        boundary = (dolfin.MeshFunction("size_t",mesh,name+"_facet_region.xml"),{'inner':1,'outer':2})
        #mesh.init()
        hdf.write(boundary[0], "/boundary")
        
        ds = dolfin.Measure("ds", subdomain_data = boundary[0])
    else:
        boundary = (dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1,0),{})
        width = mesh.coordinates()[:,0].max()
        height = mesh.coordinates()[:,1].max()
        for f in dolfin.facets(mesh):
            if dolfin.near(f.midpoint()[1],0.):
                boundary[0][f] = 1 # bottom
                boundary[1]['bottom'] = 1
            elif dolfin.near(f.midpoint()[1],height):
                boundary[0][f] = 2 # top
                boundary[1]['top'] = 2
            elif dolfin.near(f.midpoint()[0],0.):
                boundary[0][f] = 3 # left
                boundary[1]['left'] = 3
            elif dolfin.near(f.midpoint()[0],width):
                boundary[0][f] = 4 # right
                boundary[1]['right'] = 4
        ds = dolfin.Measure("ds", subdomain_data = boundary[0])
    # mesh = Mesh()
    # hdf.read(mesh,"/mesh", False)
    hdf.close()
    # Definition of measures and normal vector:
    n = dolfin.FacetNormal(mesh)
    dx = dolfin.Measure("dx", mesh)
    return (mesh,boundary,n,dx,ds)
#-------------------------
# mesh from hdf file:
def meshhdf(name):
    
    mesh = dolfin.Mesh()
    
    hdf = dolfin.HDF5File(mesh.mpi_comm(),name+"_hdf.h5","r")
    hdf.read(mesh,"/mesh",False)
    
    boundary = (dolfin.MeshFunction('size_t',mesh,mesh.topology().dim()-1,0),{'inner':1,'outer':2})
    hdf.read(boundary[0],"/boundary")
    hdf.close()
    
    n = dolfin.FacetNormal(mesh)
    dx = dolfin.Measure("dx", domain=mesh)
    ds = dolfin.Measure("ds", domain=mesh, subdomain_data = boundary[0])
    
    return (mesh,boundary,n,dx,ds)
#=========================

#------------------------
# Notes and obsolete code
#------------------------
# when generating 3D mesh for Stefan benchmark, first run ./generate_mesh.sh that will produce mesh in .msh file format from the .geo file and immediately convert it to .xml using dolfin-convert (deprecated in the newer versions of FEniCS), then mesh3d opens the .xml file and stores the mesh in hdf5 file format, that contains boundary markings from gmsh physical groups
# initialize the mesh with 0, otherwise errors in parallel (cluster) may occur
