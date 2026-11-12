import gmsh
import meshio
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import tqdm.autonotebook

from mpi4py import MPI
from petsc4py import PETSc

from basix.ufl import element

from dolfinx.fem import (
    Constant,
    Function,
    functionspace,
    assemble_scalar,
    dirichletbc,
    extract_function_spaces,
    form,
    locate_dofs_topological,
    set_bc,
)
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_vector,
    create_matrix,
    set_bc,
)
from dolfinx.geometry import bb_tree, compute_collisions_points, compute_colliding_cells
from dolfinx.io import XDMFFile, VTXWriter, gmsh as gmshio
from dolfinx.mesh import create_mesh
from ufl import (
    FacetNormal,
    Measure,
    Mesh,
    TestFunction,
    TrialFunction,
    as_vector,
    div,
    dot,
    dx,
    inner,
    lhs,
    grad,
    nabla_grad,
    rhs,
)

import pyvista as pv 
from dolfinx.plot import vtk_mesh
from pathlib import Path
from pyproj import Transformer
import json
import logging

logger = logging.getLogger(__name__)


FLUID_MARKER = 1
INLET_MARKER = 2
OUTLET_MARKER = 3
WALL_MARKER = 4
OBSTACLE_MARKER = 5
DOMAIN_MARKER = 10

def plot_mesh_pyvista(mesh, output_path="figures/mesh_preview.png"):
    if mesh.comm.rank == 0:
        os.makedirs("figures", exist_ok=True)
        cells, cell_types, x = vtk_mesh(mesh, mesh.topology.dim)
        grid = pv.UnstructuredGrid(cells, cell_types, x)
        
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black", line_width=0.4)
        plotter.view_xy()
        plotter.screenshot(output_path)
        plotter.close()
        print(f"✅ Mesh visualization generated at: {output_path}")
    mesh.comm.Barrier()  # Ensure all processes wait for the visualization to complete


def plot_mesh_matplotlib(mesh, output_path="figures/mesh_preview.png"):
    if mesh.comm.rank == 0:
        os.makedirs("figures", exist_ok=True)
        
        # Build 2D cell-to-vertex connectivity
        mesh.topology.create_connectivity(2, 0)
        conn = mesh.topology.connectivity(2, 0)
        
        # Extract 2D node coordinates (x, y)
        coords = mesh.geometry.x[:, :2]
        
        # Calculate nodes per cell dynamically (4 for quads, 3 for triangles)
        num_cells = mesh.topology.index_map(2).size_local
        nodes_per_cell = len(conn.array) // num_cells  # Evaluates to 4 for your mesh
        
        # Reshape connectivity array
        cells = conn.array.reshape(num_cells, nodes_per_cell)
        
        # Extract polygon coordinates for each cell
        polygons = coords[cells]
        
        # Plot unstructured mesh grid
        fig, ax = plt.subplots(figsize=(10, 4))
        coll = PolyCollection(polygons, facecolors='none', edgecolors='royalblue', linewidths=0.4)
        ax.add_collection(coll)
        
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title("Gmsh Recombined Mesh Preview (DFG 2D-3 Benchmark)", fontsize=12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close(fig)
        print(f"✅ Mesh preview successfully saved to {output_path}")


# ---------------------------------------------------------
# MESH GENERATION FUNCTION
# ---------------------------------------------------------
        
def generate_dfg2d_mesh(L=2.2, H=0.41, c_x=0.2, c_y=0.2, r=0.05, preview_mesh="pyvista"):

    gmsh.initialize()
    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0

    if mesh_comm.rank == model_rank:
        rectangle = gmsh.model.occ.addRectangle(0, 0, 0, L, H, tag=1)
        obstacle = gmsh.model.occ.addDisk(c_x, c_y, 0, r, r)

        fluid = gmsh.model.occ.cut([(gdim, rectangle)], [(gdim, obstacle)])
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.getEntities(dim=gdim)
        assert len(volumes) == 1
        gmsh.model.addPhysicalGroup(volumes[0][0], [volumes[0][1]], FLUID_MARKER)
        gmsh.model.setPhysicalName(volumes[0][0], FLUID_MARKER, "Fluid")

        inflow, outflow, walls, obstacle = [], [], [], []
        boundaries = gmsh.model.getBoundary(volumes, oriented=False)
        for boundary in boundaries:
            center_of_mass = gmsh.model.occ.getCenterOfMass(boundary[0], boundary[1])
            if np.allclose(center_of_mass, [0, H / 2, 0]):
                inflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L, H / 2, 0]):
                outflow.append(boundary[1])
            elif np.allclose(center_of_mass, [L / 2, H, 0]) or np.allclose(center_of_mass, [L / 2, 0, 0]):
                walls.append(boundary[1])
            else:
                obstacle.append(boundary[1])
        gmsh.model.addPhysicalGroup(1, walls, WALL_MARKER)
        gmsh.model.setPhysicalName(1, WALL_MARKER, "Walls")
        gmsh.model.addPhysicalGroup(1, inflow, INLET_MARKER)
        gmsh.model.setPhysicalName(1, INLET_MARKER, "Inlet")
        gmsh.model.addPhysicalGroup(1, outflow, OUTLET_MARKER)
        gmsh.model.setPhysicalName(1, OUTLET_MARKER, "Outlet")
        gmsh.model.addPhysicalGroup(1, obstacle, OBSTACLE_MARKER)
        gmsh.model.setPhysicalName(1, OBSTACLE_MARKER, "Obstacle")


        res_min = r / 3
        distance_field = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(distance_field, "EdgesList", obstacle)
        threshold_field = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(threshold_field, "IField", distance_field)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMin", res_min)
        gmsh.model.mesh.field.setNumber(threshold_field, "LcMax", 0.25 * H)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMin", r)
        gmsh.model.mesh.field.setNumber(threshold_field, "DistMax", 2 * H)
        min_field = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [threshold_field])
        gmsh.model.mesh.field.setAsBackgroundMesh(min_field)


        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")
        preview_mesh_data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=gdim)
        if preview_mesh == "matplotlib":
            plot_mesh_matplotlib(preview_mesh_data.mesh, "figures/global_mesh_preview.png")
        else:
            plot_mesh_pyvista(preview_mesh_data.mesh, "figures/global_mesh_preview.png")

    mesh_data = gmshio.model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
    mesh = mesh_data.mesh
    assert mesh_data.facet_tags is not None
    ft = mesh_data.facet_tags
    ft.name = "Facet markers"
    return mesh, ft


'''
    gmsh_to_xdmf(gmsh.model.mesh)
'''
def gmsh_to_xdmf(mesh):
    node_tags, node_coords, _ = mesh.getNodes()
    # Map arbitrary gmsh node tags to contiguous 0-indexed positions
    node_tag_to_index = {tag: i for i, tag in enumerate(node_tags)}
    points = node_coords.reshape(-1, 3)[:, :2]  # (x, y) coordinates
    elem_types, elem_tags, elem_node_tags = mesh.getElements(dim=2)
    tri_cells = []
    print(elem_types)
    for etype, _, enodes in zip(elem_types, elem_tags, elem_node_tags):
        props = mesh.getElementProperties(etype)
        num_nodes = props[3]
        # Reshape the flat node array based on actual nodes per element
        element_nodes = enodes.reshape(-1, num_nodes)
        for elem in element_nodes:
            if num_nodes == 3:
                # Standard 3-node triangle
                tri_cells.append([node_tag_to_index[t] for t in elem[:3]])
            elif num_nodes > 3:
                # If higher-order elements slipped through, take the first 3 primary corner nodes
                tri_cells.append([node_tag_to_index[t] for t in elem[:3]])
            elif num_nodes == 4:
                # If any quads (type 3) were generated, split them into 2 triangles for FEniCSx
                p0, p1, p2, p3 = [node_tag_to_index[t] for t in elem[:4]]
                tri_cells.append([p0, p1, p2])
                tri_cells.append([p0, p2, p3])

        #if etype == 3:  # 3-node triangle
        #    tri_nodes = enodes.reshape(-1, 3)
        #    for tri in tri_nodes:
        #        tri_cells.append([node_tag_to_index[t] for t in tri])
    print(f"✅ Mesh generation completed for process '{proc_id}' with {len(points)} vertices and {len(tri_cells)} triangular cells.")

    cells = np.array(tri_cells, dtype=np.int64)
    mesh_data_payload = {
        "points": points,
        "cells": cells
    }
    mesh_comm.Barrier()
    mesh_data_payload = mesh_comm.bcast(mesh_data_payload, root=model_rank)
    points = mesh_data_payload["points"]
    points = np.ascontiguousarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        points = points.reshape(-1, 2)

        
    cells = mesh_data_payload["cells"]
    # Ensure cells are strictly int64 with shape (M, 3)
    cells = np.ascontiguousarray(cells, dtype=np.int64)
    if cells.ndim != 2 or cells.shape[1] != 3:
        cells = cells.reshape(-1, 3)

    # --- Step 3: Build FEniCSx Mesh Natively ---
    domain = element("Lagrange", "triangle", 1, shape=(2,))
    mesh = create_mesh(mesh_comm, cells, points, domain)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

    print(f"✅ Rank {mesh_comm.rank}: Natively created mesh with {mesh.topology.index_map(0).size_global} global vertices.")
    return mesh


'''
  proc_id: process identifer for file generation
  geojson_path: Path to the GeoJSON file containing the features: inlet, outlet, obstacle, source.
  mesh_size_coarse: Coarse mesh size for the outer coastline.
  mesh_size_fine: Fine mesh size for the obstacles.
  preview_mesh: Visualization option for the generated mesh ("pyvista" or "matplotlib").
'''
def generate_surface_mesh(proc_id, geojson_path, mesh_size_coarse=500.0, mesh_size_fine=50.0, preview_mesh="pyvista"):
    if not os.path.exists(geojson_path):
        raise FileNotFoundError(f"GeoJSON file '{geojson_path}' not found.")

    #Check if the files are geojson files
    if not geojson_path.endswith('.geojson'):
        raise ValueError("Both input files must be in GeoJSON format.")

    gdim = 2
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh_filename = f"mesh_{proc_id}.msh"
    mesh_data_payload = {}
    if mesh_comm.rank == model_rank:
        #read the geojson files and extract the coordinates
        
        with open(geojson_path, "r") as f:
            geojson_data = json.load(f)

        wall_curves = []
        inlet_curves = []
        outlet_curves = []
        obstacle_curves = []
        obstacle_loops = []

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)

        gmsh.initialize()
        
        gmsh.model.add(proc_id)
        occ = gmsh.model.occ

        # --- Step 1: Parse Features & Create Geometry Entities ---
        all_outer_curves = []
        sources = []
        first_pt_tag = None  # To remember the very first point for closing the loop
        prev_end_pt_tag = None
        line_features = [f for f in geojson_data["features"] if f["geometry"]["type"] == "LineString"]
        polygon_features = [f for f in geojson_data["features"] if f["geometry"]["type"] == "Polygon"]
        sources_features =  [f for f in geojson_data["features"] if f["geometry"]["type"] == "Point"]

        min_lat, min_lon = np.inf, np.inf
        max_lat, max_lon = 0,0

        for i, feature in enumerate(line_features):
            geom = feature["geometry"]
            props = feature["properties"]
            b_type = props.get("boundary_type")
            coords = geom["coordinates"]

            pt_tags = []
            for j, (lon, lat) in enumerate(coords):
                x, y = transformer.transform(lon, lat)
                if x < min_lon:
                    min_lon = x
                if y < min_lat:
                    min_lat = y
                if x > max_lon:
                    max_lon = x
                if y > max_lat:
                    max_lat = y

                if j == 0 and prev_end_pt_tag is not None:
                    pt_tags.append(prev_end_pt_tag)  # Reuse the last point of the previous segment
                else:
                    tag = occ.addPoint(x, y, 0.0, mesh_size_coarse)
                    pt_tags.append(tag)
                    if first_pt_tag is None:
                        first_pt_tag = tag  # Capture the absolute start of the boundary

            # Handle the final loop closure on the very last line segment
            if i == len(line_features) - 1 and first_pt_tag is not None:
                # Replace the final point tag with the very first point tag to close the loop
                pt_tags[-1] = first_pt_tag

            prev_end_pt_tag = pt_tags[-1]
            
            # Create a B-Spline for this boundary segment
            spline_tag = occ.addBSpline(pt_tags)
            
            # Sort into respective buckets based on GeoJSON property
            if b_type == "wall":
                wall_curves.append(spline_tag)
            elif b_type == "inlet":
                inlet_curves.append(spline_tag)
            elif b_type == "outlet":
                outlet_curves.append(spline_tag)
            all_outer_curves.append(spline_tag)

        for feature in polygon_features:
            geom = feature["geometry"]
            # Islands/Obstacles (outer ring is coords[0])
            ring = geom["coordinates"][0]
            island_pts = []
            island_first_pt = None
            for j, (lon, lat) in enumerate(ring):
                x, y = transformer.transform(lon, lat)

                if j < len(ring) - 1:
                    tag = occ.addPoint(x, y, 0.0, mesh_size_fine) # Finer mesh near islands
                    island_pts.append(tag)
                    if j == 0:
                        island_first_pt = tag
                elif island_first_pt is not None:
                    island_pts.append(island_first_pt)

            
            island_spline = occ.addBSpline(island_pts)
            obstacle_curves.append(island_spline)
            
            obs_loop = occ.addCurveLoop([island_spline])
            obstacle_loops.append(obs_loop)

        x_origin = min_lon
        y_origin = min_lat

        for feature in sources_features:
            props = feature["properties"]
            if 'Q' not in props or 'sigma' not in props:
                continue
            geom = feature["geometry"]
            coords = geom["coordinates"]
            lon, lat = coords[0], coords[1]
            x_utm, y_utm = transformer.transform(lon, lat)
            sources.append({
                "x": x_utm - x_origin,
                "y": y_utm - y_origin,
                "Q": props.get('Q'),
                "sigma": props.get('sigma')
            })

        # --- Step 2: Build Outer Loop and Water Surface ---
        #all_outer_curves = wall_curves + inlet_curves + outlet_curves
        outer_loop = occ.addCurveLoop(all_outer_curves)

        # Create surface with holes (islands)
        water_surface = occ.addPlaneSurface([outer_loop] + obstacle_loops)
        occ.synchronize()
        occ.removeAllDuplicates()
        occ.synchronize()

        # --- Step 3: Define Physical Groups (Annotation for DOLFINx) ---
        # 1. 2D Domain Surface
        gmsh.model.addPhysicalGroup(2, [water_surface], DOMAIN_MARKER)
        gmsh.model.setPhysicalName(2, DOMAIN_MARKER, "Water Domain")

        # 2. 1D Boundary Curves
        if wall_curves:
            gmsh.model.addPhysicalGroup(1, wall_curves, WALL_MARKER)
            gmsh.model.setPhysicalName(1, WALL_MARKER, "Walls")
            
        if inlet_curves:
            gmsh.model.addPhysicalGroup(1, inlet_curves, INLET_MARKER)
            gmsh.model.setPhysicalName(1, INLET_MARKER, "Inlet")
            
        if outlet_curves:
            gmsh.model.addPhysicalGroup(1, outlet_curves, OUTLET_MARKER)
            gmsh.model.setPhysicalName(1, OUTLET_MARKER, "Outlet")
            
        if obstacle_curves:
            gmsh.model.addPhysicalGroup(1, obstacle_curves, OBSTACLE_MARKER)
            gmsh.model.setPhysicalName(1, OBSTACLE_MARKER, "Obstacle")

        # Generate 2D mesh
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

        # Force Gmsh to use linear elements (3-node triangles, type 2)
        gmsh.option.setNumber("Mesh.ElementOrder", 1)
        gmsh.model.mesh.generate(gdim)
        gmsh.model.mesh.setOrder(2)
        gmsh.model.mesh.optimize("Netgen")

        # --- Native Gmsh Off-Screen Preview Rendering ---
        os.makedirs("figures", exist_ok=True)
        image_path = f"figures/mesh_preview_{proc_id}.png"

        #gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        occ.synchronize()
        occ.removeAllDuplicates()
        occ.synchronize()

        gmsh.write(mesh_filename)
        
        if preview_mesh == "pyvista":
            temp_vtk = f"temp_preview_{proc_id}.vtk"
            gmsh.write(temp_vtk)
            grid = pv.read(temp_vtk)
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, show_edges=True, color="lightblue", edge_color="black")
            plotter.view_xy()
            plotter.screenshot(image_path, transparent_background=True)
            plotter.close()

        gmsh.finalize()
        return sources, [(min_lon, min_lat), (max_lon, max_lat)]


def load_mesh( proc_id="bay"):
    mesh_comm = MPI.COMM_WORLD
    model_rank = 0
    mesh_filename = f"mesh_{proc_id}.msh"

    ## Collective parallel read for FEniCSx solver
    mesh_data = gmshio.read_from_msh(mesh_filename, mesh_comm, model_rank, gdim=2)
    mesh_comm.Barrier()  
    mesh = mesh_data[0]
    ft = mesh_data[2]
    ft.name = "Facet markers"
    return mesh, ft

def run_ipcs_solver(mesh, ft, sources=[], T=8.0, dt=1/1600, gdim=2, verify=False):
    if mesh.comm.rank == 0:
        print(f"🚀 Starting Navier-Stokes IPCS Solver (T={T}s, dt={dt}s) sources {sources}...")
    
    t = 0.0
    T = 8.0  # Final time
    dt = 1 / 1600  # Time step size
    num_steps = int(T / dt)
    k = Constant(mesh, PETSc.ScalarType(dt))
    mu = Constant(mesh, PETSc.ScalarType(0.001))  # Dynamic viscosity
    rho = Constant(mesh, PETSc.ScalarType(1))  # Density

    v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
    c_element = element("Lagrange", mesh.basix_cell(), 1)
    V = functionspace(mesh, v_cg2)
    Q = functionspace(mesh, s_cg1)
    C_space = functionspace(mesh, c_element)

    fdim = mesh.topology.dim - 1

    class InletVelocity:
        def __init__(self, t):
            self.t = t

        def __call__(self, x):
            values = np.zeros((gdim, x.shape[1]), dtype=PETSc.ScalarType)
            values[0] = (
                4 * 1.5 * np.sin(self.t * np.pi / 8) * x[1] * (0.41 - x[1]) / (0.41**2)
            )
            return values


    # Inlet
    u_inlet = Function(V)
    inlet_velocity = InletVelocity(t)
    u_inlet.interpolate(inlet_velocity)
    bcu_inflow = dirichletbc(
        u_inlet, locate_dofs_topological(V, fdim, ft.find(INLET_MARKER))
    )
    # Walls
    u_nonslip = np.array((0,) * mesh.geometry.dim, dtype=PETSc.ScalarType)
    bcu_walls = dirichletbc(
        u_nonslip, locate_dofs_topological(V, fdim, ft.find(WALL_MARKER)), V
    )
    # Obstacle
    bcu_obstacle = dirichletbc(
        u_nonslip, locate_dofs_topological(V, fdim, ft.find(OBSTACLE_MARKER)), V
    )
    bcu = [bcu_inflow, bcu_obstacle, bcu_walls]
    # Outlet
    bcp_outlet = dirichletbc(
        PETSc.ScalarType(0), locate_dofs_topological(Q, fdim, ft.find(OUTLET_MARKER)), Q
    )
    bcp = [bcp_outlet]

    u = TrialFunction(V)
    v = TestFunction(V)
    u_ = Function(V, name="u")
    u_s = Function(V, name="u_tentative")
    u_n = Function(V)
    u_n1 = Function(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)
    p_ = Function(Q, name="p")
    phi = Function(Q, name="phi")

    c = TrialFunction(C_space)
    q_c = TestFunction(C_space)
    c_ = Function(C_space, name="c")       # Solution at step n+1
    c_n = Function(C_space, name="c_n")     # Solution at step n

    # Physical constants
    D = Constant(mesh, PETSc.ScalarType(0.001))   # Diffusivity
    K = Constant(mesh, PETSc.ScalarType(0.01))    # Reaction / decay rate

    f = Constant(mesh, PETSc.ScalarType((0, 0)))
    F1 = rho / k * dot(u - u_n, v) * dx
    F1 += inner(dot(1.5 * u_n - 0.5 * u_n1, 0.5 * nabla_grad(u + u_n)), v) * dx
    F1 += 0.5 * mu * inner(grad(u + u_n), grad(v)) * dx - dot(p_, div(v)) * dx
    F1 += dot(f, v) * dx
    a1 = form(lhs(F1))
    L1 = form(rhs(F1))
    A1 = create_matrix(a1)
    b1 = create_vector(extract_function_spaces(L1))


    a2 = form(dot(grad(p), grad(q)) * dx)
    L2 = form(-rho / k * dot(div(u_s), q) * dx)
    A2 = assemble_matrix(a2, bcs=bcp)
    A2.assemble()
    b2 = create_vector(extract_function_spaces(L2))

    a3 = form(rho * dot(u, v) * dx)
    L3 = form(rho * dot(u_s, v) * dx - k * dot(nabla_grad(phi), v) * dx)
    A3 = assemble_matrix(a3)
    A3.assemble()
    b3 = create_vector(extract_function_spaces(L3))

    # Point source (e.g., Gaussian injection near (x0, y0))
    class MultiPointGaussianSource:
        """Evaluates multi-point Gaussian plume injections across FEniCSx mesh nodes."""
        def __init__(self, sources, t=0.0):
            self.sources = sources  # List of dicts: [{'x', 'y', 'Q', 'sigma'}, ...]
            self.t = t

        def __call__(self, x):
            # x has shape (gdim, num_nodes); x[0] is X-coords, x[1] is Y-coords
            values = np.zeros(x.shape[1], dtype=np.float64)
            
            for src in self.sources:
                # Distance squared from current source center
                dist_sq = (x[0] - src["x"])**2 + (x[1] - src["y"])**2
                
                # Smooth Gaussian spatial distribution: S(x,y) = Q * exp(-dist^2 / (2 * sigma^2))
                values += src["Q"] * np.exp(-dist_sq / (2.0 * src["sigma"]**2))
                
            return values
    class SourceExpression:
        def __init__(self, t):
            self.t = t
        def __call__(self, x):
            return 10.0 * np.exp(-((x[0] - 0.15)**2 + (x[1] - 0.2)**2) / 0.005)

    S_func = Function(C_space, name="SourceTerm")
    S_expr = MultiPointGaussianSource(sources, t=0.0)
    #= SourceExpression(t)
    S_func.interpolate(S_expr)

    # Weak formulation
    F_c = (c - c_n) / k * q_c * dx \
        + dot(u_, grad(c)) * q_c * dx \
        + D * inner(grad(c), grad(q_c)) * dx \
        + K * c * q_c * dx \
        - S_func * q_c * dx

    a4 = form(lhs(F_c))
    L4 = form(rhs(F_c))

    A4 = assemble_matrix(a4)
    A4.assemble()
    b4 = create_vector(extract_function_spaces(L4))

    # Solver for step 1
    solver1 = PETSc.KSP().create(mesh.comm)
    solver1.setOperators(A1)
    solver1.setType(PETSc.KSP.Type.BCGS)
    pc1 = solver1.getPC()
    pc1.setType(PETSc.PC.Type.JACOBI)

    # Solver for step 2
    solver2 = PETSc.KSP().create(mesh.comm)
    solver2.setOperators(A2)
    solver2.setType(PETSc.KSP.Type.MINRES)
    pc2 = solver2.getPC()
    pc2.setType(PETSc.PC.Type.HYPRE)
    pc2.setHYPREType("boomeramg")

    # Solver for step 3
    solver3 = PETSc.KSP().create(mesh.comm)
    solver3.setOperators(A3)
    solver3.setType(PETSc.KSP.Type.CG)
    pc3 = solver3.getPC()
    pc3.setType(PETSc.PC.Type.SOR)

    # PETSc solver for step 4
    solver4 = PETSc.KSP().create(mesh.comm)
    solver4.setOperators(A4)
    solver4.setType(PETSc.KSP.Type.GMRES)
    pc4 = solver4.getPC()
    pc4.setType(PETSc.PC.Type.HYPRE)
    pc4.setHYPREType("boomeramg")

    n = -FacetNormal(mesh)  # Normal pointing out of obstacle
    dObs = Measure("ds", domain=mesh, subdomain_data=ft, subdomain_id=OBSTACLE_MARKER)
    u_t = inner(as_vector((n[1], -n[0])), u_)
    drag = form(2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[1] - p_ * n[0]) * dObs)
    lift = form(-2 / 0.1 * (mu / rho * inner(grad(u_t), n) * n[0] + p_ * n[1]) * dObs)
    if mesh.comm.rank == 0:
        C_D = np.zeros(num_steps, dtype=PETSc.ScalarType)
        C_L = np.zeros(num_steps, dtype=PETSc.ScalarType)
        t_u = np.zeros(num_steps, dtype=np.float64)
        t_p = np.zeros(num_steps, dtype=np.float64)


    tree = bb_tree(mesh, mesh.geometry.dim)
    points = np.array([[0.15, 0.2, 0], [0.25, 0.2, 0]])
    cell_candidates = compute_collisions_points(tree, points)
    colliding_cells = compute_colliding_cells(mesh, cell_candidates, points)
    front_cells = colliding_cells.links(0)
    back_cells = colliding_cells.links(1)
    if mesh.comm.rank == 0:
        p_diff = np.zeros(num_steps, dtype=PETSc.ScalarType)


    folder = Path("results")
    folder.mkdir(exist_ok=True, parents=True)
    vtx_u = VTXWriter(mesh.comm, folder / "dfg2D-3-u.bp", [u_], engine="BP4")
    vtx_p = VTXWriter(mesh.comm, folder / "dfg2D-3-p.bp", [p_], engine="BP4")
    vtx_c = VTXWriter(mesh.comm, folder / "dfg2D-3-c.bp", [c_], engine="BP4")
    vtx_u.write(t)
    vtx_p.write(t)
    vtx_c.write(t)
    progress = tqdm.autonotebook.tqdm(desc="Solving PDE", total=num_steps)
    for i in range(num_steps):
        progress.update(1)
        # Update current time step
        t += dt
        # Update inlet velocity
        inlet_velocity.t = t
        u_inlet.interpolate(inlet_velocity)

        # Step 1: Tentative velocity step
        A1.zeroEntries()
        assemble_matrix(A1, a1, bcs=bcu)
        A1.assemble()
        with b1.localForm() as loc:
            loc.set(0)
        assemble_vector(b1, L1)
        apply_lifting(b1, [a1], [bcu])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b1, bcu)
        solver1.solve(b1, u_s.x.petsc_vec)
        u_s.x.scatter_forward()

        # Step 2: Pressure corrrection step
        with b2.localForm() as loc:
            loc.set(0)
        assemble_vector(b2, L2)
        apply_lifting(b2, [a2], [bcp])
        b2.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b2, bcp)
        solver2.solve(b2, phi.x.petsc_vec)
        phi.x.scatter_forward()

        p_.x.petsc_vec.axpy(1, phi.x.petsc_vec)
        p_.x.scatter_forward()

        # Step 3: Velocity correction step
        with b3.localForm() as loc:
            loc.set(0)
        assemble_vector(b3, L3)
        b3.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        solver3.solve(b3, u_.x.petsc_vec)
        u_.x.scatter_forward()

        # Step 4: Concentration Transport Step
        A4.zeroEntries()
        assemble_matrix(A4, a4)
        A4.assemble()
        
        with b4.localForm() as loc:
            loc.set(0)
        assemble_vector(b4, L4)
        b4.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        solver4.solve(b4, c_.x.petsc_vec)
        c_.x.scatter_forward()

        # Update previous step concentration
        c_.x.petsc_vec.copy(c_n.x.petsc_vec)


        # Write solutions to file
        vtx_u.write(t)
        vtx_p.write(t)
        vtx_c.write(t)

        # Update variable with solution form this time step
        with (
            u_.x.petsc_vec.localForm() as loc_,
            u_n.x.petsc_vec.localForm() as loc_n,
            u_n1.x.petsc_vec.localForm() as loc_n1,
        ):
            loc_n.copy(loc_n1)
            loc_.copy(loc_n)

        # Compute physical quantities
        # For this to work in paralell, we gather contributions from all processors
        # to processor zero and sum the contributions.
        drag_coeff = mesh.comm.gather(assemble_scalar(drag), root=0)
        lift_coeff = mesh.comm.gather(assemble_scalar(lift), root=0)
        p_front = None
        if len(front_cells) > 0:
            p_front = p_.eval(points[0], front_cells[:1])
        p_front = mesh.comm.gather(p_front, root=0)
        p_back = None
        if len(back_cells) > 0:
            p_back = p_.eval(points[1], back_cells[:1])
        p_back = mesh.comm.gather(p_back, root=0)
        if mesh.comm.rank == 0:
            t_u[i] = t
            t_p[i] = t - dt / 2
            C_D[i] = sum(drag_coeff)
            C_L[i] = sum(lift_coeff)
            # Choose first pressure that is found from the different processors
            for pressure in p_front:
                if pressure is not None:
                    p_diff[i] = pressure[0]
                    break
            for pressure in p_back:
                if pressure is not None:
                    p_diff[i] -= pressure[0]
                    break
    progress.close()
    vtx_u.close()
    vtx_p.close()
    vtx_c.close


    A1.destroy()
    A2.destroy()
    A3.destroy()
    A4.destroy()
    b1.destroy()
    b2.destroy()
    b3.destroy()
    b4.destroy()
    solver1.destroy()
    solver2.destroy()
    solver3.destroy()
    solver4.destroy()
    if  verify and mesh.comm.rank == 0:
        run_verification(mesh, ft, t_u, C_D, C_L, t_p, p_diff)

def run_verification(mesh, ft, t_u, C_D, C_L, t_p, p_diff):
    if mesh.comm.rank == 0:
        v_cg2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
        s_cg1 = element("Lagrange", mesh.basix_cell(), 1)
        V = functionspace(mesh, v_cg2)
        Q = functionspace(mesh, s_cg1)

        fdim = mesh.topology.dim - 1

        if not os.path.exists("figures"):
            os.mkdir("figures")
        num_velocity_dofs = V.dofmap.index_map_bs * V.dofmap.index_map.size_global
        num_pressure_dofs = Q.dofmap.index_map_bs * V.dofmap.index_map.size_global

        turek = np.loadtxt("bdforces_lv4")
        turek_p = np.loadtxt("pointvalues_lv4")
        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(
            t_u,
            C_D,
            label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs),
            linewidth=2,
        )
        l2 = plt.plot(
            turek[1:, 1],
            turek[1:, 3],
            marker="x",
            markevery=50,
            linestyle="",
            markersize=4,
            label="FEATFLOW (42016 dofs)",
        )
        plt.title("Drag coefficient")
        plt.grid()
        plt.legend()
        plt.savefig("figures/drag_comparison.png")

        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(
            t_u,
            C_L,
            label=r"FEniCSx  ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs),
            linewidth=2,
        )
        l2 = plt.plot(
            turek[1:, 1],
            turek[1:, 4],
            marker="x",
            markevery=50,
            linestyle="",
            markersize=4,
            label="FEATFLOW (42016 dofs)",
        )
        plt.title("Lift coefficient")
        plt.grid()
        plt.legend()
        plt.savefig("figures/lift_comparison.png")

        fig = plt.figure(figsize=(25, 8))
        l1 = plt.plot(
            t_p,
            p_diff,
            label=r"FEniCSx ({0:d} dofs)".format(num_velocity_dofs + num_pressure_dofs),
            linewidth=2,
        )
        l2 = plt.plot(
            turek[1:, 1],
            turek_p[1:, 6] - turek_p[1:, -1],
            marker="x",
            markevery=50,
            linestyle="",
            markersize=4,
            label="FEATFLOW (42016 dofs)",
        )
        plt.title("Pressure difference")
        plt.grid()
        plt.legend()
        plt.savefig("figures/pressure_comparison.png")



# ---------------------------------------------------------
# 4. EXECUTION CONTROL (CLI)
# ---------------------------------------------------------
if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(description="DFG 2D-3 Navier-Stokes Solver with Modular Mesh Control")
    parser.add_argument("--mesh-only", action="store_true", help="Generate and visualize mesh without running PDE solver")
    parser.add_argument("--verify", action="store_true", help="Verify solver results against benchmark data and generate comparison plots")
    parser.add_argument("--use-matplotlib", action="store_true", help="Use matplotlib for mesh visualization")
    parser.add_argument("-c", "--size-coarse", type=float, default=500.0, help="The Mesh size at coarse intended domain")
    parser.add_argument("-f", "--size-fine", type=float, default=500.0, help="The Mesh size at fine intended domain")
    args = parser.parse_args()


    preview = "matplotlib" if args.use_matplotlib else "pyvista"

    #mesh, ft = generate_dfg2d_mesh(preview_mesh=preview)
    #bbc-harbor-shoreline.geojson
    #simple-surface.geojson
    #simplified-bbc-shoreline-island.geojson
    #simplif'ied-bbc-shoreline.geojson
    sources,bbox = generate_surface_mesh(proc_id="test0", geojson_path="./test-data/simplified-bbc-shoreline-island.geojson", mesh_size_coarse=args.size_coarse, mesh_size_fine=args.size_fine)
    L_ref = bbox[1][0] - bbox[0][0]
    print(L_ref)
    sources = comm.bcast(sources, root=0)
    comm.Barrier() 
    mesh, ft = load_mesh(proc_id="test0")
    if args.mesh_only:
        if rank == 0:
            print("Stopped execution after mesh generation (--mesh-only flag set).")
        sys.exit(0)
    
    if args.verify:
        run_ipcs_solver(mesh, ft, sources=sources, verify=True)
    else:
        run_ipcs_solver(mesh, ft, sources=sources)
