        if preview_mesh == "pyvista":
            mesh = meshio.read(mesh_filename)
            #print(mesh)
            triangles = []
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    triangles.append(cell_block.data)

            # 3. Build the PyVista UnstructuredGrid safely if triangles exist
            if triangles:
                # Stack multiple blocks if meshio broke triangles into separate pieces
                cell_matrix = np.vstack(triangles) if len(triangles) > 1 else triangles[0]
                
                # PyVista format expects a padding column showing the number of vertices per cell (3 for triangles)
                num_cells = cell_matrix.shape[0]
                padding = np.full((num_cells, 1), 3)
                
                # Flatten into VTK format: [3, v0, v1, v2, 3, v0, v1, v2...]
                cells = np.hstack((padding, cell_matrix)).ravel()
                
                # Assign the matching VTK cell type (VTK_TRIANGLE is 5)
                cell_types = np.full(num_cells, 5, dtype=np.uint8)
                
                # Create the mesh container explicitly using mesh.points
                grid = pv.UnstructuredGrid(cells, cell_types, mesh.points)
            else:
                # Fallback to an empty point cloud if no triangles are found
                grid = pv.PolyData(mesh.points)

            #plot_mesh_pyvista(mesh, f"figures/mesh_preview_{proc_id}.png")
            os.makedirs("figures", exist_ok=True)
            output_path = f"figures/mesh_preview_{proc_id}.png"
            plotter = pv.Plotter(off_screen=True)
            plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black", line_width=0.4)
            plotter.view_xy()
            plotter.screenshot(output_path)
            plotter.close()
            print(f"✅ Mesh visualization generated at: {output_path}")

            # PyVista reads .msh files natively for quick standalone inspection
            #pv_grid = pv.read(temp_vtk)
            #plotter = pv.Plotter(off_screen=True)
            #plotter.add_mesh(pv_grid, show_edges=True)
            #plotter.screenshot(f"figures/mesh_preview_{proc_id}.png")
            #plotter.close()