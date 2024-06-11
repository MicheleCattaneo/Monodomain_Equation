classdef Mesh2D < handle

    properties
        % Number of vertices
        numVertices

        % Number of mesh elements
        numMeshElements

        % Number of boundary elements
        numBoundaryElements

        % A 2 by numVertices matrix containing coordinates of the vertices
        vertices

        % An array to store flags for vertices
        vertexFlags

        % A 3 by numMeshElements matrix for mesh elements
        meshElements

        % An array to store flags for mesh elements
        meshElementFlags

        % A 2 by numBoundaryElements matrix for boundary elements
        boundaryElements

        % An array to store flags for boundary elements
        boundaryElementFlags
    end

    methods
        % Constructor method to read mesh data from a file
        function obj = Mesh2D(fileName)
            tic
            temp = readMesh_msh(fileName);
            toc

            obj.numVertices = temp.number_of_vertices;
            obj.numMeshElements = temp.number_of_elements;
            obj.numBoundaryElements = temp.number_of_boundary_sides;
            obj.vertices = temp.vertices;
            obj.vertexFlags = temp.vertices_flag;
            obj.meshElements = temp.elements;
            obj.meshElementFlags = temp.elements_flag;
            obj.boundaryElements = temp.boundary_sides;
            obj.boundaryElementFlags = temp.boundary_sides_flag;
        end

        % Method to plot the mesh using trimesh
        function plotMesh(obj)
            close all
            trimesh(obj.meshElements', obj.vertices(1 , :), obj.vertices(2 , :));
            axis equal
        end

        % Method to plot the mesh with solution values using trimesh
        function plotSolution(obj, solution)
            trimesh(obj.meshElements', obj.vertices(1 , :), obj.vertices(2 , :), solution);
        end
    end
end