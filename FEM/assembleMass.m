function M = assembleMass(mesh, feMap)
    % Reference mass matrix for a triangle
    refMassMatrix = [2 1 1;
                     1 2 1;
                     1 1 2] / 24;

    % Initialize global mass matrix in vector form
    MVector = zeros(9, mesh.numMeshElements);

    % Node indices for assembly
    nodeIndI = [1 2 3 1 2 3 1 2 3];
    nodeIndJ = [1 1 1 2 2 2 3 3 3];

    % Global row and column indices for sparse matrix assembly
    globRows = mesh.meshElements(nodeIndI, :);
    globCols = mesh.meshElements(nodeIndJ, :);

    for e = 1:mesh.numMeshElements
        % Jacobian determinant for the current element
        detJ = feMap.J(e);

        % Local mass matrix for the current element
        M_loc = detJ * refMassMatrix;

        % Store the local mass matrix in vector form
        MVector(:, e) = M_loc(:);
    end

    % Assemble the global mass matrix using sparse format
    M = sparse(globRows(:), globCols(:), MVector(:), mesh.numVertices, mesh.numVertices);
end