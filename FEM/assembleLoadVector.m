function F = assembleLoadVector(mesh, feMap, u, f_r, f_t, f_d, a)
    % Initialize global load vector
    F = zeros(mesh.numVertices, 1);

    % Reference element shape functions evaluated at integration points
    % refShapeFunctions = [2/3 1/6 1/6; 1/6 2/3 1/6; 1/6 1/6 2/3]; % Shape functions at the integration points
    refShapeFunctions = [1/3 1/3 1/3]; % Shape functions at the centroid
    q_len = size(refShapeFunctions, 1);

    for e = 1:mesh.numMeshElements
        % Nodes of the current element
        nodes = mesh.meshElements(:, e);

        % Element solution values at nodes
        u_e = u(nodes);

        % Compute the local load vector
        F_loc = zeros(3, 1);
        for q = 1:q_len
            % Evaluate the shape functions at the quadrature points
            N = refShapeFunctions(q, :);

            % Evaluate the integrand at the quadrature points
            integrand = a * (N * u_e - f_r) * (N * u_e - f_t) * (N * u_e - f_d);

            % Compute the local load vector contribution
            F_loc = F_loc + integrand * N' * feMap.J(e) / 2;
        end

        % Assemble the local load vector into the global load vector
        F(nodes) = F(nodes) + F_loc;
    end
end