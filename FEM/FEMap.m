classdef FEMap < handle

    properties
        % the map is F* vertices + b
        % The Jacobian of the transformation
        F

        % Translation vector
        b

        % Determinant of the Jacobian
        J % detF

        % Transpose of the Jacobian multiplied by the Jacobian
        C

        % Metric tensor , calculated as J * inv (C)
        metricTensor
    end

    methods
        function obj = FEMap(mesh2D)

            numElements = mesh2D.numMeshElements;
            obj.F = zeros(2, 2, numElements);
            obj.C = zeros(2, 2, numElements);
            obj.metricTensor = zeros(2, 2, numElements);
            obj.b = zeros(2, numElements);
            obj.J = zeros(numElements, 1);

            p1 = mesh2D.vertices(:, mesh2D.meshElements(1, :));
            p2 = mesh2D.vertices(:, mesh2D.meshElements(2, :));
            p3 = mesh2D.vertices(:, mesh2D.meshElements(3, :));
            obj.F(:, 1, :) = p2 - p1;
            obj.F(:, 2, :) = p3 - p1;

            obj.b = p1;

            a = mesh2D.vertices(1, mesh2D.meshElements(1, :));
            b = mesh2D.vertices(1, mesh2D.meshElements(2, :));
            c = mesh2D.vertices(1, mesh2D.meshElements(3, :));
            d = mesh2D.vertices(2, mesh2D.meshElements(1, :));
            e = mesh2D.vertices(2, mesh2D.meshElements(2, :));
            f = mesh2D.vertices(2, mesh2D.meshElements(3, :));

            obj.J = abs(a .* e + b .* f + c .* d - a .* f - b .* d - c .* e);

            for k = 1:numElements
                obj.C(:, :, k) = obj.F(:, :, k)' * obj.F(:, :, k);
                obj.metricTensor(:, :, k) = obj.J(k) *  (obj.C(:, :, k) \ eye(2));
            end
        end
    end
end