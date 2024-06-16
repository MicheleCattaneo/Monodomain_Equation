% Define the mesh
mesh = Mesh2D('mesh_0128.msh');

% Define the finite element map
feMap = FEMap(mesh);

sigma_h = 9.598e-4; % Define sigma_h
sigma_d = 0.1 * sigma_h; % Define sigma_d
a = 18.515; % Define constant a
f_t = 0.2383; % Define function f_t
f_r = 0; % Define function f_r
f_d = 1; % Define function f_d
T_f = 35; % Final time in ms
numSteps = 350; % Number of time steps
videoFileName = 'solution.mp4'; % Video file name

% Plot initial condition
% fig = figure;
% mesh.plotSolution(initialCondition(mesh));

solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName);

% Main function to solve the problem
function solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName)
    % Activation flag
    flag = 0;

    % Time step
    dt = T_f / numSteps;

    % Assemble the mass matrix
    M = assembleMass(mesh, feMap);
    M = lumpMassMatrix(M);

    % Check number of zero elements per row in the mass matrix
    % numZeroElements = sum(M(:,71) ~= 0);
    % disp(['Number of zero elements in the mass matrix: ', num2str(numZeroElements)]);

    % Assemble the diffusion matrix
    K = assembleDiffusion(mesh, feMap, sigma_d, sigma_h);

    % Form the system matrix
    A = (M / dt) + K;

    % Check if the mass matrix is M-matrix
    is_M_matrix = is_M_Matrix(M);
    if is_M_matrix
        disp('The system matrix is an M-matrix.');
    else
        disp('The system matrix is not an M-matrix.');
    end

    % Initial condition
    u = initialCondition(mesh);

    % Initialize video writer
    videoWriter = VideoWriter(videoFileName, 'MPEG-4');
    open(videoWriter);

    % Plot and capture the initial condition
    fig = figure('visible', 'off'); % Create a figure without displaying it
    mesh.plotSolution(u);
    title('Time = 0 ms');
    frame = getframe(fig); % Capture the frame
    writeVideo(videoWriter, frame); % Write the frame to the video
    close(fig); % Close the figure

    % Initialize progress bar
    hWaitBar = waitbar(0, 'Solving PDE...', 'Name', 'Progress');

    % Time-stepping loop
    for n = 1:numSteps
        % Assemble the load vector
        F = assembleLoadVector(mesh, feMap, u, f_r, f_t, f_d, a);

        % Right-hand side vector
        b = (M / dt) * u - F;

        % Solve the linear system
        u = A \ b;

        % Plot and capture the solution at each time step
        fig = figure('visible', 'off'); % Create a figure without displaying it
        mesh.plotSolution(u);
        title(['Time = ', num2str(n * dt), ' ms']);
        frame = getframe(fig); % Capture the frame
        writeVideo(videoWriter, frame); % Write the frame to the video
        close(fig); % Close the figure

        if all(u >= f_t) && flag == 0
            flag = 1;
            time = n * dt;
            disp(['The solution exceeds the threshold at time t = ', num2str(time), 'ms.']);
        end

        % Calculate potential excess
        if max(u) > 1
            %disp('The potential is above the upper bound.');
        elseif min(u) < 0
            %disp('The potential is below the lower bound.');
            %disp(min(u));
        end

        % Update progress bar
        waitbar(n / numSteps, hWaitBar);
    end

    % Close the progress bar
    close(hWaitBar);

    % Close the video writer
    close(videoWriter);
end

function u0 = initialCondition(mesh)
    % Initial condition based on the problem statement
    u0 = zeros(mesh.numVertices, 1);
    for i = 1:mesh.numVertices
        x = mesh.vertices(1, i);
        y = mesh.vertices(2, i);
        if x >= 0.9 && y >= 0.9
            u0(i) = 1;
        else
            u0(i) = 0;
        end
    end
end

function is_M_matrix = is_M_Matrix(M)
    % Diagonally dominant matrix
    cond1 = all(2 * diag(M) >= sum(abs(M), 2));

    % Check if off-diagonal elements are non-positive and diagonal elements are positive
    cond2 = all(diag(M) > 0) && all(M(~eye(size(M))) <= 0);
    
    % Check if the matrix is a m-matrix
    is_M_matrix = cond1 && cond2;
end

function M_lumped = lumpMassMatrix(M)
    % Lumped mass matrix
    M_lumped = diag(sum(M,2));
end

function PDE_error = calculateError(mesh, feMap, u0, u1, dt, sigma_h, sigma_d, f_r, f_t, f_d, a)
    % Calculate the error of the solution
    % PDE
    % du/dt = sigma_h * grad(grad(u)) + sigma_d * grad(grad(u)) - a * (u - f_r) * (u - f_t) * (u - f_d)
    grad_u_t = zeros(mesh.numVertices, 1);
    grad2_u_x = zeros(mesh.numVertices, 1);
    grad2_u_y = zeros(mesh.numVertices, 1);
    PDE_error = 0;
    for e = 1:mesh.numMeshElements
        % Nodes of the current element
        nodes = mesh.meshElements(:, e);

        % Coordinates of the nodes
        x = mesh.vertices(1, nodes);
        y = mesh.vertices(2, nodes);

        % Node solution values
        u0_e = u0(nodes);
        u1_e = u1(nodes);

        % Time gradient of the solution
        grad_u_t = grad_u_t + (u1_e - u0_e) / dt;

        for v = 1:3
            % Space gradient of the solution
            x_v = x(v);


            % Evaluate the PDE at the node
            PDE = 0;

            % Accumulate the error
            PDE_error = PDE_error + PDE / (3 * mesh.numMeshElements);
        end
    end
end
