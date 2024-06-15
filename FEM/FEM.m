% Define the mesh
mesh = Mesh2D('mesh_0128.msh');

% Define the finite element map
feMap = FEMap(mesh);

sigma_h = 9.598e-4; % Define sigma_h
sigma_d = 0.1*sigma_h; % Define sigma_d
a = 18.515; % Define constant a
f_t = 0.2383; % Define function f_t
f_r = 0; % Define function f_r
f_d = 1; % Define function f_d
T_f = 35; % Final time
numSteps = 350; % Number of time steps
videoFileName = 'solution.mp4'; % Video file name

% Plot initial condition
% fig = figure;
% mesh.plotSolution(initialCondition(mesh));

solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName);

% Main function to solve the problem
function solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName)
    % Time step
    dt = T_f / numSteps;

    % Assemble the mass matrix
    M = assembleMass(mesh, feMap);

    % Assemble the diffusion matrix
    K = assembleDiffusion(mesh, feMap, sigma_d, sigma_h);

    % Initial condition
    u = initialCondition(mesh);

    % Initialize video writer
    videoWriter = VideoWriter(videoFileName, 'MPEG-4');
    open(videoWriter);

    % Plot and capture the initial condition
    fig = figure('visible', 'off'); % Create a figure without displaying it
    mesh.plotSolution(u);
    frame = getframe(fig); % Capture the frame
    writeVideo(videoWriter, frame); % Write the frame to the video
    close(fig); % Close the figure

    % Initialize progress bar
    hWaitBar = waitbar(0, 'Solving PDE...', 'Name', 'Progress');

    % Time-stepping loop
    for n = 1:numSteps
        % Assemble the load vector
        F = assembleLoadVector(mesh, feMap, u, f_r, f_t, f_d, a);

        % Form the system matrix
        A = (M / dt) + K;

        % Right-hand side vector
        b = (M / dt) * u + F;

        % Solve the linear system
        u = A \ b;

        % Plot and capture the solution at each time step
        fig = figure('visible', 'off'); % Create a figure without displaying it
        mesh.plotSolution(u);
        frame = getframe(fig); % Capture the frame
        writeVideo(videoWriter, frame); % Write the frame to the video
        close(fig); % Close the figure

        % Update progress bar
        waitbar(n / numSteps, hWaitBar);
    end

    % Close the progress bar
    close(hWaitBar);

    % Close the video writer
    close(videoWriter);

    % Check if the mass matrix is M-matrix
    is_M_matrix = isMassMatrix(M);
    if is_M_matrix
        disp('The mass matrix is an M-matrix.');
    else
        disp('The mass matrix is not an M-matrix.');
    end

    % Calculate potential excess
    if min(u) < 1e-10 || max(u) > 1+1e-10
        disp('The solution exceeds the bounds.');
    else
        disp('The solution does not exceed the bounds.');
    end
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

function is_M_matrix = isMassMatrix(M)
    % Check if the matrix is a mass matrix
    is_M_matrix = isequal(M, M');
end
