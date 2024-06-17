sigma_h = 9.598e-4; % Define sigma_h
a = 18.515; % Define constant a
f_t = 0.2383; % Define function f_t
f_r = 0; % Define function f_r
f_d = 1; % Define function f_d
T_f = 35; % Final time in ms

% Define the meshes
meshes = ["mesh_0256", "mesh_0128"];

sigma_ds = [0.1, 1, 10] * sigma_h; % Define sigma_d
Step_list = [350, 700]; % Number of time steps

% for mesh_name = meshes
%     filename = convertStringsToChars(strcat(mesh_name, ".msh"));
%     mesh = Mesh2D(filename);
%     % Define the finite element map
%     feMap = FEMap(mesh);

%     for sigma_d = sigma_ds
%         for numSteps = Step_list
%             videoFileName = ['solution_', convertStringsToChars(mesh_name), '_', num2str(sigma_d), '_', num2str(numSteps), '.mp4'];
%             % Solve the PDE
%             solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName);
%         end
%     end
% end  

mesh = Mesh2D('mesh_0128.msh');
feMap = FEMap(mesh);
sigma_d = 0.1*sigma_h;
numSteps = 350;
videoFileName = 'solution.mp4';

solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName);

% Main function to solve the problem
function solvePDE(mesh, feMap, sigma_h, sigma_d, a, f_r, f_t, f_d, T_f, numSteps, videoFileName)
    % Activation flag
    activation_flag = 0;

    % Boundary exceeded flag
    boundary_exceeded_flag = 0;

    % Bounds for the potential
    max_u = 1;
    min_u = 0;

    % Time step
    dt = T_f / numSteps;

    % Assemble the mass matrix
    M = assembleMass(mesh, feMap);

    % Assemble the diffusion matrix
    K = assembleDiffusion(mesh, feMap, sigma_d, sigma_h);

    % Form the system matrix
    A = (M / dt) + K;

    % Check if the mass matrix is M-matrix
    is_M_matrix = check_M_Matrix(K);
    if is_M_matrix
        disp('The mass matrix is an M-matrix.');
    else
        disp('The mass matrix is not an M-matrix.');
        %M = lumpMassMatrix(M);
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
        % F = assembleLoadVector(mesh, feMap, u, f_r, f_t, f_d, a);
        f_u = f(u, f_r, f_t, f_d, a);

        % Right-hand side vector
        b = (M / dt) * u - M * f_u;

        % Solve the linear system
        u = A \ b;

        % Plot and capture the solution at each time step
        fig = figure('visible', 'off'); % Create a figure without displaying it
        mesh.plotSolution(u);
        title(['Time = ', num2str(n * dt), ' ms']);
        frame = getframe(fig); % Capture the frame
        writeVideo(videoWriter, frame); % Write the frame to the video
        close(fig); % Close the figure

        if all(u >= f_t) && activation_flag == 0
            activation_flag = 1;
            time = n * dt;
            disp(['The solution exceeds the threshold at time t = ', num2str(time), 'ms.']);
        end

        % Calculate potential excess
        if max(u) > 1 + 1e-10
            boundary_exceeded_flag = 1;
            if max(u) > max_u
                max_u = max(u);
            end
        elseif min(u) < -1e-10
            boundary_exceeded_flag = 1;
            if min(u) < min_u
                min_u = min(u);
            end
        end

        % Update progress bar
        waitbar(n / numSteps, hWaitBar);
    end

    % Close the progress bar
    close(hWaitBar);

    % Close the video writer
    close(videoWriter);

    if boundary_exceeded_flag == 0
        disp('The potential remained within the bounds.');
    else
        disp('The potential exceeded the bounds.');
        disp(['Maximum potential: ', num2str(max_u)]);
        disp(['Minimum potential: ', num2str(min_u)]);
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

function f_u = f(u, f_r, f_t, f_d, a)
    % Reaction term
    f_u = a * (u - f_r) .* (u - f_t) .* (u - f_d);
end

% function is_M_matrix = is_M_Matrix(M)
%     % Diagonally dominant matrix
%     cond1 = all(2 * diag(M) >= sum(abs(M), 2));
%     %disp(cond1);

%     % Check if diagonal elements are positive
%     cond2 = all(diag(M) > 0);
%     %disp(cond2);

%     % Check if off-diagonal elements are non-positive
%     cond3 = all(M(~speye(size(M))) <= 0);
%     disp(cond3);
    
%     % Check if the matrix is a m-matrix
%     is_M_matrix = cond1 && cond2 && cond3;
% end

function is_M_matrix = check_M_Matrix(M)
    % Extract the diagonal elements
    diag_M = diag(M);
    
    % Check if diagonal elements are positive
    cond1 = all(diag_M > 0);

    % Initialize conditions
    cond2 = true;
    cond3 = true;

    % Get the size of the matrix
    n = size(M, 1);

    % Loop through each row
    for i = 1:n
        % Get the row of M
        row = M(i, :);

        % Check diagonal dominance condition
        if full(2 * diag_M(i) - sum(abs(row)) + 1e-10) < 0
            disp("cond1 failed")
            cond2 = false;
            disp(i);
            disp(row);
            disp(2 * diag_M(i)-sum(abs(row)));
            disp(sum(abs(row)));
            disp(2 * diag_M(i));
            break;
        end
        
        % Check if off-diagonal elements are non-positive
        % (sparse matrix automatically skips zero elements)
        row(i) = 0; % Temporarily set diagonal element to zero
        if any(row > 1e-10)
            disp("cond3 failed")
            cond3 = false;
            disp(i);
            disp(row);
            disp(row(row > 0));
            break;
        end
    end

    disp(cond1);
    disp(cond2);
    disp(cond3);
    
    % Check if the matrix is an M-matrix
    is_M_matrix = cond1 && cond2 && cond3;
end

function M_lumped = lumpMassMatrix(M)
    % Lumped mass matrix
    M_lumped = diag(sum(M,2));
end
