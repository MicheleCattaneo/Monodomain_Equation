function mesh = readMesh_msh(file_name)

    commandString = ['head -n1', file_name , ' > temp.txt'];
    system(commandString);
    data = load ('temp . txt');

    mesh.number_of_vertices = data(1);
    mesh.number_of_elements = data(2);
    mesh.number_of_boundary_sides = data(3);

    commandString = ['head -n', num2str(data(1)+1), ' ', file_name, ' > temp.txt'];
    system(commandString);
    data = load('temp.txt');

    mesh.vertices = data(2:end, 1:2)';
    mesh.vertices_flag = data(2:end, 3)';

    commandString = ['tail -n', num2str(mesh.number_of_boundary_sides), ' ', file_name, ' > temp.txt'];
    system(commandString);
    data = load('temp.txt');

    mesh.boundary_sides = data(:, 1:2)';
    mesh.boundary_sides_flag = data(:, 3)';

    commandString = ['tail -n', num2str(mesh.number_of_vertices + mesh.number_of_elements + 1), ' ', file_name, ' > temp.txt'];
    system(commandString);
    commandString = ['head -n', num2str(mesh.number_of_elements), ' ', 'temp.txt', '> temp2.txt'];
    system(commandString);
    data = load('temp2.txt');

    mesh.elements = data(:, 1:3)';
    mesh.elements_flag = data(:, 4)';

    !rm temp.txt temp2.txt
    !rm temp.txt

    return