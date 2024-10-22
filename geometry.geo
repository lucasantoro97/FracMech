
    W = 100.0;
    H = 50.0;
    crack_length = 1.0;

    // Define corner points
    Point(1) = {0, 0, 0, 1.0};
    Point(2) = {100.0, 0, 0, 1.0};
    Point(3) = {100.0, 50.0, 0, 1.0};
    Point(4) = {0, 50.0, 0, 1.0};

    // Define the crack (embedded line)
    Point(5) = {49.5, 25.0, 0, 0.1};
    Point(6) = {50.5, 25.0, 0, 0.1};
    Line(5) = {5, 6};

    // Define lines (boundary)
    Line(1) = {1, 2};
    Line(2) = {2, 3};
    Line(3) = {3, 4};
    Line(4) = {4, 1};

    // Define the loop and surface
    Line Loop(1) = {1, 2, 3, 4};
    Plane Surface(1) = {1};

    // Embed the crack line in the surface
    Line{5} In Surface{1};

    // Define physical groups with explicit IDs
    Physical Line("Left") = {4};
    Physical Line("Right") = {2};
    Physical Line("Top") = {3};
    Physical Line("Bottom") = {1};
    Physical Line("Crack") = {5};
    Physical Surface("Specimen") = {1};
    