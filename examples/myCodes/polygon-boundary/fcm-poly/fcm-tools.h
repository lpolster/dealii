const unsigned int global_refinement_level = 1;
const float beta_h = 2.0/(1.0/ global_refinement_level);    // beta divided by h, 2.0/0.0625
const float dirichlet_boundary_value = 0.000;
const unsigned int refinement_cycles = 0;
//___________________________________________
dealii::Quadrature<2> collect_quadratures(typename dealii::Triangulation<2>::cell_iterator cell,
                                          const dealii::Quadrature<2>* base_quadrature)
{
    if(cell->active())
    {
        // not refined, return copy of base quadrature
        return *base_quadrature;
    }
    // get collected quadratures of each children and merge them
    std::vector<dealii::Point<2> > q_points;
    std::vector<double> q_weights;
    for(unsigned int child = 0;
        child < dealii::GeometryInfo<2>::max_children_per_cell;
        ++child)
    {
        // get child
        typename dealii::Triangulation<2>::cell_iterator child_cell =
        cell->child(child);
        // collect sub-quadratures there
        dealii::Quadrature<2> childs_collected_quadratures =
        collect_quadratures(child_cell, base_quadrature);
        // project to current cell
        dealii::Quadrature<2> child_quadrature =
        dealii::QProjector<2>::project_to_child(childs_collected_quadratures, child);
        // collect resulting quadrature
        q_points.insert(q_points.end(),
                        child_quadrature.get_points().begin(),
                        child_quadrature.get_points().end());
        q_weights.insert(q_weights.end(),
                         child_quadrature.get_weights().begin(),
                         child_quadrature.get_weights().end());
    }
    
    return dealii::Quadrature<2>(q_points, q_weights);
}
//______________________________________
dealii::Quadrature<2> collect_quadratures_on_boundary_segment(myPolygon::segment my_segment,
                                          const dealii::Quadrature<2>* base_quadrature)
{
    return dealii::Quadrature<2>(my_segment.q_points, my_segment.q_weights);
}

//______________________________________
void plot_in_global_coordinates (std::vector<dealii::Point<2>> q_points,
                                 dealii::DoFHandler<2>::cell_iterator cell, std::string filename)
{
    std::ofstream ofs_quadrature_points;
    
    ofs_quadrature_points.open (filename, std::ofstream::out | std::ofstream::app);
    
    for (unsigned int i = 0; i<q_points.size(); ++i) // loop over all quadrature points
    {
        q_points[i][0] = (q_points[i][0] * (cell->vertex(1)[0] - cell->vertex(0)[0])) + cell->vertex(0)[0]; // calculate x location of quadrature point on reference cell
        q_points[i][1] = (q_points[i][1] * (cell->vertex(2)[1] - cell->vertex(0)[1])) + cell->vertex(0)[1]; // calculate y location of quadrature point on reference cell
        
        ofs_quadrature_points<<q_points[i][0]<<" "<< q_points[i][1]<<std::endl;
    }
    
    ofs_quadrature_points.close();
}
//___________________________________________

std::vector<double> get_indicator_function_values(const std::vector<dealii::Point<2> > &points,
                                                  typename dealii::DoFHandler<2>::cell_iterator solution_cell, myPolygon my_poly)
{
    std::ofstream ofs_indicator_function_values;
    ofs_indicator_function_values.open ("indicator_function_values", std::ofstream::out | std::ofstream::app);
    
    double x, y;
    std::vector<double> indicator_function_values (points.size());
    for (unsigned int i=0; i<indicator_function_values.size(); ++i)
    {
        x = points[i][0] * (solution_cell->vertex(1)[0]-solution_cell->vertex(0)[0]) + solution_cell->vertex(0)[0];
        y = points[i][1] * (solution_cell->vertex(2)[1]-solution_cell->vertex(0)[1]) + solution_cell->vertex(0)[1];
        dealii::Point<2> p1 = {x,y};
        if (my_poly.is_inside(p1)) // point is in physical domain (circle with radius 0.4)
            indicator_function_values[i] = 1; // indicates physical domain
        else
            indicator_function_values[i] = 1e-8; // indicates fictitous domain
        
        ofs_indicator_function_values << x << " " << y << " " << indicator_function_values[i] << std::endl;
    }
    return indicator_function_values;
}
//_______________________________________________
bool contains_boundary (typename dealii::DoFHandler<2>::cell_iterator cell, myPolygon my_poly)
{
    unsigned int vertex_tracker = 0;
    dealii::Point<2> vertex;
    double x,y;
    for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
        x = cell->vertex(vertex_iterator)[0];
        y = cell->vertex(vertex_iterator)[1];
        vertex = {x,y};
        if (my_poly.is_inside(vertex))
            vertex_tracker++;
    }
    if (vertex_tracker == 0 || vertex_tracker == 4)
        return false;
    else
        return true;
}
//_____________________________________________
