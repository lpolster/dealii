const unsigned int global_refinement_level = 3;              // the level of global refininement (solution grid)
const float beta_h = 20.0/(1.0/ global_refinement_level);    // beta divided by h, 2.0/0.0625
const float dirichlet_boundary_value = 0.0000;               // the Duruchlet boundary condition
const unsigned int refinement_cycles = 2;                    // the number of cycles of adaptive refinement
const dealii::MappingQ1<2> mapping;

//___________________________________________
dealii::Quadrature<2> collect_quadratures(const typename dealii::Triangulation<2>::cell_iterator cell,
                                          const dealii::Quadrature<2>* base_quadrature)
{
    if(cell->active()) // if the cell is flagged active
    {
        // cell is not further refined, return copy of base quadrature
        return *base_quadrature;
    }
    
    // get collected quadratures of each child and merge them
    std::vector<dealii::Point<2> > q_points;
    std::vector<double> q_weights;
    for(unsigned int child = 0;
        child < dealii::GeometryInfo<2>::max_children_per_cell;
        ++child)                                        // loop over all child cells
    {
        // get child
        typename dealii::Triangulation<2>::cell_iterator child_cell =
        cell->child(child);
        // collect sub-quadratures
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
dealii::Quadrature<2> collect_quadratures_on_boundary_segment(const myPolygon::segment my_segment, const typename dealii::Triangulation<2>::cell_iterator cell)
{
    std::vector<dealii::Point<2> > q_points;
    dealii::Point<2> q_point_on_real_cell;
    dealii::Point<2> q_point_on_unit_cell;
    
    for (unsigned int i = 0; i < my_segment.q_points.size(); ++i)
    {
        q_point_on_real_cell =  my_segment.q_points[i];
        q_point_on_unit_cell = mapping.transform_real_to_unit_cell (cell,q_point_on_real_cell);
        q_points.insert(q_points.end(),q_point_on_unit_cell );
    }
    return dealii::Quadrature<2>(q_points, my_segment.q_weights);
}

//______________________________________
void plot_in_global_coordinates (const std::vector<dealii::Point<2>> q_points,
                                 const dealii::DoFHandler<2>::cell_iterator cell, const std::string filename)
{
    std::ofstream ofs_quadrature_points;
    
    ofs_quadrature_points.open (filename, std::ofstream::out | std::ofstream::app);
    
    for (unsigned int i = 0; i<q_points.size(); ++i) // loop over all quadrature points
    {
        ofs_quadrature_points<<mapping.transform_unit_to_real_cell (cell,q_points[i])<<std::endl;
    }
    
    ofs_quadrature_points.close();
}

//___________________________________________

std::vector<double> get_indicator_function_values(const std::vector<dealii::Point<2> > points,
                                                  const typename dealii::DoFHandler<2>::cell_iterator cell, myPolygon my_poly)
{
    std::ofstream ofs_indicator_function_values;
    ofs_indicator_function_values.open ("indicator_function_values", std::ofstream::out | std::ofstream::app);
    
    dealii::Point<2> q_point_in_global_coordinates;
    std::vector<double> indicator_function_values (points.size());
    for (unsigned int i=0; i<indicator_function_values.size(); ++i)
    {
        q_point_in_global_coordinates = mapping.transform_unit_to_real_cell (cell, points[i]);

        if (my_poly.is_inside(q_point_in_global_coordinates)) // point is in physical domain (circle with radius 0.4)
            indicator_function_values[i] = 1; // indicates physical domain
        else
            indicator_function_values[i] = 1e-8; // indicates fictitous domain
        
        ofs_indicator_function_values << q_point_in_global_coordinates << " " << indicator_function_values[i] << std::endl;
    }
    return indicator_function_values;
}
//_______________________________________________
bool contains_boundary (const typename dealii::DoFHandler<2>::cell_iterator cell, myPolygon my_poly)
{
    unsigned int vertex_tracker = 0;
    for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
        if (my_poly.is_inside(cell->vertex(vertex_iterator)))
            vertex_tracker++;
    }
    if (vertex_tracker == 0 || vertex_tracker == 4)
        return false;
    else
        return true;
}
//_____________________________________________
