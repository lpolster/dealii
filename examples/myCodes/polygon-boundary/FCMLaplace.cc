#include "FCMLaplace.h"
#include "find_cells.h"

using namespace dealii;

FCMLaplace::FCMLaplace ()
    :
      dof_handler (triangulation),
      dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration),
      fe (1),
      fe_adaptiveIntegration (1)
{m_boundaryFunction=&FCMLaplace::rectangle;}
//____________________________________________________________________________________________________________________________________


FCMLaplace::~FCMLaplace ()
{
    dof_handler.clear ();
    dof_handler_adaptiveIntegration.clear();
}
//____________________________________________________________________________________________________________________________________


bool FCMLaplace::cell_is_in_physical_domain (const typename Triangulation<2>::cell_iterator &cell)
{
    return (cell->material_id() == physical_domain_id);
}
//____________________________________________________________________________________________________________________________________


bool FCMLaplace::cell_is_in_fictitious_domain (const typename Triangulation<2>::cell_iterator &cell)
{
    return (cell->material_id() == fictitious_domain_id);
}
//____________________________________________________________________________________________________________________________________

bool FCMLaplace::cell_is_cut_by_boundary (const typename Triangulation<2>::cell_iterator &cell)
{
    return (cell->material_id() == boundary_domain_id);
}
//____________________________________________________________________________________________________________________________________


bool FCMLaplace::cell_is_child (const typename Triangulation<2>::cell_iterator &cell, const typename Triangulation<2>::cell_iterator &solution_cell)
{
    return (cell->vertex(0)[0] >= solution_cell->vertex(0)[0] && cell->vertex(0)[1] >= solution_cell->vertex(0)[1]
            && cell->vertex(3)[0] <= solution_cell->vertex(3)[0] && cell->vertex(3)[1] <= solution_cell->vertex(3)[1]);
}
//____________________________________________________________________________________________________________________________________

bool FCMLaplace::circle (double x, double y, const double radius)
{
    return ((x*x)+(y*y) <= radius*radius);
}
//____________________________________________________________________________________________________________________________________

bool FCMLaplace::rectangle (double x, double y, const double length)
{
    return (std::abs(x) <= length && std::abs(y) <= length);
}
//____________________________________________________________________________________________________________________________________

void FCMLaplace::get_indicator_function_values(const std::vector<Point<2> > &points,
                                               std::vector<double> &indicator_function_values,
                                               typename DoFHandler<2>::cell_iterator solution_cell,
                                               boundary_function f)
{
    std::ofstream ofs_indicator_function_values;
    ofs_indicator_function_values.open ("indicator_function_values", std::ofstream::out | std::ofstream::app);

    double x, y;
    for (unsigned int i=0; i<indicator_function_values.size(); ++i)
    {
        x = points[i][0] * (solution_cell->vertex(1)[0]-solution_cell->vertex(0)[0]) + solution_cell->vertex(0)[0];
        y = points[i][1] * (solution_cell->vertex(2)[1]-solution_cell->vertex(0)[1]) + solution_cell->vertex(0)[1];

        if ((this->*f)(x, y, threshold)) // point is in physical domain (circle with radius 0.4)
            indicator_function_values[i] = 1; // indicates physical domain
        else
            indicator_function_values[i] = 1e-8; // indicates fictitous domain

        ofs_indicator_function_values << x << " " << y << " " << indicator_function_values[i] << std::endl;
    }
}
//____________________________________________________________________________________________________________________________________

void FCMLaplace::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    dof_handler_adaptiveIntegration.distribute_dofs(fe_adaptiveIntegration);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
//    VectorTools::interpolate_boundary_values (dof_handler,
//                                              0,
//                                              ZeroFunction<2>(),
//                                              constraints);
    constraints.close ();

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(c_sparsity);

    system_matrix.reinit (sparsity_pattern);
}

std::vector<std::vector<double>> FCMLaplace::collect_normal_vector_on_boundary(typename DoFHandler<2>::cell_iterator solution_cell)
{
    std::vector<std::vector<double>> normal_vector_list;
    std::vector<double> normal_vector;

    typename DoFHandler<2>::active_cell_iterator
            cell = dof_handler_adaptiveIntegration.begin_active(),
            endc = dof_handler_adaptiveIntegration.end();

    for(; cell!=endc; ++cell)

        if (cell_is_child (cell, solution_cell))
        {
            if (cell_is_cut_by_boundary(cell))
            {
                normal_vector = get_normal_vector(cell, m_boundaryFunction);
                normal_vector_list.push_back(normal_vector);
                normal_vector_list.push_back(normal_vector);

            }
        }
    return normal_vector_list;
}
//____________________________________________________________________________________________________________________________________

Quadrature<2> FCMLaplace::collect_quadratures_on_boundary(typename Triangulation<2>::cell_iterator cell)
{
    std::vector<Point<2> > q_points;
    std::vector<double> q_weights;

    if(cell->active() && cell_is_cut_by_boundary(cell))
    {
        std::vector<Point<2> > q_points_temp;
        std::vector<double> q_weights_temp (2);
        // not refined, return quadrature rule
        q_points_temp = get_boundary_quadrature_points(cell,m_boundaryFunction);
        q_weights_temp = {0.500, 0.500};
        return dealii::Quadrature<2>(q_points_temp, q_weights_temp);
    }

    // get collected quadratures of each children and merge them
    for(unsigned int child = 0;
        child < dealii::GeometryInfo<2>::max_children_per_cell;
        ++child)
    {
        // get child
        typename dealii::Triangulation<2>::cell_iterator child_cell =
                cell->child(child);
        // collect sub-quadratures there
        if(child_cell->active() && cell_is_cut_by_boundary(child_cell)){
            dealii::Quadrature<2> childs_collected_quadratures =
                    collect_quadratures_on_boundary(child_cell);
            // project to current cell
            dealii::Quadrature<2> child_quadrature =
                    dealii::QProjector<2>::project_to_child(childs_collected_quadratures, child);
            // collect resulting quadrature
            q_points.insert(q_points.end(),
                            child_quadrature.get_points().begin(),
                            child_quadrature.get_points().end());
            q_weights.insert(q_weights.end(),
                             child_quadrature.get_weights().begin(),
                             child_quadrature.get_weights().end());}
    }

    return dealii::Quadrature<2>(q_points, q_weights);
}
//____________________________________________________________________________________________________________________________________

Quadrature<2> FCMLaplace::collect_quadratures(typename dealii::Triangulation<2>::cell_iterator cell,
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
//____________________________________________________________________________________________________________________________________

void FCMLaplace::plot_in_global_coordinates (std::vector<Point<2>> q_points,
                                             DoFHandler<2>::cell_iterator cell, std::string filename)
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
//____________________________________________________________________________________________________________________________________


void FCMLaplace::assemble_system ()
{
    const QGauss<2>  quadrature_formula(2);                   // (p+1) quadrature points (p = polynomial degree)
    Quadrature<2> collected_quadrature;                       // the quadrature rule
    Quadrature<2> collected_quadrature_on_boundary;           // quadrature rule on boundary
    std::vector<std::vector<double>> normal_vectors_list;

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;      // degrees of freedom per cell on solution grid
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell); // initialize cell matrix
    Vector<double>       cell_rhs (dofs_per_cell);              // initialize cell rhs
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<2>::active_cell_iterator
            solution_cell = dof_handler.begin_active(),         // iterator to first active cell of the solution grid
            solution_endc = dof_handler.end();                  // iterator to the one past last active cell of the solution grid

    for (; solution_cell!=solution_endc; ++solution_cell) // loop over all cells on solution grid
    {
        cell_matrix = 0;
        cell_rhs = 0;

        //       std::cout<<"Calculating contribution of cell "<<solution_cell<<std::endl;

        collected_quadrature = collect_quadratures(topological_equivalent(solution_cell, triangulation_adaptiveIntegration), &quadrature_formula);

        FEValues<2> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);

        fe_values.reinit(solution_cell);                        // reinitialize fe values on current cells

        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), solution_cell, "collected_quadrature");

        unsigned int n_q_points = collected_quadrature.size(); // number of quadrature points

        std::vector<double> indicator_function_values(n_q_points);

        get_indicator_function_values(fe_values.get_quadrature().get_points(), indicator_function_values, solution_cell,
                                      m_boundaryFunction);

        //if (cell_is_in_fictitious_domain(solution_cell) == false)
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index){      // loop over all quadrature points
            for (unsigned int i=0; i<dofs_per_cell; ++i)  {                   // loop over degrees of freedom
                for (unsigned int j=0; j<dofs_per_cell; ++j)  {              // loop over degrees of freedom

                    cell_matrix(i,j) += (fe_values.shape_grad(i,q_index) * //
                                         fe_values.shape_grad(j,q_index)  * indicator_function_values[q_index] *
                                         fe_values.JxW(q_index));
                }
                cell_rhs(i) += (fe_values.shape_value(i,q_index) * indicator_function_values[q_index] * // the cell rhs
                                1.0 *
                                fe_values.JxW(q_index));
            }
        }


        if (cell_is_cut_by_boundary(solution_cell))
        {
            std::cout<<"Cell "<<solution_cell<<" is cut by boundary."<<std::endl;
            normal_vectors_list = collect_normal_vector_on_boundary(solution_cell);
            collected_quadrature_on_boundary = collect_quadratures_on_boundary(topological_equivalent(solution_cell, triangulation_adaptiveIntegration));

            FEValues<2> fe_values_on_boundary(fe, collected_quadrature_on_boundary, update_quadrature_points |  update_gradients | update_JxW_values | update_jacobians  |  update_values);
            fe_values_on_boundary.reinit(solution_cell);

            plot_in_global_coordinates(fe_values_on_boundary.get_quadrature().get_points(), solution_cell, "collected_quadrature_on_boundary");

            unsigned int   n_q_points_boundary    = collected_quadrature_on_boundary.size();
            dealii::Tensor<1,2,double> normal_vector;

            // Nitsche Method

            for (unsigned int q_index=0; q_index<n_q_points_boundary; ++q_index){      // loop over all quadrature points
                for (unsigned int i=0; i<dofs_per_cell; ++i)  {                   // loop over degrees of freedom
                    for (unsigned int j=0; j<dofs_per_cell; ++j)  {              // loop over degrees of freedom

                        normal_vector = get_normal_vector_at_q_point(normal_vectors_list, q_index);

                        cell_matrix(i,j) -= (fe_values_on_boundary.shape_value(i,q_index) * //
                                             fe_values_on_boundary.shape_grad(j,q_index) * normal_vector *
                                             fe_values_on_boundary.JxW(q_index));

                        cell_matrix(i,j) -= (fe_values_on_boundary.shape_value(j,q_index) * //
                                             fe_values_on_boundary.shape_grad(i,q_index) * normal_vector *
                                             fe_values_on_boundary.JxW(q_index));

                        cell_matrix(i,j) +=  beta_h * (fe_values_on_boundary.shape_value(i,q_index) * //
                                                       fe_values_on_boundary.shape_value(j,q_index) *
                                                       fe_values_on_boundary.JxW(q_index));
                    }
                    cell_rhs(i) -= (dirichlet_boundary_value * fe_values_on_boundary.shape_grad(i,q_index) * normal_vector *// the cell rhs
                                    fe_values_on_boundary.JxW(q_index));
                    cell_rhs(i) +=  (beta_h * fe_values_on_boundary.shape_value(i,q_index) * //
                                     dirichlet_boundary_value *
                                     fe_values_on_boundary.JxW(q_index));
                } // endfor

            }
        } // endif

        solution_cell-> get_dof_indices (local_dof_indices);  // return the global indices of the dof located on this object

        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
    }
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::solve ()
{
    //    std::cout<<"Solve....."<<std::endl;
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);

    //    std::cout<<"Distribute constraints...."<<std::endl;
    constraints.distribute (solution);

    std::ofstream ofs;
    ofs.open ("solution", std::ofstream::out);
    solution.print(ofs, 2, false, true);
    ofs.close();
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not

    typename DoFHandler<2>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler_adaptiveIntegration.begin_active(), // the first active cell
            endc = dof_handler_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (cell_is_cut_by_boundary(cell))
            cell -> set_refine_flag();
    }

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::output_results ()
{
    std::string filename_integrationGrid = "integration_grid_FCM";
    filename_integrationGrid += ".eps";
    std::ofstream output_integrationGrid (filename_integrationGrid.c_str());
    GridOut grid_out_integrationGrid;
    grid_out_integrationGrid.write_eps (triangulation_adaptiveIntegration, output_integrationGrid);

    std::string filename_solutionGrid = "solution_grid_FCM";
    filename_solutionGrid += ".eps";
    std::ofstream output_solutionGrid (filename_solutionGrid.c_str());
    GridOut grid_out_solutionGrid;
    grid_out_solutionGrid.write_eps (triangulation, output_solutionGrid);

    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution_FCM");
    data_out.build_patches (3); // interpolation for visualization

    std::ofstream output ("solution_FCM.gpl");
    data_out.write_gnuplot (output);

    std::ofstream output2 ("solution_FCM.vtk");
    data_out.write_vtk (output2);
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::set_material_ids(DoFHandler<2> &dof_handler, boundary_function f)
{
    std::vector<bool> vec0000 = {0, 0, 0, 0};
    std::vector<bool> vec1111 = {1, 1, 1, 1};

    for (typename Triangulation<2>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
        std::vector<bool> vertex_tracker (4);
        for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
            if ((this->*f)(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
                vertex_tracker[vertex_iterator] = 1;
            }
            else
                vertex_tracker[vertex_iterator] = 0;
        }
        if (vertex_tracker == vec0000){
            cell->set_material_id (fictitious_domain_id);
//            std::cout<<cell<<" is in fictitious domain."<<std::endl;
        }
        else if (vertex_tracker == vec1111){
            cell->set_material_id (physical_domain_id);
//            std::cout<<cell<<" is in physical domain."<<std::endl;
        }
        else{
            cell->set_material_id(boundary_domain_id);
//            std::cout<<cell<<" contains boundary."<<std::endl;
        }
    }
}
//____________________________________________________________________________________________________________________________________


std::vector<Point<2>> FCMLaplace::get_boundary_quadrature_points(typename dealii::Triangulation<2>::cell_iterator cell, boundary_function f)
{
    std::vector<bool> vec0001 = {0, 0, 0, 1};
    std::vector<bool> vec0010 = {0, 0, 1, 0};
    std::vector<bool> vec0100 = {0, 1, 0, 0};
    std::vector<bool> vec1000 = {1, 0, 0, 0};
    std::vector<bool> vec1100 = {1, 1, 0, 0};
    std::vector<bool> vec0011 = {0, 0, 1, 1};
    std::vector<bool> vec1110 = {1, 1, 1, 0};
    std::vector<bool> vec1101 = {1, 1, 0, 1};
    std::vector<bool> vec1011 = {1, 0, 1, 1};
    std::vector<bool> vec0111 = {0, 1, 1, 1};
    std::vector<bool> vec1010 = {1, 0, 1, 0};
    std::vector<bool> vec0101 = {0, 1, 0, 1};

    std::vector<Point<2>> q_points_boundary;

    // std::cout<<cell<<std::endl;
    std::vector<bool> vertex_tracker (4);
    for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
        if ((this->*f)(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
            vertex_tracker[vertex_iterator] = 1;
        }
        else
            vertex_tracker[vertex_iterator] = 0;
        // std::cout<<vertex_tracker[vertex_iterator]<<" ";
    }
    //std::cout<<""<<std::endl;

    if(vertex_tracker == vec0001 || vertex_tracker == vec1110){
        q_points_boundary.insert(q_points_boundary.end(),{0.606, 0.894});
        q_points_boundary.insert(q_points_boundary.end(),{0.894, 0.606});
    }


    else if(vertex_tracker == vec0111 || vertex_tracker == vec1000){
        q_points_boundary.insert(q_points_boundary.end(),{0.106, 0.394});
        q_points_boundary.insert(q_points_boundary.end(),{0.394, 0.106});
    }

    else if (vertex_tracker == vec1011 || vertex_tracker == vec0100){
        q_points_boundary.insert(q_points_boundary.end(),{0.606, 0.106});
        q_points_boundary.insert(q_points_boundary.end(),{0.894, 0.394});
    }

    else if (vertex_tracker == vec0010 || vertex_tracker == vec1101){
        q_points_boundary.insert(q_points_boundary.end(),{0.106, 0.606});
        q_points_boundary.insert(q_points_boundary.end(),{0.394, 0.894});
    }

    else if (vertex_tracker == vec0011 || vertex_tracker == vec1100){
        q_points_boundary.insert(q_points_boundary.end(),{0.211, 0.500});
        q_points_boundary.insert(q_points_boundary.end(),{0.789, 0.500});
    }

    else if (vertex_tracker == vec1010 || vertex_tracker == vec0101){
        q_points_boundary.insert(q_points_boundary.end(),{0.500, 0.211});
        q_points_boundary.insert(q_points_boundary.end(),{0.500, 0.789});
    }

    //std::cout<<q_points_boundary[0]<<std::endl<<q_points_boundary[1]<<std::endl;

    return q_points_boundary;
}
//____________________________________________________________________________________________________________________________________


dealii::Tensor<1,2,double> FCMLaplace::get_normal_vector_at_q_point(std::vector<std::vector<double>> normal_vectors_list, unsigned int q_index)
{
    dealii::Tensor<1,2,double>normal_vector;
    normal_vector[0] = normal_vectors_list[q_index][0];
    normal_vector[1] = normal_vectors_list[q_index][1];
    return normal_vector;
}
//____________________________________________________________________________________________________________________________________


std::vector<double> FCMLaplace::get_normal_vector(typename Triangulation<2>::cell_iterator cell, boundary_function f)
{
    std::vector<bool> vec0001 = {0, 0, 0, 1};
    std::vector<bool> vec0010 = {0, 0, 1, 0};
    std::vector<bool> vec0100 = {0, 1, 0, 0};
    std::vector<bool> vec1000 = {1, 0, 0, 0};
    std::vector<bool> vec1100 = {1, 1, 0, 0};
    std::vector<bool> vec0011 = {0, 0, 1, 1};
    std::vector<bool> vec1110 = {1, 1, 1, 0};
    std::vector<bool> vec1101 = {1, 1, 0, 1};
    std::vector<bool> vec1011 = {1, 0, 1, 1};
    std::vector<bool> vec0111 = {0, 1, 1, 1};
    std::vector<bool> vec1010 = {1, 0, 1, 0};
    std::vector<bool> vec0101 = {0, 1, 0, 1};

    //std::cout<<cell<<std::endl;
    std::vector<bool> vertex_tracker (4);
    for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
        if ((this->*f)(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
            vertex_tracker[vertex_iterator] = 1;
        }
        else
            vertex_tracker[vertex_iterator] = 0;
        // std::cout<<vertex_tracker[vertex_iterator]<<" ";
    }
    //std::cout<<""<<std::endl;

    if(vertex_tracker == vec0001 || vertex_tracker == vec0111)
        return {-1/sqrt(2), -1/sqrt(2)};
    else if (vertex_tracker == vec1011 || vertex_tracker == vec0010)
        return {1/sqrt(2), -1/sqrt(2)};

    else if (vertex_tracker == vec0100|| vertex_tracker == vec1101)
        return {-1/sqrt(2), 1/sqrt(2)};

    else if (vertex_tracker == vec1000|| vertex_tracker == vec1110)
        return {1/sqrt(2), 1/sqrt(2)};

    else if (vertex_tracker == vec0011)
        return {0, -1};

    else if (vertex_tracker == vec1100)
        return {0, 1};

    else if (vertex_tracker == vec1010)
        return {1, 0};

    else if (vertex_tracker == vec0101)
        return {-1, 0};
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::output_grid(const Triangulation<2>& tria,
                             std::string name,
                             const unsigned int nr)
{
    GridOut grid_out;
    std::stringstream filename;
    filename << name << "-" << nr << ".svg";
    std::ofstream out(filename.str());
    grid_out.write_svg(tria, out);
}
//____________________________________________________________________________________________________________________________________


void FCMLaplace::run ()
{
    GridGenerator::hyper_cube (triangulation, triangulation_begin, triangulation_end);
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, triangulation_begin, triangulation_end);
    triangulation_adaptiveIntegration.refine_global (global_refinement_level);
    triangulation.refine_global (global_refinement_level);

    FCMLaplace::set_material_ids(dof_handler_adaptiveIntegration, m_boundaryFunction);
    FCMLaplace::set_material_ids(dof_handler, m_boundaryFunction);

    output_grid(triangulation, "solutionGrid", 0);

    for (unsigned int i = 0; i < refinement_cycles; i++)
    {
        FCMLaplace::refine_grid ();
        output_grid(triangulation_adaptiveIntegration, "adaptiveGrid", i);
        FCMLaplace::set_material_ids(dof_handler_adaptiveIntegration, m_boundaryFunction);
    }

    FCMLaplace::setup_system ();
    FCMLaplace::assemble_system ();
    FCMLaplace::solve ();
    FCMLaplace::output_results ();
}
