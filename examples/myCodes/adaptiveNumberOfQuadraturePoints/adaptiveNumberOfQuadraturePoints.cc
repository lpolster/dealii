#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>

using namespace dealii;


template <int dim>
class adaptiveNumberOfQuadraturePoints
{
public:
    adaptiveNumberOfQuadraturePoints ();
    ~adaptiveNumberOfQuadraturePoints ();

    void run ();

private:

    static bool
    cell_is_in_physical_domain (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_is_in_fictitious_domain (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_is_cut_by_boundary (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_is_child (const typename DoFHandler<dim>::cell_iterator &cell, const typename DoFHandler<dim>::cell_iterator &solution_cell);

    void setup_system ();
    void assemble_system ();
    void solve ();
    void refine_grid ();
    void output_results ();

    void get_indicator_function_values(const std::vector<Point<dim> > &points, std::vector<double> &indicator_function_values, typename DoFHandler<dim>::cell_iterator solution_cell,
                                                                              std::function<bool(double, double, double)> f);
    void set_material_ids(DoFHandler<dim> &dof_handler, std::function<bool(double, double, double)> f);
    std::vector<double> get_normal_vector(typename DoFHandler<dim>::cell_iterator cell, std::function<bool(double, double, double)> f);
    std::vector<Point<dim>> get_boundary_quadrature_points(typename DoFHandler<dim>::cell_iterator cell, std::function<bool(double, double, double)> f);
    dealii::Tensor<1,2,double> get_normal_vector_at_q_point(std::vector<std::vector<double>> normal_vectors_list, unsigned int q_index);
    void write_solution_to_file (Vector<double> solution);
    Quadrature<dim> collect_quadrature(typename DoFHandler<dim>::cell_iterator solution_cell,
                                       const Quadrature<dim>* quadrature_formula);
    Quadrature<dim> collect_quadrature_on_boundary(typename DoFHandler<dim>::cell_iterator solution_cell);
    std::vector<std::vector<double>> collect_normal_vector_on_boundary(typename DoFHandler<dim>::cell_iterator solution_cell);
    Quadrature<dim> map_quadrature_points_and_weights_to_reference_cell (std::vector<Point<dim>> q_points, // quadrature points
                                    std::vector<double> q_weights, std::vector<unsigned int> refinement_level_vec, typename DoFHandler<dim>::cell_iterator cell, std::string filename);



    Triangulation<dim>   triangulation; // triangulation for the solution grid
    Triangulation<dim>   triangulation_adaptiveIntegration; // triangulation for the integration grid

    DoFHandler<dim>      dof_handler;   // dof handler for the solution grid
    DoFHandler<dim>      dof_handler_adaptiveIntegration; // dof handler for the integration grid
    FE_Q<dim>            fe;            // fe for the solution grid
    FE_Q<dim>            fe_adaptiveIntegration; // fe for the integration grid

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;      // vector for the solution (coefficients of shape functions)
    Vector<double>       system_rhs;    // vector for the right hand side

    double threshold = 0.5; // threshold


    enum
    {
        physical_domain_id,
        fictitious_domain_id,
        boundary_domain_id
    };

};

template <int dim>
bool adaptiveNumberOfQuadraturePoints<dim>::cell_is_in_physical_domain (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == physical_domain_id);
}

template <int dim>
bool adaptiveNumberOfQuadraturePoints<dim>::cell_is_in_fictitious_domain (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == fictitious_domain_id);
}

template <int dim>
bool adaptiveNumberOfQuadraturePoints<dim>::cell_is_cut_by_boundary (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == boundary_domain_id);
}

template <int dim>
bool adaptiveNumberOfQuadraturePoints<dim>::cell_is_child (const typename DoFHandler<dim>::cell_iterator &cell, const typename DoFHandler<dim>::cell_iterator &solution_cell)
{
   return (cell->vertex(0)[0] >= solution_cell->vertex(0)[0] && cell->vertex(0)[1] >= solution_cell->vertex(0)[1]
           && cell->vertex(3)[0] <= solution_cell->vertex(3)[0] && cell->vertex(3)[1] <= solution_cell->vertex(3)[1]);
}

bool circle (double x, double y, double radius)
{
    return (x*x)+(y*y) <= radius*radius;
}

bool rectangle (double x, double y, double length)
{
    return (abs(x) <= length && abs(y) <= length);
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::get_indicator_function_values(const std::vector<Point<dim> > &points,
                                                                          std::vector<double> &indicator_function_values,
                                                                          typename DoFHandler<dim>::cell_iterator solution_cell,
                                                                          std::function<bool(double, double, double)> f)
{   double x, y;
    for (unsigned int i=0; i<indicator_function_values.size(); ++i)
    {
        x = points[i][0] * (solution_cell->vertex(1)[0]-solution_cell->vertex(0)[0]) + solution_cell->vertex(0)[0];
        y = points[i][1] * (solution_cell->vertex(2)[1]-solution_cell->vertex(0)[1]) + solution_cell->vertex(0)[1];

        if (f(x, y, threshold)) // point is in physical domain (circle with radius 0.4)
            indicator_function_values[i] = 1; // indicates physical domain
        else
            indicator_function_values[i] = 1e-8; // indicates fictitous domain
    }

}

template <int dim>
adaptiveNumberOfQuadraturePoints<dim>::adaptiveNumberOfQuadraturePoints ()
    :
      dof_handler (triangulation),
      dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration),
      fe (1),
      fe_adaptiveIntegration (1)
{}

template <int dim>
adaptiveNumberOfQuadraturePoints<dim>::~adaptiveNumberOfQuadraturePoints ()
{
    dof_handler.clear ();
    dof_handler_adaptiveIntegration.clear();
}


template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::setup_system ()
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
//                                              ZeroFunction<dim>(),
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

template<int dim>
std::vector<std::vector<double>> adaptiveNumberOfQuadraturePoints<dim>::collect_normal_vector_on_boundary(typename DoFHandler<dim>::cell_iterator solution_cell)
{
    std::vector<std::vector<double>> normal_vector_list;
    std::vector<double> normal_vector;

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_adaptiveIntegration.begin_active(),
            endc = dof_handler_adaptiveIntegration.end();

    for(; cell!=endc; ++cell)

        if (cell_is_child (cell, solution_cell))
        {
            if (cell_is_cut_by_boundary(cell))
            {
               normal_vector = get_normal_vector(cell, rectangle);
               normal_vector_list.push_back(normal_vector);
               normal_vector_list.push_back(normal_vector);

            }
        }
    return normal_vector_list;
}


template <int dim>
Quadrature<dim> adaptiveNumberOfQuadraturePoints<dim>::collect_quadrature_on_boundary(typename DoFHandler<dim>::cell_iterator solution_cell)
{
    std::vector<Point<dim>> boundary_q_points;
    std::vector<Point<dim>> boundary_q_points_list;
    std::vector<double> boundary_q_weights;
    std::vector<unsigned int> refinement_level_vec_boundary;
    unsigned int refinement_level;

//    std::ofstream ofs_boundary_quadrature_points;
//    ofs_boundary_quadrature_points.open ("boundary_quadrature_points", std::ofstream::out | std::ofstream::app);

    typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler_adaptiveIntegration.begin_active(),
                endc = dof_handler_adaptiveIntegration.end();

    for(; cell!=endc; ++cell)

        if (cell_is_child(cell, solution_cell))
        {
            if (cell_is_cut_by_boundary(cell))
            {
                boundary_q_points = get_boundary_quadrature_points(cell,rectangle);

                // maps only to plot
                boundary_q_points[0][0] = boundary_q_points[0][0] * (cell->vertex(1)[0]-cell->vertex(0)[0]) + cell->vertex(0)[0];
                boundary_q_points[0][1] = boundary_q_points[0][1] * (cell->vertex(2)[1]-cell->vertex(0)[1]) + cell->vertex(0)[1];
                boundary_q_points[1][0] = boundary_q_points[1][0] * (cell->vertex(1)[0]-cell->vertex(0)[0]) + cell->vertex(0)[0];
                boundary_q_points[1][1] = boundary_q_points[1][1] * (cell->vertex(2)[1]-cell->vertex(0)[1]) + cell->vertex(0)[1];

                boundary_q_points_list.insert(boundary_q_points_list.end(), boundary_q_points.begin(), boundary_q_points.end());;

                refinement_level = cell->level() - solution_cell->level();

                refinement_level_vec_boundary.insert(refinement_level_vec_boundary.end(),refinement_level);
                refinement_level_vec_boundary.insert(refinement_level_vec_boundary.end(),refinement_level);
                boundary_q_weights.insert(boundary_q_weights.end(), 0.500);
                boundary_q_weights.insert(boundary_q_weights.end(), 0.500);

            }
        }

    return map_quadrature_points_and_weights_to_reference_cell(boundary_q_points_list, boundary_q_weights,
                                                               refinement_level_vec_boundary, solution_cell, "collected_quadrature_on_boundary");
}

template <int dim>
Quadrature<dim> adaptiveNumberOfQuadraturePoints<dim>::collect_quadrature(typename DoFHandler<dim>::cell_iterator solution_cell, const Quadrature<dim>* quadrature_formula)
{

    std::vector<Point<dim>> q_points; // quadrature points
    std::vector<double> q_weights; // quadrature weights
    std::vector<unsigned int> refinement_level_vec; // vector containing the levels of refinement
    unsigned int refinement_level; // refinement level

    // fe values of the adaptive grid
    FEValues<dim> fe_values_temp (fe_adaptiveIntegration, *quadrature_formula, update_quadrature_points);

    // fe values of the solutions grid
    FEValues<dim> fe_values_solution_cell_temp (fe, *quadrature_formula, update_quadrature_points);

    if (cell_is_cut_by_boundary(solution_cell)) // if cell on solution grid is cut by boundary
    {
        typename DoFHandler<dim>::active_cell_iterator
                cell = dof_handler_adaptiveIntegration.begin_active(),
                endc = dof_handler_adaptiveIntegration.end();
        for(; cell!=endc; ++cell) // loop over all cells of the adaptive grid

            // find cells on adaptive grid that are located at the position on the cell in the solution grid
            if (cell_is_child(cell, solution_cell))
            {
                fe_values_temp.reinit(cell); // reinitialize fe values on cell of adaptive grid
                q_points.insert(q_points.end(),fe_values_temp.get_quadrature_points().begin(), // add the quadrature points of this cell to the vector of quadrature points
                                fe_values_temp.get_quadrature_points().end());
                q_weights.insert(q_weights.end(),fe_values_temp.get_quadrature().get_weights().begin(), // add the quadrature weights of this cell to the vector of quadrature weights
                                 fe_values_temp.get_quadrature().get_weights().end());

                refinement_level = cell->level() - solution_cell->level(); // calculate the level of refinement of the current cell relative to the solution grid

                for (unsigned int i = 0; i < fe_values_temp.get_quadrature().size(); i++)
                {
                    refinement_level_vec.insert(refinement_level_vec.end(),refinement_level); // add the refinement level of the current cell to the vector containing the refinement levels
                    refinement_level_vec.insert(refinement_level_vec.end(),refinement_level); // add the refinement level of the current cell to the vector containing the refinement levels
                }
            }

    } // endif

    else // if cell on solution grid is not cur by the boundary
    {
        fe_values_solution_cell_temp.reinit(solution_cell); // reinitialize fe values on the solution cell

        q_points.insert(q_points.end(),fe_values_solution_cell_temp.get_quadrature_points().begin(), // add the quadrature points of this cellto the vector of quadrature points
                        fe_values_solution_cell_temp.get_quadrature_points().end());
        q_weights.insert(q_weights.end(),fe_values_solution_cell_temp.get_quadrature().get_weights().begin(), // add the quadrature weights of this cell to the vector of quadrature weights
                         fe_values_solution_cell_temp.get_quadrature().get_weights().end());

        for (unsigned int i = 0; i < fe_values_solution_cell_temp.get_quadrature().size(); i++)
        {
            refinement_level_vec.insert(refinement_level_vec.end(),0);    // add the refinement level of the current cell to the vector containing the refinement levels
            refinement_level_vec.insert(refinement_level_vec.end(),0);    // add the refinement level of the current cell to the vector containing the refinement levels

        } //endfor

    } // endelse

    return  map_quadrature_points_and_weights_to_reference_cell(q_points, q_weights, refinement_level_vec, solution_cell, "collected_quadrature");
}

template <int dim>
Quadrature<dim> adaptiveNumberOfQuadraturePoints<dim>::map_quadrature_points_and_weights_to_reference_cell (std::vector<Point<dim>> q_points, // quadrature points
                                std::vector<double> q_weights, std::vector<unsigned int> refinement_level_vec, typename DoFHandler<dim>::cell_iterator cell, std::string filename)
{

    std::ofstream ofs_quadrature_points;

    ofs_quadrature_points.open (filename, std::ofstream::out | std::ofstream::app);

    for (unsigned int i = 0; i<q_points.size(); ++i) // loop over all quadrature points
    {
    q_weights[i] = q_weights[i]/pow(4,refinement_level_vec[i]); // caluclate weight of quadrature point such that the weights add up to 1

    ofs_quadrature_points<<q_points[i][0]<<" "<< q_points[i][1]<<std::endl;
    std::cout<<" "<<q_weights[i]<<std::endl;

    q_points[i][0] = (q_points[i][0] - cell->vertex(0)[0]) / (cell->vertex(1)[0] - cell->vertex(0)[0]); // calculate x location of quadrature point on reference cell
    q_points[i][1] = (q_points[i][1] - cell->vertex(0)[1]) / (cell->vertex(2)[1] - cell->vertex(0)[1]); // calculate y location of quadrature point on reference cell
    }

    ofs_quadrature_points.close();

    return Quadrature<dim>(q_points, q_weights);
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::assemble_system ()
{
    const QGauss<dim>  quadrature_formula(2);                   // (p+1) quadrature points (p = polynomial degree)
    Quadrature<dim> collected_quadrature;                       // the quadrature rule
    Quadrature<dim> collected_quadrature_on_boundary;           // quadrature rule on boundary
    std::vector<std::vector<double>> normal_vectors_list;

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;      // degrees of freedom per cell on solution grid
    const float beta_h = 2.0/0.0625;                                    // beta divided by h, 2.0/0.0625
    const float dirichlet_boundary_value = 0.000;               // value for Dirichlet boundary condition
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell); // initialize cell matrix
    Vector<double>       cell_rhs (dofs_per_cell);              // initialize cell rhs
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
            solution_cell = dof_handler.begin_active(),         // iterator to first active cell of the solution grid
            solution_endc = dof_handler.end();                  // iterator to the one past last active cell of the solution grid

    for (; solution_cell!=solution_endc; ++solution_cell) // loop over all cells on solution grid
    {
        cell_matrix = 0;
        cell_rhs = 0;

        std::cout<<"Calculating contribution of cell "<<solution_cell<<std::endl;

        collected_quadrature = collect_quadrature(solution_cell, &quadrature_formula); // get quadrature on current cell
        FEValues<dim> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);

        fe_values.reinit(solution_cell);                        // reinitialize fe values on current cells

        unsigned int n_q_points = collected_quadrature.size(); // number of quadrature points

        std::vector<double> indicator_function_values(n_q_points);
        get_indicator_function_values(fe_values.get_quadrature().get_points(), indicator_function_values, solution_cell,
                                                                   rectangle);

        //if (cell_is_in_fictitious_domain(solution_cell) == false)
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index){      // loop over all quadrature points

            //std::cout<<indicator_function_values[q_index]<<std::endl;

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
            normal_vectors_list = collect_normal_vector_on_boundary(solution_cell);
            collected_quadrature_on_boundary = collect_quadrature_on_boundary(solution_cell);
            FEValues<dim> fe_values_on_boundary(fe, collected_quadrature_on_boundary, update_quadrature_points |  update_gradients | update_JxW_values | update_jacobians  |  update_values);
            fe_values_on_boundary.reinit(solution_cell);
            unsigned int   n_q_points_boundary    = collected_quadrature_on_boundary.size();
            dealii::Tensor<1,2,double> normal_vector;
//            std::cout<<"Number of normal vectors: "<<normal_vectors_list.size()<<std::endl;
//            std::cout<<"Number of quadrature points: "<<n_q_points_boundary<<std::endl;

//        // Nitsche Method

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


template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::solve ()
{
    std::cout<<"Solve....."<<std::endl;
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);

    std::cout<<"Distribute constraints...."<<std::endl;

    constraints.distribute (solution);

    std::ofstream ofs;
    ofs.open ("solution", std::ofstream::out);
    solution.print(ofs, 2, false, true);
    ofs.close();
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not

    typename DoFHandler<dim>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler_adaptiveIntegration.begin_active(), // the first active cell
            endc = dof_handler_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (cell_is_cut_by_boundary(cell))
            cell -> set_refine_flag();
    }

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::output_results ()
{
    std::string filename_integrationGrid = "integration_grid";
    filename_integrationGrid += ".eps";
    std::ofstream output_integrationGrid (filename_integrationGrid.c_str());
    GridOut grid_out_integrationGrid;
    grid_out_integrationGrid.write_eps (triangulation_adaptiveIntegration, output_integrationGrid);

    std::string filename_solutionGrid = "solution_grid";
    filename_solutionGrid += ".eps";
    std::ofstream output_solutionGrid (filename_solutionGrid.c_str());
    GridOut grid_out_solutionGrid;
    grid_out_solutionGrid.write_eps (triangulation, output_solutionGrid);

    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();

    std::ofstream output ("solution.gpl");
    data_out.write_gnuplot (output);

    std::ofstream output2 ("solution.vtk");
    data_out.write_vtk (output2);
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::set_material_ids(DoFHandler<dim> &dof_handler, std::function<bool(double, double, double)> f)
{
    std::vector<bool> vec0000 = {0, 0, 0, 0};
    std::vector<bool> vec1111 = {1, 1, 1, 1};

    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
        std::vector<bool> vertex_tracker (4);
        for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
            if (f(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
                vertex_tracker[vertex_iterator] = 1;
            }
            else
                vertex_tracker[vertex_iterator] = 0;
        }
        if (vertex_tracker == vec0000){
            cell->set_material_id (fictitious_domain_id);
            // std::cout<<cell<<" is in fictitious domain."<<std::endl;
        }
        else if (vertex_tracker == vec1111){
            cell->set_material_id (physical_domain_id);
            // std::cout<<cell<<" is in physical domain."<<std::endl;
        }
        else{
            cell->set_material_id(boundary_domain_id);
            //  std::cout<<cell<<" contains boundary."<<std::endl;
        }
    }
}

template <int dim>
std::vector<Point<dim>> adaptiveNumberOfQuadraturePoints<dim>::get_boundary_quadrature_points(typename DoFHandler<dim>::cell_iterator cell, std::function<bool(double, double, double)> f)
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

    std::vector<Point<dim>> q_points_boundary;

   // std::cout<<cell<<std::endl;
    std::vector<bool> vertex_tracker (4);
    for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
        if (f(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
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

template <int dim>
dealii::Tensor<1,2,double> adaptiveNumberOfQuadraturePoints<dim>::get_normal_vector_at_q_point(std::vector<std::vector<double>> normal_vectors_list, unsigned int q_index)
{
     dealii::Tensor<1,2,double>normal_vector;
    normal_vector[0] = normal_vectors_list[q_index][0];
    normal_vector[1] = normal_vectors_list[q_index][1];
    return normal_vector;
}

template <int dim>
std::vector<double> adaptiveNumberOfQuadraturePoints<dim>::get_normal_vector(typename DoFHandler<dim>::cell_iterator cell, std::function<bool(double, double, double)> f)
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
        if (f(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1], threshold)){
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

    else
        return {0,0};
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::run ()
{
    GridGenerator::hyper_cube (triangulation, -1.0, 1.0);
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -1.0, 1.0);
    triangulation_adaptiveIntegration.refine_global (5);
    triangulation.refine_global (5);

    set_material_ids(dof_handler_adaptiveIntegration, rectangle);
    set_material_ids(dof_handler, rectangle);

    const unsigned int refinement_cycles = 2;
    for (unsigned int i = 0; i < refinement_cycles; i++)
    {
        refine_grid ();
        set_material_ids(dof_handler_adaptiveIntegration, rectangle);
    }

    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
}



int main ()
{
    try
    {
        deallog.depth_console (0);

        adaptiveNumberOfQuadraturePoints<2> laplace_problem_2d;
        laplace_problem_2d.run ();
    }

    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;

        return 1;
    }

    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
