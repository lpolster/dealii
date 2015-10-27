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

    void setup_system ();
    void assemble_system (std::function<double(double, double)> f);
    void solve ();
    void refine_grid ();
    void output_results ();
    void set_material_ids(DoFHandler<dim> &dof_handler, std::function<double(double, double)> f);
    void write_solution_to_file (Vector<double> solution);
    void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             std::function<double(double, double)> f,
                             const unsigned int              component = 0) const;
    Quadrature<dim> collect_quadrature(typename DoFHandler<dim>::cell_iterator solution_cell,
                                       const Quadrature<dim>* quadrature_formula);

    Triangulation<dim>   triangulation; // triangulation for the solution grid
    Triangulation<dim>   triangulation_adaptiveIntegration; // triangulation for the integration grid

    DoFHandler<dim>      dof_handler; // dof handler for the solution grid
    DoFHandler<dim>      dof_handler_adaptiveIntegration; // dof handler for the integration grid
    FE_Q<dim>            fe; // fe for the solution grid
    FE_Q<dim>            fe_adaptiveIntegration; // fe for the integration grid

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;

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

double circle (double x, double y)
{
    return (x*x)+(y*y);
}


template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   std::function<double(double, double)> f,
                                   const unsigned int             component) const
{
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
    {
        if (f(points[i][0], points[i][1])  <= 1)
            values[i] = 1; // indicates physical domain
        else
            values[i] = 1e-8; // indicates fictitous domain
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
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<dim>(),
                                              constraints);
    constraints.close ();

    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(c_sparsity);

    system_matrix.reinit (sparsity_pattern);
}

template <int dim>
Quadrature<dim> adaptiveNumberOfQuadraturePoints<dim>::collect_quadrature(typename DoFHandler<dim>::cell_iterator solution_cell, const Quadrature<dim>* quadrature_formula)
{
    typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler_adaptiveIntegration.begin_active(),
    endc = dof_handler_adaptiveIntegration.end();
    std::vector<Point<dim>> q_points;
    std::vector<double> q_weights;
    double JxW;
    double jacobian;
    double temp;
    
    FEValues<dim> fe_values_temp (fe_adaptiveIntegration, *quadrature_formula, update_gradients | update_quadrature_points  |  update_JxW_values | update_jacobians);
    
    
    FEValues<dim> fe_values_solution_cell_temp (fe, *quadrature_formula, update_gradients | update_quadrature_points  |  update_JxW_values | update_jacobians);
    
    if (cell_is_cut_by_boundary(solution_cell))
    {
        for(; cell!=endc; ++cell)

            if (cell->vertex(0)[0] >= solution_cell->vertex(0)[0] && cell->vertex(0)[1] >= solution_cell->vertex(0)[1]  && cell->vertex(3)[0] <= solution_cell->vertex(3)[0] && cell->vertex(3)[1] <= solution_cell->vertex(3)[1] )
            {   fe_values_temp.reinit(cell);
                
                q_points.insert(q_points.end(),fe_values_temp.get_quadrature_points().begin(),
                                fe_values_temp.get_quadrature_points().end());
            }
        for (unsigned int i = 0; i<q_points.size(); ++i)
            q_weights.insert(q_weights.end(),1.0/q_points.size());
        
        return Quadrature<dim>(q_points, q_weights);
        
    }
    else
    {   fe_values_solution_cell_temp.reinit(solution_cell);
        
        q_points.insert(q_points.end(),fe_values_solution_cell_temp.get_quadrature_points().begin(),
                        fe_values_solution_cell_temp.get_quadrature_points().end());
        q_weights.insert(q_weights.end(),fe_values_solution_cell_temp.get_quadrature().get_weights().begin(),
                         fe_values_solution_cell_temp.get_quadrature().get_weights().end());

        return Quadrature<dim>(q_points, q_weights);
    }
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::assemble_system (std::function<double(double, double)> f)
{
    const QGauss<dim>  quadrature_formula(2); // (p+1) quadrature points (p = polynomial degree)
    Quadrature<dim> collected_quadrature;
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const float beta = 0.01;
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    typename DoFHandler<dim>::active_cell_iterator
            solution_cell = dof_handler.begin_active(),        // iterator to first active cell of the solution grid
            solution_endc = dof_handler.end();               // iterator to the one past last active cell of the solution grid
    
    std::ofstream ofs_quadrature_points;
    ofs_quadrature_points.open ("quadrature_points", std::ofstream::out | std::ofstream::app);
    
    for (; solution_cell!=solution_endc; ++solution_cell)
    {
        std::cout<<"Collect quadrature for solution_cell "<<solution_cell<<std::endl;
        collected_quadrature = collect_quadrature(solution_cell, &quadrature_formula);
        FEValues<dim> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values | update_jacobians  |  update_values);

        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(solution_cell);
        
        unsigned int   n_q_points    = collected_quadrature.size();
        std::vector<double>    coefficient_values (n_q_points);
        
        value_list (fe_values.get_quadrature().get_points(),
                    coefficient_values, circle);
        
        for (unsigned int i = 0; i< n_q_points; ++i)
        {
            ofs_quadrature_points<<fe_values.get_quadrature().get_points()[i]<<" "<< coefficient_values[i]<<std::endl;
            std::cout<<fe_values.get_quadrature().get_points()[i]<<std::endl;
            std::cout<<fe_values.get_quadrature().get_weights()[i]<<std::endl;
            
        }
        
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index){      // loop over all quadrature points
            for (unsigned int i=0; i<dofs_per_cell; ++i)  {                   // loop over degrees of freedom
                for (unsigned int j=0; j<dofs_per_cell; ++j)  {              // loop over degrees of freedom

                    cell_matrix(i,j) += (fe_values.shape_grad(i,q_index) * //
                                         fe_values.shape_grad(j,q_index) *
                                         fe_values.JxW(q_index)); //coefficient_values[q_index] *
                    //if (cell_is_cut_by_boundary(solution_cell))
                    //cell_matrix(i,j) += beta * fe_values.shape_value(i,q_index) * fe_values.shape_value(j,q_index);
                }
                cell_rhs(i) += (fe_values.shape_value(i,q_index) * // the cell rhs
                                1.0 *
                                fe_values.JxW(q_index));
                //if (cell_is_cut_by_boundary(solution_cell))
                //cell_rhs(i) += beta * fe_values.shape_value(i,q_index) * 0.0; // Dirichlet boundary condition zero
            }

        }

    solution_cell-> get_dof_indices (local_dof_indices);  // return the global indices of the dof located on this object

    constraints.distribute_local_to_global (cell_matrix,
                                            cell_rhs,
                                            local_dof_indices,
                                            system_matrix,
                                            system_rhs);
    }
    ofs_quadrature_points.close();

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
void adaptiveNumberOfQuadraturePoints<dim>::set_material_ids(DoFHandler<dim> &dof_handler,std::function<double(double, double)> f)
{
    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
        unsigned int counter = 0;
        for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
            if (f(cell->vertex(vertex_iterator)[0], cell->vertex(vertex_iterator)[1])  <= 1){
                counter ++;
            }
        }
        if (counter == 4)
            cell->set_material_id (physical_domain_id);
        
        else if (counter == 0)
            cell->set_material_id (fictitious_domain_id);
        
        else
            cell->set_material_id(boundary_domain_id);
    }
}

template <int dim>
void adaptiveNumberOfQuadraturePoints<dim>::run ()
{
    GridGenerator::hyper_cube (triangulation, -2, 2);
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -2, 2);
    triangulation_adaptiveIntegration.refine_global (2);
    triangulation.refine_global (2);

    set_material_ids(dof_handler_adaptiveIntegration, circle);
    set_material_ids(dof_handler,circle);

        const unsigned int refinement_cycles = 1;
        for (unsigned int i = 0; i < refinement_cycles; i++)
        {
            refine_grid ();
            set_material_ids(dof_handler_adaptiveIntegration, circle);
        }

    setup_system ();
    assemble_system (circle);
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
