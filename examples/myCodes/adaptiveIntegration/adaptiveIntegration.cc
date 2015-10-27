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

using namespace dealii;


template <int dim>
class Step6
{
public:
    Step6 ();
    ~Step6 ();

    void run ();

private:

    static bool
    cell_is_in_physical_domain (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_is_in_fictitious_domain (const typename DoFHandler<dim>::cell_iterator &cell);
    static bool
    cell_is_cut_by_boundary (const typename DoFHandler<dim>::cell_iterator &cell);

    void setup_system ();
    void assemble_system (const float length, const float height);
    void solve ();
    void refine_grid ();
    void output_results ();
    void set_material_ids(DoFHandler<dim> &dof_handler, const float length, const float height);
    void add_constraints_on_interface(DoFHandler<dim> &dof_handler, FE_Q<dim> &fe, ConstraintMatrix &constraints);
    void write_solution_to_file (Vector<double> solution);

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
bool Step6<dim>::cell_is_in_physical_domain (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == physical_domain_id);
}

template <int dim>
bool Step6<dim>::cell_is_in_fictitious_domain (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == fictitious_domain_id);
}

template <int dim>
bool Step6<dim>::cell_is_cut_by_boundary (const typename DoFHandler<dim>::cell_iterator &cell)
{
    return (cell->material_id() == boundary_domain_id);
}

template <int dim>
class Coefficient : public Function<dim>
{
public:
    Coefficient () : Function<dim>() {}

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const float length, const float height,
                             const unsigned int              component = 0) const;
};

template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const float length, const float height,
                                   const unsigned int             component) const
{
    const unsigned int n_points = points.size();

    Assert (values.size() == n_points,
            ExcDimensionMismatch (values.size(), n_points));

    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    for (unsigned int i=0; i<n_points; ++i)
    {
        if (points[i](0) <= length/2 && points[i](0) >= -length/2 && points[i](1) <= height/2 && points[i](1) >= -height/2)
            values[i] = 1; // indicates physical domain
        else
            values[i] = 1e-8; // indicates fictitous domain
    }
}

template <int dim>
Step6<dim>::Step6 ()
    :
      dof_handler (triangulation),
      dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration),
      fe (1),
      fe_adaptiveIntegration (1)
{}

template <int dim>
Step6<dim>::~Step6 ()
{
    dof_handler.clear ();
    dof_handler_adaptiveIntegration.clear();
}

template <int dim>
void Step6<dim>::add_constraints_on_interface(DoFHandler<dim> &dof_handler, FE_Q<dim> &fe, ConstraintMatrix &constraints)
{
    std::vector<types::global_dof_index> local_face_dof_indices (fe.dofs_per_face);
    for (typename DoFHandler<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
        if (cell_is_in_physical_domain (cell))
        {
//            std::cout<<"Cell "<<cell<<" is in physical domain."<<std::endl;
            for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
                if (!cell->at_boundary(f))
                {
                    bool face_is_on_interface = false;
//                    if ((cell->neighbor(f)->has_children() == false)
//                            &&
//                            (cell_is_cut_by_boundary (cell->neighbor(f))))
//                        face_is_on_interface = true;
//                    else if (cell->neighbor(f)->has_children() == true)
//                    {
//                        for (unsigned int sf=0; sf<cell->face(f)->n_children(); ++sf)
//                            if (cell_is_cut_by_boundary (cell->neighbor_child_on_subface
//                                                              (f, sf)))
//                            {
//                                face_is_on_interface = true;
//                                break;
//                            }
//                    }
                    if(cell_is_cut_by_boundary (cell->neighbor(f)) )face_is_on_interface = true;
                    if (face_is_on_interface)
                    {
                        cell->face(f)->get_dof_indices (local_face_dof_indices, 0);
                        for (unsigned int i=0; i<local_face_dof_indices.size(); ++i)
                            if (fe.face_system_to_component_index(i).first < dim)
                                constraints.add_line (local_face_dof_indices[i]);
                    }
                }
        }
}

template <int dim>
void Step6<dim>::setup_system ()
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
    add_constraints_on_interface(dof_handler, fe, constraints);
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
void Step6<dim>::assemble_system (const float length, const float height)
{
    const QGauss<dim>  quadrature_formula(2); // (p+1) quadrature points (p = polynomial degree)
    FEValues<dim> fe_values (fe_adaptiveIntegration, quadrature_formula,
                             update_values    |  update_gradients |
                             update_quadrature_points  |  update_JxW_values | update_jacobians);
    FEValues<dim> fe_values_solutionGrid (fe, quadrature_formula,  update_quadrature_points  |
                                          update_JxW_values | update_jacobians);
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    unsigned int refinement_level;
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const Coefficient<dim> coefficient;
    std::vector<double>    coefficient_values (n_q_points);

    typename DoFHandler<dim>::active_cell_iterator
            cell = dof_handler_adaptiveIntegration.begin_active(),        // iterator to first active cell of the adaptive grid
            endc = dof_handler_adaptiveIntegration.end(),                 // iterator to the one past last active cell of the adaptive grid
            solution_cell = dof_handler.begin_active(),                   // iterator to first active cell of the solution grid
            solution_endc = dof_handler.end();                            // iterator to the one past last active cell of the solution grid

    for (; solution_cell!=solution_endc; ++solution_cell){
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values_solutionGrid.reinit(solution_cell);
        for (; cell!=endc; ++cell)                                            // loop over all active cells
        {
            if ( !cell_is_in_fictitious_domain(cell) && (solution_cell->vertex(0)[0] <= cell->vertex(0)[0] && solution_cell->vertex(1)[0] >= cell->vertex(1)[0]
                    && solution_cell->vertex(0)[1] <= cell->vertex(0)[1] && solution_cell->vertex(3)[1] >= cell->vertex(3)[1]))
            {
                refinement_level = (cell->level()) - (solution_cell->level());
                fe_values.reinit (cell);
                coefficient.value_list (fe_values.get_quadrature_points(),
                                        coefficient_values, length, height);
                for (unsigned int q_index=0; q_index<n_q_points; ++q_index){      // loop over all quadrature points
                    //                    std::cout<<"Quadrature point = "<< fe_values.get_quadrature_points()[q_index]<<", Value = "
                    //                            << coefficient_values[q_index] <<std::endl;
                    for (unsigned int i=0; i<dofs_per_cell; ++i)  {                   // loop over degrees of freedom
                        for (unsigned int j=0; j<dofs_per_cell; ++j)  {              // loop over degrees of freedom

                            cell_matrix(i,j) += (coefficient_values[q_index] * fe_values.shape_grad(i,q_index) *
                                                 fe_values.shape_grad(j,q_index) *
                                                 fe_values.JxW(q_index))/pow((dim-1)*4,refinement_level); //* fe_values_solutionGrid.JxW(q_index)
//                            std::cout<<"cell_matrix("<<i<<", "<<j<<") = "<< cell_matrix(i,j)<<std::endl;
                        }
                        cell_rhs(i) += (coefficient_values[q_index] * fe_values.shape_value(i,q_index) *           // the cell right hand side
                                        1.0 *
                                        fe_values.JxW(q_index));
                    }
//                    std::cout<<"fe_values.JxW(q_index) = "<<fe_values.JxW(q_index)<<std::endl;
//                    std::cout<<"fe_values_solutionGrid.JxW(q_index) = "<<fe_values_solutionGrid.JxW(q_index)<<std::endl;
                }
            }
        }
        std::ofstream ofs_cell_matrix;
        ofs_cell_matrix.open ("cell_matrices", std::ofstream::out | std::ofstream::app); // In case we want to append | std::ofstream::app
        ofs_cell_matrix<<"Solution cell: "<<solution_cell<<std::endl;
        cell_matrix.print(ofs_cell_matrix);
        ofs_cell_matrix<<"----------------------------------------------"<<std::endl;
        ofs_cell_matrix.close();

        solution_cell-> get_dof_indices (local_dof_indices);              // return the global indices of the degrees of freedom located on this object

        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);

        cell = dof_handler_adaptiveIntegration.begin_active();
    }
    std::ofstream ofs_system_matrix;
    ofs_system_matrix.open ("system_matrix", std::ofstream::out); // In case we want to append | std::ofstream::app
    system_matrix.print_formatted(ofs_system_matrix,2, false);
    ofs_system_matrix.close();

    std::ofstream ofs_system_rhs;
    ofs_system_rhs.open ("system_rhs", std::ofstream::out); // In case we want to append | std::ofstream::app
    system_rhs.print(ofs_system_rhs,2,false,true);
    ofs_system_rhs.close();
}


template <int dim>
void Step6<dim>::solve ()
{
    SolverControl      solver_control (1000, 1e-12);
    SolverCG<>         solver (solver_control);

    PreconditionSSOR<> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());

    constraints.distribute (solution);

    std::ofstream ofs;
    ofs.open ("solution", std::ofstream::out);
    solution.print(ofs, 2, false, true);
    ofs.close();
}

template <int dim>
void Step6<dim>::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    Vector<float> contains_boundary (triangulation_adaptiveIntegration.n_active_cells());
    contains_boundary = 0;
    unsigned int i = 0; // integer for accessing the right cell

    typename DoFHandler<dim>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler_adaptiveIntegration.begin_active(), // the first active cell
            endc = dof_handler_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (cell_is_cut_by_boundary(cell))
        {
            contains_boundary[i] = 1.0;
        }
        else
            contains_boundary[i] = 0.0;
        i++;
    }

    GridRefinement::refine (triangulation_adaptiveIntegration,contains_boundary, 0.9,3000);
    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}

template <int dim>
void Step6<dim>::output_results ()
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
void Step6<dim>::set_material_ids(DoFHandler<dim> &dof_handler,const float length, const float height)
{
    for (typename Triangulation<dim>::active_cell_iterator
         cell = dof_handler.begin_active();
         cell != dof_handler.end(); ++cell)
    {
        unsigned int counter = 0;
        for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
            if (cell->vertex(vertex_iterator)[0] <= (length/2) && cell->vertex(vertex_iterator)[0] >= -(length/2)
                    && cell->vertex(vertex_iterator)[1] <= (height/2) && cell->vertex(vertex_iterator)[1] >= -(height/2)){
                counter ++;
            }
        }
        if (counter == 4)
        {
            cell->set_material_id (physical_domain_id);
//            std::cout<<"Setting material id of cell "<<cell<<" to "<<physical_domain_id<<std::endl;
        }
        else if (counter == 0)
        {
            cell->set_material_id (fictitious_domain_id);
//            std::cout<<"Setting material id of cell "<<cell<<" to "<<fictitious_domain_id<<std::endl;
        }
        else
        {
            cell->set_material_id(boundary_domain_id);
//            std::cout<<"Setting material id of cell "<<cell<<" to "<<boundary_domain_id<<std::endl;
        }
    }
}

template <int dim>
void Step6<dim>::run ()
{
    GridGenerator::hyper_cube (triangulation, -2, 2);
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -2, 2);
    triangulation_adaptiveIntegration.refine_global (5);
    triangulation.refine_global (5);

    const float rectangle_length = 2.0, rectangle_height = 2.0;
    set_material_ids(dof_handler_adaptiveIntegration,rectangle_length, rectangle_height);
    set_material_ids(dof_handler,rectangle_length, rectangle_height);

//    const unsigned int refinement_cycles = 3;
//    for (unsigned int i = 0; i < refinement_cycles; i++)
//    {
//        refine_grid ();
//        set_material_ids(dof_handler_adaptiveIntegration, rectangle_length, rectangle_height);
//    }

    setup_system ();
    assemble_system (rectangle_length, rectangle_height);
    solve ();
    output_results ();
}



int main ()
{

    try
    {
        deallog.depth_console (0);

        Step6<2> laplace_problem_2d;
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
