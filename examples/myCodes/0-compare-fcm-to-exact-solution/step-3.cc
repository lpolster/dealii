// The finicte cell method on a rectangle. The boundary is given as a polygon (list of vertices clockwise). There is a vertex at least at every cell boundary.
// Poisson problem. Rhs = 1.0. Homogeneous Dirichlet boundary conditions (0.0).
// Nitsche Method for boundary condition.

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/base/smartpointer.h>

// for function VectorTools::integrate_difference()
// and ConvergenceTable:
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include <typeinfo>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <math.h>
#include "mypolygon.h"
#include "fcm-tools.h"
#include "find_cells.h"


namespace FCMImplementation{ // use namespace to avoid the problems that result if names of different functions or variables collide
using namespace dealii;

int dim = 2;

class SolutionBase
{
protected:
    const Point<2>   source_center  = {0.0, 0.0};
    const double       width = 0.25;
};

template <int dim>
class Solution : public Function<dim>,
        protected SolutionBase
{
public:
    Solution () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
    double return_value = 0;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    return_value = std::exp(-x_minus_xi.norm_square() /
                            (width * width));

    return return_value;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
    Tensor<1,dim> return_value;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    return_value = (-2 / (width * width) *
                    std::exp(-x_minus_xi.norm_square() /
                             (width * width)) *
                    x_minus_xi);

    return return_value;
}


template <int dim>
class RightHandSide : public Function<dim>,
        protected SolutionBase
{
public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int) const
{
    double return_value = 0;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    // The Laplacian:
    return_value = ((2*2 - 4*x_minus_xi.norm_square()/
                     (width * width)) /
                    (width * width) *
                    std::exp(-x_minus_xi.norm_square() /
                             (width * width)));

    return return_value;
}



class Step3
{
public:
    Step3 ();

    void run ();

private:
    void make_grid ();
    void setup_system ();
    void setup_grid_and_boundary ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    void refine_grid();
    void coarsen_grid(int global_refinement_level);

    void process_solution(const unsigned int cycle);

    Triangulation<2>                triangulation;                     // triangulation for the solution grid
    FE_Q<2>                         fe;                                // fe for the solution grid
    DoFHandler<2>                   dof_handler;                       // dof handler for the solution grid

    Triangulation<2>                triangulation_adaptiveIntegration;   // triangulation for the integration grid
    FE_Q<2>                         fe_adaptiveIntegration;              // fe for the integration grid
    DoFHandler<2>                   dof_handler_adaptiveIntegration;     // dof handler for the integration grid

    ConstraintMatrix                constraints;

    SparsityPattern                 sparsity_pattern;
    SparseMatrix<double>            system_matrix;                     // system/stiffness matrix

    Vector<double>                  solution;                          // solution/coefficent vector
    Vector<double>                  system_rhs;                        // the right hand side

    ConvergenceTable                convergence_table;

    myPolygon                       my_poly;                           // the polygon boundary
    std::vector<dealii::Point<2>>   point_list;

    double penalty_term;

    double Nitsche_matrix_terms(const int q_index, const int i, const int j, FEValues<2> &fe_values_on_boundary_segment,  myPolygon::segment my_segment);
    double Nitsche_rhs_terms(const int q_index, const int i, FEValues<2> &fe_values_on_boundary_segment,  myPolygon::segment my_segment, const Solution<2> exact_solution);
    void print_cond(double cond);

};

Step3::Step3 ()
    :
      fe (polynomial_degree),                                      // bilinear
      dof_handler (triangulation),
      fe_adaptiveIntegration (1),                                 // bilinear
      dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration)
{}


void Step3::setup_grid_and_boundary ()
{
    // point_list = {{-0.9,0.9}, {0.9, 0.9}, {0.9, -0.9}, {0.2, 0.2}, {-0.9,0.9}}; // this is working
    // point_list = {{-0.9,0.9}, {0.9, 0.9}, {0.9, -0.9}, {-0.9, -0.9}, {-0.9,0.9}};
    // point_list = {{-0.9,0.9}, {0.9, 0.9}, {0.9, -0.9}, {-0.9,0.9}};
    point_list = {{-0.9,0.9}, {0.9, -0.9}, {-0.9, -0.9}, {-0.9,0.9}};

    //        point_list = {{0,0.9}, {0.6, 0.1}, {0, -0.8}, {-0.7,-0.1}, {0,0.9}};
    GridGenerator::hyper_cube (triangulation, -1, 1);       // generate triangulation for solution grid
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -1, 1); // generate triangulation for integration grid

    for (unsigned int i = 0; i< global_refinement_level; i++)
    {
        triangulation.refine_global (1);
        triangulation_adaptiveIntegration.refine_global (1);
        point_list = update_point_list(point_list, triangulation_adaptiveIntegration);
    }
}

void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    dof_handler_adaptiveIntegration.distribute_dofs(fe_adaptiveIntegration);

    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    constraints.close ();

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}

void Step3::assemble_system ()
{
    QGauss<2>  quadrature_formula(polynomial_degree+1);
    Quadrature<2> collected_quadrature;                       // the quadrature rule
    Quadrature<2> collected_quadrature_on_boundary_segment;           // quadrature rule on boundary

    const unsigned int   dofs_per_cell = fe.dofs_per_cell;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);    // the cell matrix
    Vector<double>       cell_rhs (dofs_per_cell);                      // the cell right hand side
    const RightHandSide<2>  right_hand_side;
    const Solution<2>       exact_solution;

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell!=endc; ++cell) // iterate over all cells of solution grid
    {
        cell_matrix = 0;
        cell_rhs = 0;

        // collect quadrature on the cell in solution grid
        collected_quadrature = collect_quadratures(topological_equivalent(cell, triangulation_adaptiveIntegration), &quadrature_formula);

        // man kann denke ich auch ohne fe values arbeiten...
        FEValues<2> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);

        std::vector<double>  rhs_values (collected_quadrature.size());


        fe_values.reinit(cell);                        // reinitialize fe values on current cells
        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);

        //        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), cell, "collected_quadrature");

        // get values of indicator function (alpha)
        std::vector<double> indicator_function_values(collected_quadrature.size());
        indicator_function_values = get_indicator_function_values(fe_values.get_quadrature().get_points(), cell, my_poly);


        //        double weight_counter = 0.0;

        for (unsigned int q_index=0; q_index<collected_quadrature.size(); ++q_index) // loop over all quadrature points in that cell
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j){
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *        // assemble cell matrix
                                         fe_values.shape_grad (j, q_index) *
                                         indicator_function_values[q_index] *
                                         fe_values.JxW (q_index));
                    Assert(std::isfinite(cell_matrix(i,j)), ExcNumberNotFinite(std::complex<double>(cell_matrix(i,j))));
                }

            for (unsigned int i=0; i<dofs_per_cell; ++i){
                cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                                indicator_function_values[q_index] *                // assemble cell right hand side
                                rhs_values [q_index] *
                                fe_values.JxW (q_index));
                Assert(std::isfinite(cell_rhs(i)), ExcNumberNotFinite(std::complex<double>(cell_rhs(i))));
            }

            //            weight_counter += fe_values.get_quadrature().get_weights()[q_index];
            //            std::cout<<"weight = "<<fe_values.get_quadrature().get_weights()[q_index]<<std::endl;
        }

        // Assertion for sum of weihts in cell (= 0.0)
        //        std::cout<<"Sum of weights in cell: "<<weight_counter<<std::endl;

        if (contains_boundary(cell, my_poly))
        {
            std::vector<int> segment_indices = my_poly.get_segment_indices_inside_cell(cell);

            for (unsigned int k = 0; k < segment_indices.size(); ++ k){

                myPolygon::segment my_segment = my_poly.segment_list[segment_indices[k]];

                // Nitsche method

                collected_quadrature_on_boundary_segment = collect_quadratures_on_boundary_segment(my_segment, cell);

                FEValues<2> fe_values_on_boundary_segment (fe, collected_quadrature_on_boundary_segment, update_quadrature_points |  update_gradients |  update_values | update_JxW_values);

                fe_values_on_boundary_segment.reinit(cell);

                for (unsigned int q_index=0; q_index<my_segment.q_points.size(); ++q_index)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)  { // loop over degrees of freedom
                        for (unsigned int j=0; j<dofs_per_cell; ++j)  {// loop over degrees of freedom

                            cell_matrix(i,j) +=  Nitsche_matrix_terms(q_index, i, j, fe_values_on_boundary_segment, my_segment);

                        } // endfor
                        cell_rhs(i) += Nitsche_rhs_terms(q_index, i, fe_values_on_boundary_segment, my_segment, exact_solution);

                    } // endfor

                } // endfor
            }

        } // endfor


        cell->get_dof_indices (local_dof_indices);

        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
    }
    //    std::ofstream ofs_system_matrix;
    //    ofs_system_matrix.open ("matrix.txt", std::ofstream::out);
    //    system_matrix.print(ofs_system_matrix);
    //    ofs_system_matrix.close();

    //    std::ofstream ofs_sparsity_pattern;
    //    ofs_sparsity_pattern.open ("sparsity_pattern.txt", std::ofstream::out);
    //    system_matrix.print_pattern(ofs_sparsity_pattern);
    //    ofs_sparsity_pattern.close();
}

double Step3::Nitsche_matrix_terms(const int q_index, const int i, const int j, FEValues<2> &fe_values_on_boundary_segment,  myPolygon::segment my_segment)
{
    double Nitsche_matrix_terms = 0.0;
    Nitsche_matrix_terms += penalty_term * (fe_values_on_boundary_segment.shape_value(i,q_index) *
                                            fe_values_on_boundary_segment.shape_value(j,q_index) *
                                            my_segment.length *
                                            fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

    Nitsche_matrix_terms -= (fe_values_on_boundary_segment.shape_value(i,q_index) *
                             my_segment.normalVector *
                             fe_values_on_boundary_segment.shape_grad(j,q_index) * my_segment.length * //fe_values_on_boundary_segment.JxW (q_index));
                             fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

    Nitsche_matrix_terms -= (fe_values_on_boundary_segment.shape_value(j,q_index) *
                             my_segment.normalVector *
                             fe_values_on_boundary_segment.shape_grad(i,q_index) *
                             my_segment.length*
                             fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

    Assert(std::isfinite(Nitsche_matrix_terms), ExcNumberNotFinite(std::complex<double>(Nitsche_matrix_terms)));

    return Nitsche_matrix_terms;
}

double Step3::Nitsche_rhs_terms(const int q_index, const int i, FEValues<2> &fe_values_on_boundary_segment,  myPolygon::segment my_segment, const Solution<2> exact_solution)
{
    double Nitsche_rhs_terms = 0.0;

    double dirichlet_boundary_value = exact_solution.value (fe_values_on_boundary_segment.quadrature_point(q_index));
    Nitsche_rhs_terms -= (dirichlet_boundary_value * fe_values_on_boundary_segment.shape_grad(i,q_index) *
                          my_segment.normalVector * my_segment.length *
                          fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
    Nitsche_rhs_terms +=  (penalty_term * fe_values_on_boundary_segment.shape_value(i,q_index) *
                           dirichlet_boundary_value * my_segment.length *
                           fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

    Assert(std::isfinite(Nitsche_rhs_terms), ExcNumberNotFinite(std::complex<double>(Nitsche_rhs_terms)));

    return Nitsche_rhs_terms;
}

void Step3::print_cond(double cond){
    std::cout<<"cond="<<cond <<std::endl;
}

void Step3::solve ()
{
    //    SparseDirectUMFPACK  A_direct;              // use direct solver
    //    A_direct.initialize(system_matrix);
    //    A_direct.vmult (solution, system_rhs);

    SolverControl           solver_control (100000, 1e-12);
    SolverCG<>              solver (solver_control);

    solver.connect_condition_number_slot(std_cxx11::bind(&Step3::print_cond,this,std_cxx11::_1));

    solver.solve (system_matrix, solution, system_rhs,
                  PreconditionIdentity());
}


void Step3::output_results () const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches (); // linear interpolation for plotting

    std::ofstream output_gpl ("solution.gpl");
    data_out.write_gnuplot (output_gpl);

    std::ofstream output_vtk ("solution.vtk");
    data_out.write_vtk (output_vtk);
}


void output_grid(const dealii::Triangulation<2>& tria,
                 std::string name,
                 const unsigned int nr1, const unsigned int nr2)
{
    GridOut grid_out;
    std::stringstream filename;
    filename << name << "-" << nr1 << "-" << nr2 << ".svg";
    std::ofstream out(filename.str());
    grid_out.write_svg(tria, out);
}

void Step3::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    myPolygon  my_poly;
    my_poly.constructPolygon(point_list);
    typename DoFHandler<2>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler_adaptiveIntegration.begin_active(), // the first active cell
            endc = dof_handler_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (contains_boundary(cell, my_poly))
        {
            cell -> set_refine_flag();
        }
    }

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}

void Step3::coarsen_grid (int global_refinement_level)
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    myPolygon  my_poly;
    my_poly.constructPolygon(point_list);
    typename DoFHandler<2>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler_adaptiveIntegration.begin_active(), // the first active cell
            endc = dof_handler_adaptiveIntegration.end(); // one past the last active cell
    typename DoFHandler<2>::active_cell_iterator previous_cell  = dof_handler_adaptiveIntegration.begin_active();

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (cell->level() > global_refinement_level)
            cell -> set_coarsen_flag();
    }

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}

void Step3::process_solution (const unsigned int cycle)
{

    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<2>(),
                                       difference_per_cell,
                                       QGauss<2>(3),
                                       VectorTools::L2_norm);
    const double L2_error = difference_per_cell.l2_norm();

    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<2>(),
                                       difference_per_cell,
                                       QGauss<2>(3),
                                       VectorTools::H1_seminorm);
    const double H1_error = difference_per_cell.l2_norm();

    const QTrapez<1>     q_trapez;
    const QIterated<2> q_iterated (q_trapez, 5);
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<2>(),
                                       difference_per_cell,
                                       q_iterated,
                                       VectorTools::Linfty_norm);
    const double Linfty_error = difference_per_cell.linfty_norm();

    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs = dof_handler.n_dofs();

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
}


void Step3::run ()
{
    setup_grid_and_boundary ();
    for (unsigned int global_refinement_cycles = 1; global_refinement_cycles < 7; global_refinement_cycles++ )
    {
        std::cout << "Cycle " << global_refinement_cycles << std::endl;
        triangulation.refine_global (1);
        triangulation_adaptiveIntegration.refine_global (1);
        penalty_term = 2.0 * polynomial_degree * (polynomial_degree+1) * (global_refinement_cycles + global_refinement_level);
        std::cout<<"Update point list..."<<std::endl;
        point_list = update_point_list(point_list, triangulation_adaptiveIntegration);

        output_grid(triangulation_adaptiveIntegration, "adaptiveGrid", (global_refinement_cycles+global_refinement_level), 0);

        for (unsigned int i = 0; i < refinement_cycles; i++)
        {
            refine_grid();
            std::cout<<"Update point list..."<<std::endl;
            point_list = update_point_list(point_list, triangulation_adaptiveIntegration);
            output_grid(triangulation_adaptiveIntegration, "adaptiveGrid", (global_refinement_cycles+global_refinement_level), i+1);
        }
        std::cout<<"Construct poly..."<<std::endl;
        my_poly.constructPolygon(point_list);                   // construct polygon from list of points
        std::cout<<"Setting up the system..."<<std::endl;
        setup_system ();
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells()
                  << std::endl
                  << "   Number of degrees of freedom: "
                  << dof_handler.n_dofs()
                  << std::endl;
        std::cout<<"Assembling the system..."<<std::endl;
        assemble_system ();
        std::cout<<"Solving..."<<std::endl;
        solve ();
        std::cout<<"Process solution..."<<std::endl;
        process_solution(global_refinement_cycles);

        for (unsigned int i = 0; i < refinement_cycles; i++)
        {
            coarsen_grid(global_refinement_cycles+global_refinement_level);
            output_grid(triangulation_adaptiveIntegration, "adaptiveGrid_coarsened", (global_refinement_cycles+global_refinement_level), i+1);
        }
    }

    std::cout<<"Output results..."<<std::endl;
    output_results ();

    std::cout<<"Constructing the convergence table..."<<std::endl;
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    convergence_table.add_column_to_supercolumn("cycle", "n cells");
    convergence_table.add_column_to_supercolumn("cells", "n cells");

    std::vector<std::string> new_order;
    new_order.push_back("n cells");
    new_order.push_back("H1");
    new_order.push_back("L2");
    convergence_table.set_column_order (new_order);

    convergence_table
            .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate);
    convergence_table
            .evaluate_convergence_rates("L2", ConvergenceTable::reduction_rate_log2);
    convergence_table
            .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate);
    convergence_table
            .evaluate_convergence_rates("H1", ConvergenceTable::reduction_rate_log2);

    std::cout << std::endl;
    convergence_table.write_text(std::cout);
}
}


int main ()
{
    try
    {
        using namespace dealii;
        using namespace FCMImplementation;
        
        std::remove("indicator_function_values");
        std::remove("collected_quadrature");
        
        Step3 laplace_problem;
        laplace_problem.run ();
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
