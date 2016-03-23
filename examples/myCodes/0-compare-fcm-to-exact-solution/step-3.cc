 /* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2015 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Wolfgang Bangerth and Ralf Hartmann, University of Heidelberg, 2000
 */


// @sect3{Include files}

// These first include files have all been treated in previous examples, so we
// won't explain what is in them again.
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
#include <cstdlib>

// for function VectorTools::integrate_difference()
// and ConvergenceTable:
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>

#include <typeinfo>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <math.h>

#include "parameters.h"
#include "find_cells.h"

#include "mypolygon.h"
#include "fcm-tools.h"
#include "function_classes.h"


#define FCM_DEF
//#define FEM_DEF

namespace Step7
{
using namespace dealii;

template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem (const FiniteElement<dim> &fe);
    ~LaplaceProblem ();

    void run ();

private:
    void setup_system ();
    void assemble_system ();
    void solve ();
    void refine_grid_globally ();
    void process_solution (const unsigned int cycle);
    void output_grid(const dealii::Triangulation<dim>& tria,
                     std::string name,
                     const unsigned int nr1);

    Triangulation<dim>                      triangulation;
    DoFHandler<dim>                         dof_handler;

    SmartPointer<const FiniteElement<dim> > fe;

    SparsityPattern                         sparsity_pattern;
    SparseMatrix<double>                    system_matrix;

    Vector<double>                          solution;
    Vector<double>                          system_rhs;

    ConvergenceTable                        convergence_table;

#ifdef FCM_DEF
    void setup_grid_and_boundary ();
    void refine_grid_adaptively ();
    void coarsen_grid_adaptively (int global_refinement_level);

    std::vector<dealii::Point<2>>           point_list;
    Triangulation<dim>                      triangulation_adaptiveIntegration;   // triangulation for the integration grid
    myPolygon                               my_poly;
    double                                  penalty_term;
#endif

};


template <int dim>
LaplaceProblem<dim>::LaplaceProblem (const FiniteElement<dim> &fe) : // The constructor of the class
    triangulation_adaptiveIntegration(Triangulation<dim>::none),
    dof_handler (triangulation),
    fe (&fe)
{}

template <int dim>
LaplaceProblem<dim>::~LaplaceProblem () // The destructor of the class
{
    dof_handler.clear ();
}
    
// // // Set-up the triangulation and the polygon boundary // // //
#ifdef FCM_DEF
template <int dim>
void LaplaceProblem<dim>::setup_grid_and_boundary ()
{
//   point_list = {{-0.5,0.5}, {0.5, 0.5}, {0.5, -0.5}, {-0.5, -0.5}, {-0.5,0.5}};
//    point_list = {{-0.9,0.9}, {0.9, 0.9}, {0.9, -0.9}, {-0.9, -0.9}, {-0.9,0.9}};
//    point_list = {{-0.7,0.7}, {0.6, 0.6}, {0.75, -0.75}, {-0.7, -0.6}, {-0.7,0.7}};
//    point_list = {{-1.0,1.0}, {1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0,1.0}};
    point_list = {{-0.7,0.7}, {0.7, 0.7}, {0.0, -0.7}, {-0.7,0.7}};
//    point_list = {{-1.0,1.0}, {1.0, 1.0}, {1.0, -1.0}, {-1.0, -1.0}, {-1.0,1.0}};

    GridGenerator::hyper_cube (triangulation, lower_embedded_domain, upper_embedded_domain);       // generate triangulation for solution grid
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, lower_embedded_domain, upper_embedded_domain); // generate triangulation for integration grid

    triangulation.refine_global (1);
    triangulation_adaptiveIntegration.refine_global (1);
    point_list = update_point_list(point_list, triangulation_adaptiveIntegration);

    triangulation.refine_global (1);
    triangulation_adaptiveIntegration.refine_global (1);
    point_list = update_point_list(point_list, triangulation_adaptiveIntegration);
}
#endif


// // // Set-up the system // // //
template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
    dof_handler.distribute_dofs (*fe);

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from (dsp);

    system_matrix.reinit (sparsity_pattern);
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}


// // // Assemble the system // // //
template <int dim>
void LaplaceProblem<dim>::assemble_system ()
{
    QGauss<dim>   quadrature_formula(n_quadrature_points); // Gauss quadrature with n quadrator points

#ifdef FCM_DEF
    my_poly.constructPolygon(point_list);                   // construct polygon from list of points
    my_poly.save_q_points();

    Quadrature<dim> collected_quadrature;                   // the quadrature rule

    std::remove("FCM_quadrature");
    std::remove("indicator_function_values");
#endif

    const unsigned int dofs_per_cell = fe->dofs_per_cell;

    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

    const RightHandSide<dim> right_hand_side;
    const Solution<dim> exact_solution;


#ifdef FEM_DEF
    FEValues<dim>  fe_values (*fe, quadrature_formula,
                              update_values   | update_gradients |
                              update_quadrature_points | update_JxW_values);
    std::vector<double>  rhs_values (quadrature_formula.size());
    std::remove("FEM_quadrature");
#endif


    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {
        cell_matrix = 0;
        cell_rhs = 0;

#ifdef FCM_DEF
        collected_quadrature = collect_quadratures(topological_equivalent(cell, triangulation_adaptiveIntegration), &quadrature_formula);
        FEValues<dim> fe_values(*fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);
        std::vector<double>  rhs_values (collected_quadrature.size());
#endif

//        std::cout<<"Cell "<<cell<<" has "<<collected_quadrature.size()<<" quadrature points."<<std::endl;
        fe_values.reinit (cell);
        right_hand_side.value_list (fe_values.get_quadrature_points(),
                                    rhs_values);

#ifdef FCM_DEF
        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), cell, "FCM_quadrature");

        std::vector<double> indicator_function_values(collected_quadrature.size());
        indicator_function_values = get_indicator_function_values(fe_values.get_quadrature().get_points(), cell, my_poly);
#endif

#ifdef FEM_DEF
        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), cell, "FEM_quadrature");
#endif

        // Assemble the cell matrix //
        for (unsigned int q_point=0; q_point<collected_quadrature.size(); ++q_point)
            for (unsigned int i=0; i<dofs_per_cell; ++i){
                for (unsigned int j=0; j<dofs_per_cell; ++j){
                    double temp_matrix_entry = (fe_values.shape_grad(i,q_point) *
                                                fe_values.shape_grad(j,q_point) *
                                                fe_values.JxW(q_point));
#ifdef FCM_DEF
                    temp_matrix_entry *= indicator_function_values[q_point];
#endif
                    cell_matrix(i,j) += temp_matrix_entry;
                }

                double temp_rhs_entry = (fe_values.shape_value(i,q_point) *
                                         rhs_values [q_point] *
                                         fe_values.JxW(q_point));
#ifdef FCM_DEF
                temp_rhs_entry *= indicator_function_values[q_point];
#endif
                cell_rhs(i) += temp_rhs_entry;
            } //endfor

        // Nitsche boundary conditions

#ifdef FCM_DEF
        Point<dim>      normal_vector;
        double          segment_length;
        Quadrature<dim> collected_quadrature_on_boundary_segment;           // quadrature rule on boundary

        if (contains_boundary(cell, my_poly))
        {
            {
                std::vector<int> segment_indices = my_poly.get_segment_indices_inside_cell(cell);

                for (unsigned int k = 0; k < segment_indices.size(); ++ k){

                    myPolygon::segment my_segment = my_poly.segment_list[segment_indices[k]];
                    segment_length = my_segment.length;
//                    std::cout<<"Cell "<<cell<<" contains segment ["<<my_segment.beginPoint<<" "<<my_segment.endPoint<<"]"<<std::endl;
                    normal_vector =  my_segment.normalVector;

                    collected_quadrature_on_boundary_segment = collect_quadratures_on_boundary_segment(my_segment, cell);

                    FEValues<dim> fe_values_on_boundary_segment (*fe, collected_quadrature_on_boundary_segment, update_quadrature_points |  update_gradients |  update_values | update_JxW_values);

                    fe_values_on_boundary_segment.reinit(cell);

                    for (unsigned int q_index=0; q_index<my_segment.q_points.size(); ++q_index) {
                        for (unsigned int i=0; i<dofs_per_cell; ++i)  { // loop over degrees of freedom
                            for (unsigned int j=0; j<dofs_per_cell; ++j)  {// loop over degrees of freedom

                                cell_matrix(i,j) += penalty_term * (fe_values_on_boundary_segment.shape_value(i,q_index) *
                                                                    fe_values_on_boundary_segment.shape_value(j,q_index) *
                                                                    fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index] *
                                                                    segment_length);

                                cell_matrix(i,j) -= (fe_values_on_boundary_segment.shape_value(i,q_index)*
                                                     fe_values_on_boundary_segment.shape_grad(j,q_index) *
                                                     normal_vector *
                                                     fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index] *
                                                     segment_length);

                                cell_matrix(i,j) -= (fe_values_on_boundary_segment.shape_value(j,q_index) *
                                                     fe_values_on_boundary_segment.shape_grad(i,q_index) *
                                                     normal_vector *
                                                     fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index] *
                                                     segment_length);

//                                Assert(std::isfinite(cell_matrix(i,j)), ExcNumberNotFinite(std::complex<double>(cell_matrix(i,j))));

                            } // endfor
                            cell_rhs(i) += (penalty_term * fe_values_on_boundary_segment.shape_value(i,q_index) *
                                            exact_solution.value (fe_values_on_boundary_segment.quadrature_point(q_index)) *
                                            segment_length *
                                            fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
                            cell_rhs(i) -= (exact_solution.value (fe_values_on_boundary_segment.quadrature_point(q_index)) *
                                            fe_values_on_boundary_segment.shape_grad(i,q_index) *
                                            normal_vector * segment_length *
                                            fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

                        } // endfor

                    } // endfor
                }
            };
        }
#endif

//        if (contains_boundary(cell, my_poly)){
////            std::cout<<"Cell: "<<cell<<" ("<<i<<","<<j<<") = "<<cell_matrix(i,j)<<std::endl;
//            std::cout<<"Cell matrix of cell "<<cell<<std::endl;
//            for (unsigned int i = 0; i < 4; ++i){
//                for (unsigned int j = 0; j < 4; ++j)
//                    std::cout<<cell_matrix(i,j)<<" ";
//                std::cout<<std::endl;
//            }

//        }
        cell->get_dof_indices (local_dof_indices);


        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs(local_dof_indices[i]) += cell_rhs(i);

    }


#ifdef FEM_DEF
    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              Solution<dim>(),
                                              boundary_values);
    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
#endif
}

template <int dim>
void LaplaceProblem<dim>::solve ()
{
    SparseDirectUMFPACK  A_direct;              // use direct solver
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);

    std::ofstream ofs_system_matrix;
    ofs_system_matrix.open("system_matrix.txt", std::ofstream::out);
    system_matrix.print(ofs_system_matrix);
    ofs_system_matrix.close();
}



template <int dim>
void LaplaceProblem<dim>::refine_grid_globally ()
{
    triangulation.refine_global (1);

#ifdef FCM_DEF
    triangulation_adaptiveIntegration.refine_global(1);
    point_list = update_point_list(point_list, triangulation);
#endif

}

#ifdef FCM_DEF
template <int dim>
void LaplaceProblem<dim>::refine_grid_adaptively ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    my_poly.constructPolygon(point_list);
    typename Triangulation<dim>::active_cell_iterator cell = triangulation_adaptiveIntegration.begin_active(), // the first active cell
            endc = triangulation_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
        if (contains_boundary(cell, my_poly))
            cell -> set_refine_flag();

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}

template <int dim>
void LaplaceProblem<dim>::coarsen_grid_adaptively (int global_refinement_level)
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    typename Triangulation<dim>::active_cell_iterator // an iterator over all active cells
            cell = triangulation_adaptiveIntegration.begin_active(), // the first active cell
            endc = triangulation_adaptiveIntegration.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
        if (cell->level() > global_refinement_level)
            cell -> set_coarsen_flag();

    triangulation_adaptiveIntegration.execute_coarsening_and_refinement ();
}
#endif

template <int dim>
void LaplaceProblem<dim>::process_solution (const unsigned int cycle)
{

#ifdef FCM_DEF
    std::vector<Point<2>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    MaskFunction<dim> mask (my_poly, triangulation);

    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(n_quadrature_points + 4),
                                       VectorTools::L2_norm, &mask);
#endif

#ifdef FEM_DEF
    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(n_quadrature_points + 4),
                                       VectorTools::L2_norm);
#endif

    const double L2_error = difference_per_cell.l2_norm();

    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);

#ifdef FEM_DEF
    data_out.add_data_vector (difference_per_cell, "FEM_L2error_solution");
#endif

#ifdef FCM_DEF
    data_out.add_data_vector (difference_per_cell, "FCM_L2error_solution");
#endif

    data_out.build_patches (fe->degree); // linear interpolation for plotting

    std::string vtk_filename;

#ifdef FEM_DEF
    std::ofstream output_gpl ("FEM_L2error_solution.gpl");
    vtk_filename = "FEM_L2error_solution";
#endif

#ifdef FCM_DEF
    std::ofstream output_gpl ("FCM_L2error_solution.gpl");
    vtk_filename = "FCM_L2error_solution";
#endif

    vtk_filename += ".vtk";
    std::ofstream output_vtk (vtk_filename.c_str());

    data_out.write_vtk (output_vtk);
    data_out.write_gnuplot (output_gpl);

#ifdef FCM_DEF
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(n_quadrature_points + 4),
                                       VectorTools::H1_seminorm, &mask);
#endif

#ifdef FEM_DEF
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       Solution<dim>(),
                                       difference_per_cell,
                                       QGauss<dim>(n_quadrature_points + 4),
                                       VectorTools::H1_seminorm);
#endif
    const double H1_error = difference_per_cell.l2_norm();

    const unsigned int n_active_cells=triangulation.n_active_cells();
    const unsigned int n_dofs=dof_handler.n_dofs();

    std::cout << "Cycle " << cycle << ':'
              << std::endl
              << "   Number of active cells:       "
              << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: "
              << n_dofs
              << std::endl;

    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
}

template <int dim>
void LaplaceProblem<dim>::output_grid(const dealii::Triangulation<dim>& tria,
                                      std::string name,
                                      const unsigned int nr1)
{
    GridOut grid_out;
    std::stringstream filename;
    filename << name << "-" << nr1 << ".svg";
    std::ofstream out(filename.str());
    grid_out.write_svg(tria, out);
}


template <int dim>
void LaplaceProblem<dim>::run ()
{
    const unsigned int n_cycles = 3;
    for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
    {
        if (cycle == 0)
        {
#ifdef FCM_DEF
            setup_grid_and_boundary();
#endif

#ifdef FEM_DEF
            GridGenerator::hyper_cube (triangulation, lower_embedded_domain, upper_embedded_domain);
            triangulation.refine_global (2);
#endif
           output_grid(triangulation_adaptiveIntegration, "globalGrid", 0);

        }
        else{
            refine_grid_globally ();
//        my_poly.save_q_points();
        }

#ifdef FCM_DEF
        penalty_term = polynomial_degree * (polynomial_degree+1) * (cycle+1);

        //        std::cout<<"Penalty parameter: "<<penalty_term<<std::endl;
        //        output_grid(triangulation_adaptiveIntegration, "globalGrid", cycle+1);

        
        clock_t begin = clock();
        for (unsigned int i = 0; i < n_adaptive_refinement_cycles; ++i)
        {
            refine_grid_adaptively();
            output_grid(triangulation_adaptiveIntegration, "adaptiveGrid", cycle);
        }
        clock_t end = clock();
        std::cout<<"Elapsed time for adaptive refinement = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;
#endif

        setup_system ();

        begin = clock();
        assemble_system ();
        end = clock();
        std::cout<<"Elapsed time for matrix assembly = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

        solve ();

        DataOut<dim> data_out;
        data_out.attach_dof_handler (dof_handler);
        data_out.add_data_vector (solution, "solution");
        data_out.build_patches (fe->degree); // linear interpolation for plotting

        std::ofstream output_gpl ("solution.gpl");
        data_out.write_gnuplot (output_gpl);

        begin = clock();
        process_solution (cycle);
        end = clock();
        std::cout<<"Elapsed time for solution processing = " << double(end - begin) / CLOCKS_PER_SEC << std::endl;

#ifdef FCM_DEF
        for (unsigned int i = 0; i < n_adaptive_refinement_cycles; ++i)
            coarsen_grid_adaptively(1+cycle);

#endif
    }
    
    my_poly.save_segments();                                // save segments to text file for plotting


    std::string vtk_filename;

    vtk_filename = "solution-global";

    vtk_filename += ".vtk";
    std::ofstream output (vtk_filename.c_str());

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");


    data_out.build_patches (fe->degree);
    data_out.write_vtk (output);

    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);

    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);

    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "$L^2$-error");
    convergence_table.set_tex_caption("H1", "$H^1$-error");

    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");

    std::cout << std::endl;
    convergence_table.write_text(std::cout);

    std::string error_filename = "error";

    error_filename += "-global";

    error_filename += ".tex";
    std::ofstream error_table_file(error_filename.c_str());

    convergence_table.write_tex(error_table_file);


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

    std::string conv_filename = "convergence";

    conv_filename += "-global";

    std::ofstream table_file(conv_filename.c_str());
    convergence_table.write_tex(table_file);

}
}


int main ()
{
    try
    {
        using namespace dealii;
        using namespace Step7;

        deallog.depth_console (0);

        {
            std::srand(0);
            FE_Q<dim> fe(polynomial_degree);
            LaplaceProblem<dim> helmholtz_problem_2d (fe);

            helmholtz_problem_2d.run ();

            std::cout << std::endl;
        }


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
