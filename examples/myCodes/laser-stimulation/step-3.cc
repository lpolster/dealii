/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2015 by the deal.II authors
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
 * Authors: Wolfgang Bangerth, 1999,
 *          Guido Kanschat, 2011
 */


// @sect3{Many new include files}

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
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include <deal.II/lac/sparse_direct.h>

using namespace dealii;


class Step3
{
public:
    Step3 ();

    void run ();


private:
    void make_grid ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results () const;

    Triangulation<2>     triangulation;
    FE_Q<2>              fe;
    DoFHandler<2>        dof_handler;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
    Vector<double>       tmp;

};

class RightHandSide : public Function<2>
{
public:
    RightHandSide ()
        :
          Function<2>(),
          period (0.2)
    {}
    virtual double value (const Point<2> &p,
                          const unsigned int component = 0) const;
private:
    const double period;
};

double RightHandSide::value (const Point<2> &p,
                                const unsigned int component) const
{
    if ((p[0] > -0.1) && (p[0] < 0.1) && (p[1] > 0.999) && (p[1] < 1.0))
        return 1;
    else
        return 0;
}


Step3::Step3 ()
    :
      fe (1),
      dof_handler (triangulation)
{}

void Step3::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.begin_active()->face(0)->set_boundary_id(1);
    triangulation.refine_global (4);

    std::cout << "Number of active cells: "
              << triangulation.n_active_cells()
              << std::endl;
}


void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    std::cout << "Number of degrees of freedom: "
              << dof_handler.n_dofs()
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern (dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_matrix.reinit (sparsity_pattern);

    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());

    tmp.reinit (solution.size()); //...

}


void Step3::assemble_system ()
{

    RightHandSide rhs_function; // .....
    VectorTools::create_right_hand_side(dof_handler,QGauss<2>(fe.degree+1), rhs_function, tmp);   //......

    QGauss<2>  quadrature_formula(2);

    FEValues<2> fe_values (fe, quadrature_formula,
                           update_values | update_gradients | update_JxW_values);


    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();


    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);


    DoFHandler<2>::active_cell_iterator
            cell = dof_handler.begin_active(),
            endc = dof_handler.end();
    for (; cell!=endc; ++cell)
    {

        fe_values.reinit (cell);

        cell_matrix = 0;
        cell_rhs = 0;

        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                         fe_values.shape_grad (j, q_index) *
                                         fe_values.JxW (q_index));

            for (unsigned int i=0; i<dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                                1 *
                                fe_values.JxW (q_index));
        }

        cell->get_dof_indices (local_dof_indices);

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
                system_matrix.add (local_dof_indices[i],
                                   local_dof_indices[j],
                                   cell_matrix(i,j));

        for (unsigned int i=0; i<dofs_per_cell; ++i)
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }

    std::map<types::global_dof_index,double> boundary_values;
    VectorTools::interpolate_boundary_values (dof_handler,
                                              1,
                                              ConstantFunction<2> (1),
                                              boundary_values);

    MatrixTools::apply_boundary_values (boundary_values,
                                        system_matrix,
                                        solution,
                                        system_rhs);
}


// @sect4{Step3::solve}

// The following function simply solves the discretized equation. As the
// system is quite a large one for direct solvers such as Gauss elimination or
// LU decomposition, we use a Conjugate Gradient algorithm. You should
// remember that the number of variables here (only 1089) is a very small
// number for finite element computations, where 100.000 is a more usual
// number.  For this number of variables, direct methods are no longer usable
// and you are forced to use methods like CG.
void Step3::solve ()
{
    // First, we need to have an object that knows how to tell the CG algorithm
    // when to stop. This is done by using a SolverControl object, and as
    // stopping criterion we say: stop after a maximum of 1000 iterations (which
    // is far more than is needed for 1089 variables; see the results section to
    // find out how many were really used), and stop if the norm of the residual
    // is below $10^{-12}$. In practice, the latter criterion will be the one
    // which stops the iteration:
    //  SolverControl           solver_control (1000, 1e-12);
    // Then we need the solver itself. The template parameters to the SolverCG
    // class are the matrix type and the type of the vectors, but the empty
    // angle brackets indicate that we simply take the default arguments (which
    // are <code>SparseMatrix@<double@></code> and
    // <code>Vector@<double@></code>):
    //  SolverCG<>              solver (solver_control);

    // Now solve the system of equations. The CG solver takes a preconditioner
    // as its fourth argument. We don't feel ready to delve into this yet, so we
    // tell it to use the identity operation as preconditioner:
    //  solver.solve (system_matrix, solution, system_rhs,
    //                PreconditionIdentity());
    // Now that the solver has done its job, the solution variable contains the
    // nodal values of the solution function.

    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
}


// @sect4{Step3::output_results}

// The last part of a typical finite element program is to output the results
// and maybe do some postprocessing (for example compute the maximal stress
// values at the boundary, or the average flux across the outflow, etc). We
// have no such postprocessing here, but we would like to write the solution
// to a file.
void Step3::output_results () const
{
    // To write the output to a file, we need an object which knows about output
    // formats and the like. This is the DataOut class, and we need an object of
    // that type:
    DataOut<2> data_out;
    // Now we have to tell it where to take the values from which it shall
    // write. We tell it which DoFHandler object to use, and the solution vector
    // (and the name by which the solution variable shall appear in the output
    // file). If we had more than one vector which we would like to look at in
    // the output (for example right hand sides, errors per cell, etc) we would
    // add them as well:
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    // After the DataOut object knows which data it is to work on, we have to
    // tell it to process them into something the back ends can handle. The
    // reason is that we have separated the frontend (which knows about how to
    // treat DoFHandler objects and data vectors) from the back end (which knows
    // many different output formats) and use an intermediate data format to
    // transfer data from the front- to the backend. The data is transformed
    // into this intermediate format by the following function:
    data_out.build_patches ();

    // Now we have everything in place for the actual output. Just open a file
    // and write the data into it, using GNUPLOT format (there are other
    // functions which write their data in postscript, AVS, GMV, or some other
    // format):
    std::ofstream output ("solution.gpl");
    data_out.write_gnuplot (output);
}


// @sect4{Step3::run}

// Finally, the last function of this class is the main function which calls
// all the other functions of the <code>Step3</code> class. The order in which
// this is done resembles the order in which most finite element programs
// work. Since the names are mostly self-explanatory, there is not much to
// comment about:
void Step3::run ()
{
    make_grid ();
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
}


// @sect3{The <code>main</code> function}

// This is the main function of the program. Since the concept of a main
// function is mostly a remnant from the pre-object era in C/C++ programming,
// it often does not much more than creating an object of the top-level class
// and calling its principle function. This is what is done here as well:
int main ()
{
    Step3 laplace_problem;
    laplace_problem.run ();

    return 0;
}
