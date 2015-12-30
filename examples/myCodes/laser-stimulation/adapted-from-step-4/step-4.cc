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
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */


// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
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

// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;

// @sect3{The <code>Step4</code> class template}

// This is again the same <code>Step4</code> class as in the previous
// example. The only difference is that we have now declared it as a class
// with a template parameter, and the template parameter is of course the
// spatial dimension in which we would like to solve the Laplace equation. Of
// course, several of the member variables depend on this dimension as well,
// in particular the Triangulation class, which has to represent
// quadrilaterals or hexahedra, respectively. Apart from this, everything is
// as before.
template <int dim>
class Step4
{
public:
  Step4 ();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;

  Triangulation<dim>   triangulation;
  FE_Q<dim>            fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{ 
  if ((p[0] > -0.25)  && (p[0] < 0.25) && (p[1] > 0.9))
    return 0.060 / (0.5*0.1); // 60mW laser, area 0.1*0.5
  else
    return 0;
}


// As boundary values, we choose x*x+y*y in 2D, and x*x+y*y+z*z in 3D. This
// happens to be equal to the square of the vector from the origin to the
// point at which we would like to evaluate the function, irrespective of the
// dimension. So that is what we return:
template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return 0.0; // 20°C in K: 293.15
}

template <int dim>
Step4<dim>::Step4 ()
  :
  fe (1),
  dof_handler (triangulation)
{}


template <int dim>
void Step4<dim>::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (4);

  std::cout << "   Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: "
            << triangulation.n_cells()
            << std::endl;
}

// @sect4{Step4::setup_system}

// This function looks exactly like in the previous example, although it
// performs actions that in their details are quite different if
// <code>dim</code> happens to be 3. The only significant difference from a
// user's perspective is the number of cells resulting, which is much higher
// in three than in two space dimensions!
template <int dim>
void Step4<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  std::cout << "   Number of degrees of freedom: "
            << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern (dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
}


template <int dim>
void Step4<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);
  const RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              cell_matrix(i,j) += 0.6 * (fe_values.shape_grad (i, q_index) * // thermal capacity of water = 0.6
                                   fe_values.shape_grad (j, q_index) *
                                   fe_values.JxW (q_index));

            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
          }

      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          for (unsigned int j=0; j<dofs_per_cell; ++j)
            system_matrix.add (local_dof_indices[i],
                               local_dof_indices[j],
                               cell_matrix(i,j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  std::map<types::global_dof_index,double> boundary_values;
  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            BoundaryValues<dim>(), //  ZeroFunction<dim>()
                                            boundary_values);
  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}


template <int dim>
void Step4<dim>::solve ()
{
  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;
}


template <int dim>
void Step4<dim>::output_results () const
{
  DataOut<dim> data_out;

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");

  data_out.build_patches ();

  std::ofstream output (dim == 2 ?
                        "solution-2d.vtk" :
                        "solution-3d.vtk");
  data_out.write_vtk (output);
}



template <int dim>
void Step4<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;

  make_grid();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}


int main ()
{
  deallog.depth_console (0);
  {
    Step4<2> laplace_problem_2d;
    laplace_problem_2d.run ();
  }

  {
    Step4<3> laplace_problem_3d;
    laplace_problem_3d.run ();
  }

  return 0;
}
