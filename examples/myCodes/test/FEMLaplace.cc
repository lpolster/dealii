#include "FEMLaplace.h"

using namespace dealii;

FEMLaplace::FEMLaplace ()
  :
  fe (1),
  dof_handler (triangulation)
{}

void FEMLaplace::make_grid ()
{
  GridGenerator::hyper_cube (triangulation, -1, 1);
  triangulation.refine_global (5);

  std::cout << "Number of active cells: "
            << triangulation.n_active_cells()
            << std::endl;
}

void FEMLaplace::setup_system ()
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
}

void FEMLaplace::assemble_system ()
{

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
                                            0,
                                            ZeroFunction<2>(),
                                            boundary_values);

  MatrixTools::apply_boundary_values (boundary_values,
                                      system_matrix,
                                      solution,
                                      system_rhs);
}

void FEMLaplace::solve ()
{

  SolverControl           solver_control (1000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());
}


void FEMLaplace::output_results () const
{

  DataOut<2> data_out;
  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution_FEM");
  data_out.build_patches ();
  std::ofstream output ("solution_FEM.gpl");
  data_out.write_gnuplot (output);
}

void FEMLaplace::run ()
{
  FEMLaplace::make_grid ();
  FEMLaplace::setup_system ();
  FEMLaplace::assemble_system ();
  FEMLaplace::solve ();
  FEMLaplace::output_results ();
}
