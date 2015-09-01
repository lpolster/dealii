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

using namespace dealii;


template <int dim>
class Step6
{
public:
  Step6 ();
  ~Step6 ();

  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>   triangulation;

  DoFHandler<dim>      dof_handler;
  FE_Q<dim>            fe;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
};


template <int dim>
class Coefficient : public Function<dim>
{
public:
  Coefficient () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;

  virtual void value_list (const std::vector<Point<dim> > &points,
                           std::vector<double>            &values,
                           const unsigned int              component = 0) const;
};



template <int dim>
double Coefficient<dim>::value (const Point<dim> &p,
                                const unsigned int) const
{
  if (p.square() < 0.5*0.5)
    return 1;
  else
    return 1;
}



template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
  const unsigned int n_points = points.size();

  Assert (values.size() == n_points,
          ExcDimensionMismatch (values.size(), n_points));

  Assert (component == 0,
          ExcIndexRange (component, 0, 1));

  for (unsigned int i=0; i<n_points; ++i)
    {
      if (points[i].square() < 0.5*0.5)
        values[i] = 1;
      else
        values[i] = 1;
    }
}

template <int dim>
Step6<dim>::Step6 ()
  :
  dof_handler (triangulation),
  fe (2)
{}

template <int dim>
Step6<dim>::~Step6 ()
{
  dof_handler.clear();

}

template <int dim>
void Step6<dim>::setup_system ()
{
  dof_handler.distribute_dofs(fe);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);
}

template <int dim>
void Step6<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(3); // (p+1) quadrature points (p = polynomial degree)

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  const Coefficient<dim> coefficient;
  std::vector<double>    coefficient_values (n_q_points);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();


  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      coefficient.value_list (fe_values.get_quadrature_points(),
                              coefficient_values);

      std::ofstream ofs;
        ofs.open ("Quadrature Points", std::ofstream::out | std::ofstream::app);

      for (unsigned int ii = 0; ii < n_q_points; ii++ )
        ofs << fe_values.get_quadrature_points()[ii] << " " << coefficient_values[ii] << std::endl;
       ofs.close();

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index){
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
            for (unsigned int j=0; j<dofs_per_cell; ++j){
              cell_matrix(i,j) += (coefficient_values[q_index] *
                                   fe_values.shape_grad(i,q_index) *
                                   fe_values.shape_grad(j,q_index) *
                                   fe_values.JxW(q_index));
//             std::cout<<"Cell Matrix (i,j) = "<<cell_matrix(i,j)<<std::endl;

            cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                            1.0 *
                            fe_values.JxW(q_index));
//            std::cout<<"Cell RHS (i) = "<<cell_rhs(i) <<std::endl;
            }

          }
      }

      cell->get_dof_indices (local_dof_indices);

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j){
//            std::cout << "Indices: ["<< local_dof_indices[i] << ", " << local_dof_indices[j] << "]" << std::endl;
          system_matrix.add (local_dof_indices[i],
                             local_dof_indices[j],
                             cell_matrix(i,j));
        }

      for (unsigned int i=0; i<dofs_per_cell; ++i)
        system_rhs(local_dof_indices[i]) += cell_rhs(i);

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

}

template <int dim>
void Step6<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

}

template <int dim>
void Step6<dim>::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    Vector<float> contains_boundary (triangulation.n_active_cells());
    unsigned int i = 0; // integer for accessing the right cell
    unsigned int counter; // integer to count the numer of vertices that are located in the physical domain

    typename DoFHandler<dim>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler.begin_active(), // the first active cell
            endc = dof_handler.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        counter = 0; // the counter is set to zero
        //std::cout<<cell<<std::endl; // prints what cell we are currently in

        // We loop over all 4 vertices (0-3) of the current cell. Vertex n of a cell is accessed by cell->vertex(n).
        // The x and y coordinate of a vertex are accessed by cell->vertex(n)[0] and cell->vertex(n)[1].
        // The if-statement test whether the vertex is located within a radius of 1.0, i.e. the physical domain.
        // The amount of vertices of a cell that are located in the physical domain are counter (increment counter).
        for (unsigned int vertex_iterator = 0; vertex_iterator < 4; vertex_iterator ++){
            if ((cell->vertex(vertex_iterator).square()) < 0.5*0.5){
                counter ++;
            }
        }

        // If the counter of vertices of a cell equals 0, the cell is located completely in the fictitious domain.
        // If the counter of vertices of a cell equals 4, the cell is located completely inside the physical domain.
        // Values in the interval [1,3] indicate that the boundary cuts through the cell.
        if (counter == 0)
            contains_boundary[i] = 0.0;
        else if (counter == 4)
            contains_boundary[i] = 0.5;
        else
            contains_boundary[i] = 1.0;

        //std::cout<<counter<<std::endl; // print the value of counter
        //std::cout<<contains_boundary[i]<<std::endl; // print the value that indicates whether cell contains boundary
        i++;
    }

    GridRefinement::refine (triangulation,contains_boundary, 0.9,3000);

    triangulation.execute_coarsening_and_refinement ();
}


template <int dim>
void Step6<dim>::output_results (const unsigned int cycle) const
{
  Assert (cycle < 3, ExcNotImplemented());

  std::string filename_integrationGrid = "integration_grid-";
  filename_integrationGrid += ('0' + cycle);
  filename_integrationGrid += ".eps";

  std::ofstream output_integrationGrid (filename_integrationGrid.c_str());
  GridOut grid_out_integrationGrid;
  grid_out_integrationGrid.write_eps (triangulation, output_integrationGrid);

  std::string filename_solutionGrid = "solution_grid-";
  filename_solutionGrid += ('0' + cycle);
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

//  std::ofstream output2 ("solution.vtk");
//  data_out.write_vtk (output2);

}

template <int dim>
void Step6<dim>::run ()
{
  for (unsigned int cycle=0; cycle<2; ++cycle)
    {
      std::cout << "Cycle " << cycle << ':' << std::endl;

      if (cycle == 0)
        {
          GridGenerator::hyper_cube (triangulation);
          triangulation.refine_global (3);

        }
      else
        refine_grid ();
      setup_system ();
      assemble_system ();
      solve ();
    }

  DataOutBase::EpsFlags eps_flags;
  eps_flags.z_scaling = 4;

  DataOut<dim> data_out;
  data_out.set_flags (eps_flags);

  data_out.attach_dof_handler (dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();

  std::ofstream output ("solution.gpl");
  data_out.write_gnuplot (output);

  std::ofstream output2 ("final-solution.eps");
  data_out.write_eps (output2);
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
