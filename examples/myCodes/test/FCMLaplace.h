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

#ifndef FCMLAPLACE_H
#define FCMLAPLACE_H

class FCMLaplace
{
    typedef bool (FCMLaplace::*boundary_function)(double, double, const double);
    
public:
    FCMLaplace ();
    ~FCMLaplace ();

    void run ();

private:

    static bool
    cell_is_in_physical_domain (const typename DoFHandler<2>::cell_iterator &cell);
    static bool
    cell_is_in_fictitious_domain (const typename DoFHandler<2>::cell_iterator &cell);
    static bool
    cell_is_cut_by_boundary (const typename DoFHandler<2>::cell_iterator &cell);
    static bool
    cell_is_child (const typename DoFHandler<2>::cell_iterator &cell, const typename DoFHandler<2>::cell_iterator &solution_cell);

    void setup_system ();
    void assemble_system ();
    void solve ();
    void refine_grid ();
    void output_results ();

    void get_indicator_function_values(const std::vector<Point<2> > &points, std::vector<double> &indicator_function_values, typename DoFHandler<2>::cell_iterator solution_cell,
                                                                              boundary_function f);
    void set_material_ids(DoFHandler<2> &dof_handler, boundary_function f);
    std::vector<double> get_normal_vector(typename DoFHandler<2>::cell_iterator cell, boundary_function f);
    std::vector<Point<2>> get_boundary_quadrature_points(typename DoFHandler<2>::cell_iterator cell, boundary_function f);
    dealii::Tensor<1,2,double> get_normal_vector_at_q_point(std::vector<std::vector<double>> normal_vectors_list, unsigned int q_index);
    void write_solution_to_file (Vector<double> solution);
    Quadrature<2> collect_quadratures_pesser(typename dealii::Triangulation<2>::cell_iterator cell,
                                             const dealii::Quadrature<2>* base_quadrature);
    Quadrature<2> collect_quadrature(typename DoFHandler<2>::cell_iterator solution_cell,
                                       const Quadrature<2>* quadrature_formula);
    Quadrature<2> collect_quadrature_on_boundary(typename DoFHandler<2>::cell_iterator solution_cell);
    std::vector<std::vector<double>> collect_normal_vector_on_boundary(typename DoFHandler<2>::cell_iterator solution_cell);
    std::vector<dealii::Point<2> > map_to_global_coordinates (std::vector<Point<2>> q_points,
                                                              DoFHandler<2>::cell_iterator cell, std::string filename);
    Quadrature<2> map_quadrature_points_and_weights_to_reference_cell (std::vector<Point<2>> q_points, // quadrature points
                                    std::vector<double> q_weights, std::vector<unsigned int> refinement_level_vec, typename DoFHandler<2>::cell_iterator cell, std::string filename);
    void output_grid(const Triangulation<2>& tria, std::string name, const unsigned int nr);

    bool rectangle (double x, double y, const double length);
    bool circle (double x, double y, const double radius);

    Triangulation<2>   triangulation; // triangulation for the solution grid
    Triangulation<2>   triangulation_adaptiveIntegration; // triangulation for the integration grid

    DoFHandler<2>      dof_handler;   // dof handler for the solution grid
    DoFHandler<2>      dof_handler_adaptiveIntegration; // dof handler for the integration grid
    FE_Q<2>            fe;            // fe for the solution grid
    FE_Q<2>            fe_adaptiveIntegration; // fe for the integration grid

    ConstraintMatrix     constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;      // vector for the solution (coefficients of shape functions)
    Vector<double>       system_rhs;    // vector for the right hand side

    const double threshold = 0.500; // threshold
    const double triangulation_begin = -1.0;
    const double triangulation_end = 1.0;
    const unsigned int refinement_cycles = 1;
    const unsigned int global_refinement_level = 5;
    const float beta_h = 2.0/(1.0/ global_refinement_level);    // beta divided by h, 2.0/0.0625
    const float dirichlet_boundary_value = 0.000;               // value for Dirichlet boundary condition
    boundary_function m_boundaryFunction = NULL;


    enum
    {
        physical_domain_id,
        fictitious_domain_id,
        boundary_domain_id
    };

};

#endif

