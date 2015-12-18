
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
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_direct.h>


#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
#include "mypolygon.h"
#include "fcm-tools.h"
#include "find_cells.h"


using namespace dealii;


class Step3
{
public:
    Step3 ();
    
    void run ();
    
private:
    void make_grid ();
    void setup_system ();
    void setup_boundary ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    
    Triangulation<2>     triangulation;
    FE_Q<2>              fe;
    DoFHandler<2>        dof_handler;
    
    Triangulation<2>   triangulation_adaptiveIntegration; // triangulation for the integration grid
    FE_Q<2>            fe_adaptiveIntegration; // fe for the integration grid
    DoFHandler<2>      dof_handler_adaptiveIntegration; // dof handler for the integration grid
    
    ConstraintMatrix     constraints;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    
    Vector<double>       solution;
    Vector<double>       system_rhs;
    
    myPolygon            my_poly;
};

Step3::Step3 ()
:
fe (1),
dof_handler (triangulation),
fe_adaptiveIntegration (1),
dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration)
{}


void Step3::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -2, 2);
    triangulation.refine_global (1);
    
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -2, 2);
    triangulation_adaptiveIntegration.refine_global (2);
    
    std::cout << "Number of active cells: "
    << triangulation.n_active_cells()
    << std::endl;
}

void Step3::setup_boundary ()
{
    std::vector<dealii::Point<2>> point_list;
    point_list = {{0.0,1.0}, {1.0,1.0}, {1.0,0.0}, {1.0,-1.0}, {0.0,-1.0}, {-1.0,-1.0}, {-1.0,0.0}, {-1.0,1.0}, {0.0,1.0}};
    my_poly.constructPolygon(point_list);
    //    my_poly.list_segments();
    my_poly.save_segments();
    my_poly.save_q_points();
}

void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    dof_handler_adaptiveIntegration.distribute_dofs(fe_adaptiveIntegration);
    std::cout << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    // Toggle comment for fcm
    VectorTools::interpolate_boundary_values (dof_handler,
                                              0,
                                              ZeroFunction<2>(),
                                              constraints);
    constraints.close ();
    
    
    CompressedSparsityPattern c_sparsity(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    c_sparsity,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(c_sparsity);
    
    system_matrix.reinit (sparsity_pattern);
    
    solution.reinit (dof_handler.n_dofs());
    system_rhs.reinit (dof_handler.n_dofs());
}

void Step3::assemble_system ()
{
    QGauss<2>  quadrature_formula(2);
    Quadrature<2> collected_quadrature;                       // the quadrature rule
    Quadrature<2> collected_quadrature_on_boundary;           // quadrature rule on boundary
    std::vector<std::vector<double>> normal_vectors_list;
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
        
        collected_quadrature = collect_quadratures(topological_equivalent(cell, triangulation_adaptiveIntegration), &quadrature_formula);
        
        // man kann denke ich auch ohne fe values arbeiten...
        FEValues<2> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);
        
        fe_values.reinit(cell);                        // reinitialize fe values on current cells
        
        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), cell, "collected_quadrature");
        
        unsigned int n_q_points = collected_quadrature.size(); // number of quadrature points
        
        std::vector<double> indicator_function_values(n_q_points);
        
        indicator_function_values = get_indicator_function_values(fe_values.get_quadrature().get_points(), cell, my_poly);
        
        for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *
                                         fe_values.shape_grad (j, q_index) *
                                         indicator_function_values[q_index] *
                                         fe_values.JxW (q_index));
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                                indicator_function_values[q_index] *
                                1 *
                                fe_values.JxW (q_index));
        }
        
        cell->get_dof_indices (local_dof_indices);
        
        
        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
        
        if (contains_boundary(cell, my_poly))
        {
            std::vector<int> segment_indices = my_poly.get_segment_indices_inside_cell(cell);
            for (unsigned int i = 0; i < segment_indices.size(); ++i)
                std::cout<<"Cell nr. "<<cell->index()<<"contains segment "<<segment_indices[i]<<std::endl;

        }
        
        
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


void Step3::solve ()
{
    SparseDirectUMFPACK  A_direct;
    A_direct.initialize(system_matrix);
    A_direct.vmult (solution, system_rhs);
}


void Step3::output_results () const
{
    DataOut<2> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution, "solution");
    data_out.build_patches ();
    
    std::ofstream output ("solution.gpl");
    data_out.write_gnuplot (output);
}


void Step3::run ()
{
    make_grid ();
    setup_system ();
    setup_boundary ();
    assemble_system ();
    solve ();
    output_results ();
}


int main ()
{
    Step3 laplace_problem;
    laplace_problem.run ();
    
    return 0;
}
