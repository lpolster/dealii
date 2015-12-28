
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
#include <deal.II/fe/mapping.h>


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
    void refine_grid();
    
    Triangulation<2>     triangulation;                     // triangulation for the solution grid
    FE_Q<2>              fe;                                // fe for the solution grid
    DoFHandler<2>        dof_handler;                       // dof handler for the solution grid
    
    Triangulation<2>   triangulation_adaptiveIntegration;   // triangulation for the integration grid
    FE_Q<2>            fe_adaptiveIntegration;              // fe for the integration grid
    DoFHandler<2>      dof_handler_adaptiveIntegration;     // dof handler for the integration grid
    
    ConstraintMatrix     constraints;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;                     // system/stiffness matrix
    
    Vector<double>       solution;                          // solution/coefficent vector
    Vector<double>       system_rhs;                        // the right hand side
    
    myPolygon            my_poly;                           // the polygon boundary
};

Step3::Step3 ()
:
fe (1),                                                     // bilinear
dof_handler (triangulation),
fe_adaptiveIntegration (1),                                 // bilinear
dof_handler_adaptiveIntegration (triangulation_adaptiveIntegration)
{}


void Step3::make_grid ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);       // generate triangulation for solution grid
    triangulation.refine_global (global_refinement_level);
    
    GridGenerator::hyper_cube (triangulation_adaptiveIntegration, -1, 1); // generate triangulation for integration grid
    triangulation_adaptiveIntegration.refine_global (global_refinement_level);
    
    std::cout << "Number of active cells: "
    << triangulation.n_active_cells()
    << std::endl;
}

void Step3::setup_boundary ()                               // setup the polygon boundary
{
    std::vector<dealii::Point<2>> point_list;
//    point_list = {{0.0,1.0}, {0.5, 1.0}, {1.0,1.0}, {1.0, 0.5}, {1.0,0.0}, {1.0, -0.5}, {1.0,-1.0}, {0.5, -1.0},
//                  {0.0,-1.0}, {-0.5, -1.0}, {-1.0,-1.0}, {-1.0, -0.5}, {-1.0,0.0}, {-1.0, 0.5}, {-1.0,1.0}, {-0.5, 1}, {0.0,1.0}};
    point_list = {{0.0,1.0}, {0.25, 1.0}, {0.5, 1.0}, {0.75, 1.0}, {1.0,1.0}, {1.0, 0.75}, {1.0, 0.5}, {1.0, 0.25}, {1.0,0.0}, {1.0, -0.25},
                  {1.0, -0.5}, {1.0, -0.75}, {1.0,-1.0}, {0.75, -1.0}, {0.5, -1.0}, {0.25, -1.0},
                  {0.0,-1.0}, {-0.25, -1.0}, {-0.5, -1.0}, {-0.75, -1.0}, {-1.0,-1.0}, {-1.0, -0.75},
                  {-1.0, -0.5}, {-1.0, -0.25}, {-1.0,0.0}, {-1.0, 0.25}, {-1.0, 0.5}, {-1.0, 0.75}, {-1.0,1.0},
                  {-0.75, 1.0}, {-0.5, 1.0}, {-0.25, 1.0}, {0.0,1.0}};

//    point_list = {{0.0,1.0}, {1.0,1.0}, {1.0,0.0}, {1.0,-1.0}, {0.0,-1.0}, {-1.0,-1.0}, {-1.0,0.0}, {-1.0,1.0}, {0.0,1.0}};
    my_poly.constructPolygon(point_list);                   // construct polygon from list of points
    //    my_poly.list_segments();
    my_poly.save_segments();
    my_poly.save_q_points();
}

void Step3::setup_system ()
{
    dof_handler.distribute_dofs (fe);
    dof_handler_adaptiveIntegration.distribute_dofs(fe_adaptiveIntegration);
    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
    
    constraints.clear ();
    DoFTools::make_hanging_node_constraints (dof_handler,
                                             constraints);
    // Toggle comment for fcm
    //    VectorTools::interpolate_boundary_values (dof_handler,
    //                                              0,
    //                                              ZeroFunction<2>(),
    //                                              constraints);
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
    Quadrature<2> collected_quadrature_on_boundary_segment;           // quadrature rule on boundary
    
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);    // the cell matrix
    Vector<double>       cell_rhs (dofs_per_cell);                      // the cell right hand side
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    DoFHandler<2>::active_cell_iterator cell = dof_handler.begin_active(), endc = dof_handler.end();
    
    for (; cell!=endc; ++cell) // iterate over all cells of solution grid
    {
        cell_matrix = 0;
        cell_rhs = 0;
        
        // colect quadrature on the cell in solution grid
        collected_quadrature = collect_quadratures(topological_equivalent(cell, triangulation_adaptiveIntegration), &quadrature_formula);
        
        // man kann denke ich auch ohne fe values arbeiten...
        FEValues<2> fe_values(fe, collected_quadrature, update_quadrature_points |  update_gradients | update_JxW_values |  update_values);
        
        fe_values.reinit(cell);                        // reinitialize fe values on current cells
        
        plot_in_global_coordinates(fe_values.get_quadrature().get_points(), cell, "collected_quadrature");
        
        // get values of indicator function (alpha)
        std::vector<double> indicator_function_values(collected_quadrature.size());
        indicator_function_values = get_indicator_function_values(fe_values.get_quadrature().get_points(), cell, my_poly);
        
//        double weight_counter = 0.0;

        for (unsigned int q_index=0; q_index<collected_quadrature.size(); ++q_index) // loop over all quadrature points in that cell
        {
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                for (unsigned int j=0; j<dofs_per_cell; ++j)
                    cell_matrix(i,j) += (fe_values.shape_grad (i, q_index) *        // assemble cell matrix
                                         fe_values.shape_grad (j, q_index) *
                                         indicator_function_values[q_index] *
                                         fe_values.JxW (q_index));
            
            for (unsigned int i=0; i<dofs_per_cell; ++i)
                cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                                indicator_function_values[q_index] *                // assemble cell right hand side
                                1 *
                                fe_values.JxW (q_index));

//            weight_counter += fe_values.get_quadrature().get_weights()[q_index];
//            std::cout<<"weight = "<<fe_values.get_quadrature().get_weights()[q_index]<<std::endl;
        }

        // Assertion for sum of weihts in cell (= 0.0)
//        std::cout<<"Sum of weights in cell: "<<weight_counter<<std::endl;
        
        if (contains_boundary(cell, my_poly))
        {
            std::vector<int> segment_indices = my_poly.get_segment_indices_inside_cell(cell);
//            std::cout<<"No. of segment indices: "<<segment_indices.size()<<std::endl;

            for (unsigned int k = 0; k < segment_indices.size(); ++ k){

                std::cout<<"New segment..."<<std::endl;

                myPolygon::segment my_segment = my_poly.segment_list[segment_indices[k]];

                // Nitsche method
                
                collected_quadrature_on_boundary_segment = collect_quadratures_on_boundary_segment(my_segment, cell);
                
                FEValues<2> fe_values_on_boundary_segment (fe, collected_quadrature_on_boundary_segment, update_quadrature_points |  update_gradients |  update_values | update_JxW_values);
                
                fe_values_on_boundary_segment.reinit(cell);
                
                for (unsigned int q_index=0; q_index<my_segment.q_points.size(); ++q_index)
                {
                    for (unsigned int i=0; i<dofs_per_cell; ++i)  { // loop over degrees of freedom
                        for (unsigned int j=0; j<dofs_per_cell; ++j)  {// loop over degrees of freedom
                                                        
                            cell_matrix(i,j) -= (fe_values_on_boundary_segment.shape_value(i,q_index) *
                                                 my_segment.normalVector *
                                                 fe_values_on_boundary_segment.shape_grad(j,q_index) * my_segment.length * //fe_values_on_boundary_segment.JxW (q_index));
                                                 fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);

                            cell_matrix(i,j) -= (fe_values_on_boundary_segment.shape_value(j,q_index) *
                                                 my_segment.normalVector *
                                                 fe_values_on_boundary_segment.shape_grad(i,q_index) *
                                                 my_segment.length*
                                                 fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
                            
                            cell_matrix(i,j) +=  beta_h * (fe_values_on_boundary_segment.shape_value(i,q_index) *
                                                           fe_values_on_boundary_segment.shape_value(j,q_index) *
                                                           my_segment.length *
                                                           fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
                            
                        } // endfor
                        cell_rhs(i) -= (dirichlet_boundary_value * fe_values_on_boundary_segment.shape_grad(i,q_index) *
                                        my_segment.normalVector * my_segment.length *
                                        fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
                        cell_rhs(i) +=  (beta_h * fe_values_on_boundary_segment.shape_value(i,q_index) *
                                         dirichlet_boundary_value * my_segment.length *
                                         fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]);
                    } // endfor

                    std::cout<<fe_values_on_boundary_segment.get_quadrature().get_points()[q_index]<<" "<<
                               fe_values_on_boundary_segment.get_quadrature().get_weights()[q_index]<<std::endl;
                    
                } // endfor
//                std::cout<<"Cell nr. "<<cell->index()<<"contains segment "<<segment_indices[k]<<std::endl;
            }
            
        } // endfor
        
        
        cell->get_dof_indices (local_dof_indices);
        
        constraints.distribute_local_to_global (cell_matrix,
                                                cell_rhs,
                                                local_dof_indices,
                                                system_matrix,
                                                system_rhs);
    }
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
    data_out.build_patches (); // linear interpolation for plotting
    
    std::ofstream output_gpl ("solution.gpl");
    data_out.write_gnuplot (output_gpl);
    
    std::ofstream output_vtk ("solution.vtk");
    data_out.write_vtk (output_vtk);
}


void output_grid(const dealii::Triangulation<2>& tria,
                 std::string name,
                 const unsigned int nr)
{
    GridOut grid_out;
    std::stringstream filename;
    filename << name << "-" << nr << ".svg";
    std::ofstream out(filename.str());
    grid_out.write_svg(tria, out);
}

void Step3::refine_grid ()
{
    // Create a vector of floats that contains information about whether the cell contains the boundary or not
    
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


void Step3::run ()
{
    make_grid ();
    setup_boundary ();
    
    for (unsigned int i = 0; i < refinement_cycles; i++)
    {
        refine_grid ();
        output_grid(triangulation_adaptiveIntegration, "adaptiveGrid", i);
        
    }
    
    setup_system ();
    assemble_system ();
    solve ();
    output_results ();
}


int main ()
{
    std::remove("indicator_function_values");
    std::remove("collected_quadrature");


    Step3 laplace_problem;
    laplace_problem.run ();
    
    return 0;
}
