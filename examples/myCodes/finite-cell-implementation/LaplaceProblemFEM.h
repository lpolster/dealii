//
//  LaplaceProblemFEM.h
//  
//
//  Created by Lisa on 04/04/16.
//
//

#ifndef LAPLACEPROBLEMFEM_H
#define LAPLACEPROBLEMFEM_H


using namespace dealii;

class LaplaceProblemFEM
{
public:
    LaplaceProblemFEM ();
    ~LaplaceProblemFEM ();
    
    void run ();
    
    
private:
    void make_grid ();
    void setup_system ();
    void assemble_system ();
    void solve ();
    void output_results () const;
    void refine_grid_globally ();
    void process_solution (const unsigned int cycle);
    void output_grid(const dealii::Triangulation<2>& tria,
                     std::string name,
                     const unsigned int nr1);
    
    Triangulation<2>     triangulation;
    FE_Q<2>              fe;
    DoFHandler<2>        dof_handler;
    
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
    
    Vector<double>       solution;
    Vector<double>       system_rhs;
    
    ConvergenceTable     convergence_table;
};



#endif 
