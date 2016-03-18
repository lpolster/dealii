const unsigned int global_refinement_level = 1;              // the level of global refininement (solution grid)
const unsigned int polynomial_degree = 1;
const unsigned int n_quadrature_points = polynomial_degree + 1; //  polynomial_degree + 1 quadrature points required to integrate polynomial of degree 2*polynomial_degree exactly
const int dim = 2;
const unsigned int n_adaptive_refinement_cycles = 0;                    // the number of cycles of adaptive refinement
const double alpha = 1e-5;
const dealii::MappingQ<2> mapping(polynomial_degree,true); 
const double lower_embedded_domain = -1;
const double upper_embedded_domain = 1;
