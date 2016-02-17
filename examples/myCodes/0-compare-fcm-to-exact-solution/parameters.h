const unsigned int global_refinement_level = 1;              // the level of global refininement (solution grid)
const unsigned int polynomial_degree = 1;
//const float beta_h = 20.0/(1.0/ global_refinement_level);    // beta
//const float beta_h = (2.0 * polynomial_degree * (polynomial_degree+1))/(1.0/ global_refinement_level);    // beta divided by h, 2*p*(p+1)/h
const unsigned int refinement_cycles = 2;                    // the number of cycles of adaptive refinement
const double alpha = 1e-5;
const dealii::MappingQ<2> mapping(polynomial_degree,true); 
