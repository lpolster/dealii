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
#include "FCMLaplace.h"


using namespace dealii;


int main ()
{
    std::remove("indicator_function_values");
    std::remove("collected_quadrature");
    std::remove("collected_quadrature_on_boundary");
    FCMLaplace laplace_problem2;
    laplace_problem2.run();

    return 0;
}
