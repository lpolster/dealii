
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <iostream>
#include <fstream>
#include <cmath>

using namespace dealii;

void first_grid ()
{
  Triangulation<2> triangulation;
  GridGenerator::hyper_cube (triangulation, 0, 1);
  triangulation.refine_global (2);
  std::ofstream out ("grid-1.eps");
  GridOut grid_out;
  grid_out.write_eps (triangulation, out);
  std::cout << "Grid written to grid-1.eps" << std::endl;
  MappingQ1<2> mapping;
  std::pair<Triangulation<2>::active_cell_iterator, Point<2> > result;
  Point<2> q_point = {0,0.9};

  result = GridTools::find_active_cell_around_point (mapping,triangulation, q_point);
        
  std::cout << "index: " << result.first->index() << " local coords: " << result.second << std::endl;
}



int main ()
{
  first_grid ();
}
