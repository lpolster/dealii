#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

#include "mypolygon.h"
#include "fcm-tools.h"

//#define MY_DEBUG_DEF

using namespace dealii;

class createPolygon
{
public:
    createPolygon();
    void run();
private:
    void refine_grid (std::vector<dealii::Point<2>> point_list);

    Triangulation<2>     triangulation;                     // triangulation for the solution grid
    FE_Q<2>              fe;                                // fe for the solution grid
    DoFHandler<2>        dof_handler;                       // dof handler for the solution grid
    myPolygon            my_poly;                           // the polygon boundary
};

createPolygon::createPolygon()
    :
      fe (1),                                                     // bilinear
      dof_handler (triangulation)
{}

void createPolygon::refine_grid (std::vector<dealii::Point<2>> point_list)
{
    myPolygon  my_poly;
    my_poly.constructPolygon(point_list);                   // construct polygon from list of points
    typename DoFHandler<2>::active_cell_iterator // an iterator over all active cells
            cell = dof_handler.begin_active(), // the first active cell
            endc = dof_handler.end(); // one past the last active cell

    for (; cell!=endc; ++cell) // loop over all active cells
    {
        if (contains_boundary(cell, my_poly))
        {
            cell -> set_refine_flag();
        }

    }

    triangulation.execute_coarsening_and_refinement ();
}

void createPolygon::run()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);


    std::vector<dealii::Point<2>> point_list;
    point_list = {{0.1,0.9}, {0.6, -0.4}, {-0.2, -0.8}, {0.1,0.9}};
    //    point_list = {{-0.6,0.6}, {0.7, 0.7}, {-0.8,-0.8}, {-0.6, 0.6}}; //{0.4, -0.4},

    //    point_list = {{-0.4,0.4}, {0.4, 0.4},{0.4, -0.4}, {-0.4,-0.4}, {-0.4,0.4}};

    //    point_list = {{-0.4,0.4}, {0.4, 0.4}, {-0.4,-0.4}, {-0.4,0.4}};

    for (unsigned int i = 0; i < 3; i++)
    {

        triangulation.refine_global (1);

        point_list = update_point_list(point_list, triangulation);
    }

    GridOut grid_out;
    std::stringstream filename;
    filename << "grid" << ".svg";
    std::ofstream out(filename.str());
    grid_out.write_svg(triangulation, out);

    dof_handler.distribute_dofs (fe);

    for (unsigned int i = 0; i<4; i++)
    {
        refine_grid( point_list);
        point_list = update_point_list(point_list, triangulation);
        GridOut grid_out;
        std::stringstream filename;
        filename << "grid - " <<i<< ".svg";
        std::ofstream out(filename.str());
        grid_out.write_svg(triangulation, out);
    }

    my_poly.constructPolygon(point_list);                   // construct polygon from list of points

    //my_poly.list_segments();
    my_poly.save_segments();

}


int main()
{
    createPolygon polygon_creator;
    polygon_creator.run();

#ifdef MY_DEBUG_DEF
    std::cout << "This is only printed if MY_DEBUG_DEF is defined\n";
#endif

    return 0;
}
