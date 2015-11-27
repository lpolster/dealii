/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2014 by the deal.II authors
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
#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_tools.h>
#include <fstream>
#include <iostream>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/segment.hpp>

using namespace dealii;

typedef boost::geometry::model::d2::point_xy<double> point;
typedef boost::geometry::model::polygon<point> poly;
typedef boost::geometry::model::segment<point> segment;
std::vector<point > vertices;
std::vector<dealii::Point<2> > q_points;
std::vector<double> q_weights;
Triangulation<2> triangulation;
FE_Q<2>              fe(1);
DoFHandler<2>        dof_handler(triangulation);
MappingQ1<2> mapping;

template <typename Point>
void list_coordinates(Point const& p)
{
    using boost::geometry::get;
    std::ofstream ofs_poly;
    ofs_poly.open ("plot_poly", std::ofstream::out | std::ofstream::app);
    ofs_poly << get<0>(p) << " " << get<1>(p) << std::endl;
    ofs_poly.close();
}

boost::geometry::model::polygon<point> get_boundary ()
{
    poly boundary;
    boost::geometry::read_wkt("POLYGON((-0.5 -0.5,0.5 -0.5,0.5 0.5,-0.5 0.5, -0.5 -0.5))", boundary);
    return boundary;
}

bool check_if_inside(poly boundary, point p)
{
    if (boost::geometry::within(p, boundary) == true)
        return true;
    else
        return false;
}

template <typename Point>
void get_vertices(Point const& p)
{
    using boost::geometry::get;
    vertices.insert(vertices.end(), point(get<0>(p), get<1>(p)));
}

// hier aus Quadraturregel und Vertices die Quadraturpunkte ausrechnen.
Quadrature<2> get_quadrature()
{
    Point<2> q_point;
    Point<2> vertex_i, vertex_ii;
    using boost::geometry::get;
    
    std::ofstream ofs_q_points;
    ofs_q_points.open ("plot_q_points", std::ofstream::out | std::ofstream::app);
    
    for (unsigned int i = 0; i < (vertices.size()-1); ++i)
    {
        vertex_i = {get<0>(vertices[i]), get<1> (vertices[i])};
        vertex_ii = {get<0>(vertices[i+1]), get<1> (vertices[i+1])};
        
        q_point = vertex_i+ ((vertex_ii-vertex_i)*0.211325);
        ofs_q_points <<  q_point[0] << " " << q_point[1] << std::endl;
        q_points.insert(q_points.end(), q_point);
        
        q_weights.insert(q_weights.end(), 0.50000);
        
        q_point = vertex_i+ ((vertex_ii-vertex_i)*0.788675);
        ofs_q_points <<  q_point[0] << " " << q_point[1] << std::endl;
        
        q_points.insert(q_points.end(), q_point);
        q_weights.insert(q_weights.end(), 0.50000);
        
        //std::cout<< get<0> (vertices[i]) << " " << get<1> (vertices[i]) << std::endl;
    }
    ofs_q_points.close();
    
    return Quadrature<2> (q_points, q_weights);
}

std::vector<std::vector<double>> get_normals() // normals still need to be normalized and it needs to be chekced whether they are pointing outwards
{
    Point<2> vertex_i, vertex_ii;
    double dx, dy;
    std::vector<std::vector<double>> normals;
    using boost::geometry::get;
    
    
    for (unsigned int i = 0; i < (vertices.size()-1); ++i)
    {
        vertex_i = {get<0>(vertices[i]), get<1> (vertices[i])};
        vertex_ii = {get<0>(vertices[i+1]), get<1> (vertices[i+1])};
        dx = get<0>(vertices[i+1]) - get<0>(vertices[i]);
        dy = get<1>(vertices[i+1]) - get<1>(vertices[i]);
        normals.insert(normals.end(), {-dx, dy});
    }
    return normals;
}


double integrate_over_bundary (MappingQ1<2> mapping, Quadrature<2> collected_quadrature_on_boundary)
{
    std::pair<Triangulation<2>::active_cell_iterator, Point<2> > result;
    const unsigned int   dofs_per_cell = fe.dofs_per_cell;
    const unsigned int   n_q_points    = collected_quadrature_on_boundary.size();
    
    for (unsigned int i = 0; i < n_q_points; ++i)
    {
        std::cout << "Quadrature point in global coordinates: "<<collected_quadrature_on_boundary.get_points()[i]<<std::endl;
        result = GridTools::find_active_cell_around_point (mapping,triangulation, collected_quadrature_on_boundary.get_points()[i]);
        
        // Quadrature<2> temp_quadrature = Quadrature<2> (result.second, 0.5000);
        
        Point<2> q_point_on_ref_cell;
        q_point_on_ref_cell = result.second;
        
        std::cout << "index: " << result.first->index() << " local coords: "
        << result.second << std::endl;
        std::cout << "point: " << q_points[i] << std::endl;
        
        for (unsigned int j=0; j<dofs_per_cell; ++j)  {                   // loop over degrees of freedom
            for (unsigned int k=0; k<dofs_per_cell; ++k)  {
                
                std::cout << "Value of shape function "<<k<<" at point (" << q_point_on_ref_cell<<") : "<< fe.shape_value(k,q_point_on_ref_cell)<<std::endl;
                std::cout << "Value of shape function "<<j<<" at point (" << q_point_on_ref_cell<<") : "<< fe.shape_value(j,q_point_on_ref_cell)<<std::endl;
                std::cout << "Gradient of shape function "<<k<<" at point (" << q_point_on_ref_cell<<") : "<< fe.shape_grad(k,q_point_on_ref_cell)<<std::endl;
                std::cout << "Gradient of shape function "<<j<<" at point (" << q_point_on_ref_cell<<") : "<< fe.shape_grad(j,q_point_on_ref_cell)<<std::endl;
            }
            
        }
    }
    
    
    //boost::geometry::for_each_point(boundary, get_quadrature_points<point>);
    //std::cout<<quadrature_formula.get_points()[0]<<" "<< quadrature_formula.get_points()[1]<<std::endl;
    return 1.0;
}


int main ()
{
    GridGenerator::hyper_cube (triangulation, -1, 1);
    triangulation.refine_global (2);
    dof_handler.distribute_dofs (fe);
    Quadrature<2> collected_quadrature_on_boundary;
    
    std::cout << "Number of active cells: "
    << triangulation.n_active_cells()
    << std::endl;
    
    std::cout << "Number of degrees of freedom: "
    << dof_handler.n_dofs()
    << std::endl;
    
    std::remove("plot_poly");
    std::remove("plot_q_points");
    
    //segment AB( point(0.0,1.0), point(2.0,1.0) ); // segment containing high and low point
    poly boundary = get_boundary();
    point p1(0.0, 1.0), p2(0.0, 2.0), p3(0.5, 0.05);
    // bool is_inside = check_if_inside(boundary, p3);
    std::cout << "within: " << (check_if_inside(boundary, p3) ? "yes" : "no") << std::endl; // checks if fully inside (not on boundary)
    std::cout << "Distance p1-p2 is: " << boost::geometry::distance(p1, p2) << std::endl; // distance between two points
    std::cout << "covered by: " << (boost::geometry::covered_by(p3, boundary) ? "yes" : "no") << std::endl; // includes boundary
    
    boost::geometry::for_each_point(boundary, get_vertices<point>);
    collected_quadrature_on_boundary = get_quadrature();
    double boundary_integral = integrate_over_bundary(mapping, collected_quadrature_on_boundary);
    boost::geometry::for_each_point(boundary, list_coordinates<point>);
    
    
    return 0;
}
