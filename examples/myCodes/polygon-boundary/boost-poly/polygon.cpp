#include <iostream>
#include <fstream>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/segment.hpp>


template <typename Point>
void list_coordinates(Point const& p)
{
    using boost::geometry::get;
    std::ofstream ofs_poly;
    ofs_poly.open ("plot_poly", std::ofstream::out | std::ofstream::app);
    ofs_poly << get<0>(p) << " " << get<1>(p) << std::endl;
    ofs_poly.close();
}

int main()
{
    std::remove("plot_poly");
    typedef boost::geometry::model::d2::point_xy<double> point;
    typedef boost::geometry::model::segment<point> segment;
    segment AB( point(0.0,1.0), point(2.0,1.0) ); // segment containing high and low point
    boost::geometry::model::polygon<point> poly;
    boost::geometry::read_wkt("POLYGON((0 0,0 4,4 0,0 0))", poly);
    
    point p(4.0, 0.0);
    boost::geometry::model::d2::point_xy<double> p1(0.0, 1.0), p2(0.0, 2.0);
    std::cout << "Distance p1-p2 is: " << boost::geometry::distance(p1, p2) << std::endl;
    
    
    std::cout << "covered by: " << (boost::geometry::covered_by(p, poly) ? "yes" : "no") << std::endl; // includes boundary
    std::cout << "within: " << (boost::geometry::within(p, poly) ? "yes" : "no") << std::endl; // checks if fully inside (not on boundary)
    
    boost::geometry::for_each_point(poly, list_coordinates<point>);

    return 0;
}



