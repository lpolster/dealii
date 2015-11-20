#include <iostream>
#include <fstream>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>


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
    boost::geometry::model::polygon<point> poly;
    boost::geometry::read_wkt("POLYGON((0 0,0 4,4 0,0 0))", poly);
    boost::geometry::for_each_point(poly, list_coordinates<point>);

    return 0;
}



