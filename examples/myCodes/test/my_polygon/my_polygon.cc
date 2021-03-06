#include <iostream>
#include <fstream>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <cmath>

//**********************************
//   1 x-----x-----x
//     |           |
//   0 x           x
//     |           |
//  -1 x-----x-----x
//     -1    0     1

class myPolygon
{
public:
    myPolygon(){ }
    void constructPolygon(const std::vector<dealii::Point<2>> point_list){
        for (unsigned int i = 0; i < point_list.size()-1; ++i)
        { segment my_segment;
            my_segment.beginPoint = point_list[i];
            my_segment.endPoint = point_list[i+1];
            my_segment.length = my_segment.beginPoint.distance(my_segment.endPoint);
            my_segment.normalVector = calculate_normals(my_segment);
            my_segment.q_points = calculate_q_points(my_segment);

            segment_list.push_back(my_segment);
            vertices_list.push_back(point_list[i]);
        }
    }

    void list_segments(){
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            std::cout<<my_segment.beginPoint<<std::endl;
        }
    }

    void save_segments(){ // save to text file for plotting
        std::remove("plot_poly");
        std::ofstream ofs_poly;
        ofs_poly.open ("plot_poly", std::ofstream::out | std::ofstream::app);
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            ofs_poly<<my_segment.beginPoint<<std::endl;
        }
        segment end_segment = segment_list[segment_list.size()-1];
        ofs_poly<<end_segment.endPoint<<std::endl;
        ofs_poly.close();
    }

    void save_q_points(){
        std::remove("plot_q_points");
        std::ofstream ofs_q_points;
        ofs_q_points.open ("plot_q_points", std::ofstream::out | std::ofstream::app);
        for (unsigned int i = 0; i <segment_list.size(); ++i)
        {
          segment my_segment = segment_list[i];
          for (unsigned int j = 0; j <my_segment.q_points .size(); ++j)
                ofs_q_points <<  my_segment.q_points[j] << std::endl;
        }
        ofs_q_points.close();
    }


private:
    struct segment{
        dealii::Point<2> beginPoint;
        dealii::Point<2> endPoint;
        double length;
        dealii::Point<2> normalVector;
        std::vector<dealii::Point<2>> q_points;
        std::vector<double> q_weights = {0.5000, 0.5000};
    };

    std::vector<dealii::Point<2>> calculate_q_points(const segment my_segment)
    {
        dealii::Point<2> q_point;
        std::vector<dealii::Point<2>> q_points;

        q_point = my_segment.beginPoint + ((my_segment.endPoint-my_segment.beginPoint) * 0.211325);
        q_points.insert(q_points.end(), q_point);
        q_point = my_segment.beginPoint + ((my_segment.endPoint-my_segment.beginPoint) * 0.788675);
        q_points.insert(q_points.end(), q_point);

        return q_points;
    }

    dealii::Point<2> calculate_normals (const segment my_segment) // it needs to be checked whether they are pointing outwards
    {
        double dx, dy;
        dx = (my_segment.beginPoint[0] - my_segment.endPoint[0]) / my_segment.length;
        dy = (my_segment.beginPoint[1] - my_segment.endPoint[1]) / my_segment.length;
        std::cout<<-dx<<" "<< dy<<std::endl;
        return {-dx, dy};
    }

    bool is_inside(const dealii::Point<2> p){
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];

        }
    }

    std::vector<segment> segment_list;
    std::vector<dealii::Point<2>> vertices_list;
};

int main()
{
    myPolygon my_poly;
    std::vector<dealii::Point<2>> point_list;
    point_list = {{0,1}, {1,1}, {1,0}, {1,-1}, {0,-1}, {-1,-1}, {-1,0}, {-1,1}, {0,1}};
    my_poly.constructPolygon(point_list);
    my_poly.list_segments();
    my_poly.save_segments();
    my_poly.save_q_points();

    return 0;
}



