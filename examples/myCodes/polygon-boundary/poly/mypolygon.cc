#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <cmath>
#include <iostream>
#include <fstream>


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
        {
            segment my_segment;
            my_segment.beginPoint = point_list[i];
            my_segment.endPoint = point_list[i+1];
            my_segment.length = my_segment.beginPoint.distance(my_segment.endPoint);
            my_segment.q_points = calculate_q_points(my_segment);

            segment_list.push_back(my_segment);
            vertices_list.push_back(point_list[i]);
        }
        for (unsigned int i = 0; i < segment_list.size()-1; ++i)
        {
            segment my_segment = segment_list[i];
            my_segment.normalVector = calculate_normals(my_segment);
            std::cout<<"Normal vector: "<<std::endl;
            std::cout<< my_segment.normalVector<<std::endl;
        }
    }
    
    void list_segments(){
        std::cout<<"Listing segments: "<<std::endl;
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            std::cout<<"["<<my_segment.beginPoint<<"]"<<" "<<"["<<my_segment.endPoint<<"]"<<std::endl;
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
        dealii::Point<2> normalVector = {-dy, dx};
        const dealii::Point<2> p = my_segment.beginPoint + (my_segment.endPoint - my_segment.beginPoint)/2.0 + normalVector*0.5;
        
        dealii::Point<2> p1 = {-0.5, 0.5}, q1 = {10.0, 0.5}, p2 = {-1.0, 0.0}, q2 = {-1.0, 1.0};
        doIntersect(p1, q1, p2, q2)? std::cout << "Yes\n": std::cout << "No\n";
        
        if (is_inside(p))
        {
            std::cout<<"Point "<<p<<" is inside."<<std::endl;
            return -normalVector;
        }
        else
        {
            std::cout<<"Point "<<p<<" is outside."<<std::endl;
            return normalVector;
        }
    }
    
    bool is_inside(const dealii::Point<2> p1){
        unsigned int counter = 0;
        const dealii::Point<2> q1 = {10.0, p1[1]};
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            dealii::Point<2> p2 = my_segment.beginPoint;
            dealii::Point<2> q2 = my_segment.endPoint;
            if (doIntersect(p1, q1, p2, q2))
            {
                std::cout<<"Segment "<<"["<<p1<<"]"<<" "<<"["<<q1<<"]"<<" intersects"<<"["<<p2<<"]"<<"["<<q2<<"]"<<std::endl;
                counter ++;
            }
        }
        std::cout<<"Counter: "<<counter<<std::endl;
        if (counter%2 == 0)
            return false;
        else
            return true;
    }
    
    //________________________________
    // from http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    
    // Given three colinear points p, q, r, the function checks if
    // point q lies on line segment 'pr'
    bool onSegment(dealii::Point<2> p, dealii::Point<2> q, dealii::Point<2> r)
    {
        if (q[0] <= std::max(p[0], r[0]) && q[0] >= std::min(p[0], r[0]) &&
            q[1] <= std::max(p[1], r[1]) && q[1] >= std::min(p[1], r[1]))
            return true;
        
        return false;
    }
    
    // To find orientation of ordered triplet (p, q, r).
    // The function returns following values
    // 0 --> p, q and r are colinear
    // 1 --> Clockwise
    // 2 --> Counterclockwise
    int orientation(dealii::Point<2> p, dealii::Point<2> q, dealii::Point<2> r)
    {
        // See http://www.geeksforgeeks.org/orientation-3-ordered-points/
        // for details of below formula.
        int val = (q[1] - p[1]) * (r[0] - q[0]) -
        (q[0] - p[0]) * (r[1] - q[1]);
        
        if (val == 0) return 0;  // colinear
        
        return (val > 0)? 1: 2; // clock or counterclock wise
    }
    
    // The main function that returns true if line segment 'p1q1'
    // and 'p2q2' intersect.
    bool doIntersect(dealii::Point<2> p1, dealii::Point<2> q1, dealii::Point<2> p2, dealii::Point<2> q2)
    {
        // Find the four orientations needed for general and
        // special cases
        int o1 = orientation(p1, q1, p2);
        int o2 = orientation(p1, q1, q2);
        int o3 = orientation(p2, q2, p1);
        int o4 = orientation(p2, q2, q1);
        
        // General case
        if (o1 != o2 && o3 != o4)
            return true;
        
        // Special Cases
        // p1, q1 and p2 are colinear and p2 lies on segment p1q1
        if (o1 == 0 && onSegment(p1, p2, q1)) return true;
        
        // p1, q1 and p2 are colinear and q2 lies on segment p1q1
        if (o2 == 0 && onSegment(p1, q2, q1)) return true;
        
        // p2, q2 and p1 are colinear and p1 lies on segment p2q2
        if (o3 == 0 && onSegment(p2, p1, q2)) return true;
        
        // p2, q2 and q1 are colinear and q1 lies on segment p2q2
        if (o4 == 0 && onSegment(p2, q1, q2)) return true;
        
        return false; // Doesn't fall in any of the above cases
    }
    
    //________________________________
    
    std::vector<segment> segment_list;
    std::vector<dealii::Point<2>> vertices_list;
};

int main()
{
    myPolygon my_poly;
    std::vector<dealii::Point<2>> point_list;
    point_list = {{0.0,1.0}, {1.0,1.0}, {1.0,0.0}, {1.0,-1.0}, {0.0,-1.0}, {-1.0,-1.0}, {-1.0,0.0}, {-1.0,1.0}, {0.0,1.0}};
    my_poly.constructPolygon(point_list);
    my_poly.list_segments();
    my_poly.save_segments();
    my_poly.save_q_points();
    
    return 0;
}



