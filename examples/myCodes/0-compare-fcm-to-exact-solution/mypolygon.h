#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>

#include <assert.h>
#include <cmath>
#include <iostream>
#include <fstream>


dealii::QGauss<1>  boundary_quadrature(n_quadrature_points); // quadrature on boundary

class myPolygon
{
public:
    struct segment{
        dealii::Point<2> beginPoint;
        dealii::Point<2> endPoint;
        double length;
        dealii::Point<2> normalVector;
        std::vector<dealii::Point<2>> q_points;
        std::vector<double> q_weights;
    };
    std::vector<segment> segment_list;
    
    myPolygon(){ }

    void constructPolygon(const std::vector<dealii::Point<2>> point_list){
        segment_list.clear();
        for (unsigned int i = 0; i < point_list.size()-1; ++i)
        {
            segment my_segment;
            my_segment.beginPoint = point_list[i];
            my_segment.endPoint = point_list[i+1];
            my_segment.length = std::abs(my_segment.beginPoint.distance(my_segment.endPoint)); assert(my_segment.length > 0);
            my_segment.q_points = calculate_q_points(my_segment);
            my_segment.q_weights = calculate_q_weights();
            my_segment.normalVector = calculate_normals(my_segment);
            
            segment_list.push_back(my_segment);
            
            //std::cout << "Segment: [" <<my_segment.beginPoint<<"], ["<<my_segment.endPoint<<"]";
            //std::cout << ", normal vector: "<< my_segment.normalVector<<std::endl;
        }
    }
    
    double scalar_product(const dealii::Point<2> a, const dealii::Point<2> b) const
    {
        double product = 0;
        for (unsigned int i = 0; i < 2; i++)
            for (unsigned int i = 0; i < 2; i++)
                product = product + (a[i])*(b[i]);
        return product;
    }
    
    void list_segments() const // plot to console
    {
        std::cout<<"Listing segments: "<<std::endl;
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            std::cout<<"["<<my_segment.beginPoint<<"]"<<" "<<"["<<my_segment.endPoint<<"]"<<std::endl;
        }
    }
    
    void save_segments() const // save to text file for plotting
    {
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

//    int point_in_polygon(const dealii::Point<dim> p){ // https://de.wikipedia.org/wiki/Punkt-in-Polygon-Test_nach_Jordan // not working yet
//        int return_value = -1;
//        for (unsigned int i = 0; i <segment_list.size(); ++i){
//            segment my_segment = segment_list[i];
//            return_value = return_value * cross_prod_test(p, my_segment.beginPoint, my_segment.endPoint);
//        }
//    return return_value;
//    }
//
//    int cross_prod_test(dealii::Point<dim> p, dealii::Point<dim> segment_begin,  dealii::Point<dim> segment_end){
//        int return_value;
//        if (p[1] = segment_begin[1] = segment_end[1]){
//            if (segment_begin[0] <= p[0] <= segment_end[0] || segment_end[0] <= p[0] <= segment_begin[0])
//                return_value = 0; // on boundary
//            else
//                return_value = 1;} // inside
//
//        if (segment_begin[1] > segment_end[1]){
//            dealii::Point<dim> temp = segment_begin;
//            segment_begin = segment_end;
//            segment_end = temp;}
//
//        if (p[1] = segment_begin[1] && p[0] == segment_begin[0])
//            return_value = 0; // on boundary
//        if (p[1] <= segment_begin[1] || p[1] > segment_end[1])
//            return_value = 1; // inside
//        int delta = (segment_begin[0] - p[0]) * (segment_end[1]- p[1])- (segment_begin[1] - p[1]) * (segment_end[0] - p[0]);
//        if (delta > 0)
//            return_value = -1; // outside
//        else if (delta < 0)
//            return_value = 1; // inside
//        else
//            return_value = 0; // on boundary
//       return return_value;
//    }


    bool is_inside(const dealii::Point<dim> p1) const // test whether a point is inside the polygon
    {
        segment closest_segment = segment_list[0];
        double minimum_distance = calculate_distance(closest_segment, p1);
        segment my_segment;
        double distance;

        for (unsigned int i = 1; i <segment_list.size(); ++i)
        {
            my_segment = segment_list[i];
            distance = calculate_distance(my_segment, p1);
            if(distance < minimum_distance)
            {
                minimum_distance = distance;
                closest_segment = my_segment;
            }
        }
        //        std::cout<<"Closest segment: ["<<closest_segment.beginPoint<<"] ["<<closest_segment.endPoint<<"]"<<std::endl;
        //        std::cout<<"Min Distance: "<<minimum_distance<<std::endl;
        dealii::Point<2> point_vector =  {closest_segment.beginPoint[0] - p1[0], closest_segment.beginPoint[1] - p1[1]};

        if (scalar_product(closest_segment.normalVector, point_vector) > 0) // if scalar product == 0 -> on boundary
            return true;
        else
            return false;
    }

    bool is_on_boundary(const dealii::Point<dim> p1) const 
    {
        segment my_segment = segment_list[0];
        double minimum_distance = calculate_distance(my_segment, p1);
        segment closest_segment = my_segment;

        for (unsigned int i = 1; i <segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            double distance = calculate_distance(my_segment, p1);
            if(distance < minimum_distance)
            {
                minimum_distance = distance;
                closest_segment = my_segment;
            }
        }
        //        std::cout<<"Closest segment: ["<<closest_segment.beginPoint<<"] ["<<closest_segment.endPoint<<"]"<<std::endl;
        //        std::cout<<"Min Distance: "<<minimum_distance<<std::endl;
        dealii::Point<2> point_vector =  {closest_segment.beginPoint[0] - p1[0], closest_segment.beginPoint[1] - p1[1]};

        if (scalar_product(closest_segment.normalVector, point_vector) == 0) // if scalar product == 0 -> on boundary
            return true;
        else
            return false;
    }

    void save_q_points() const // save quadrature points to txt file
    {
        std::remove("plot_q_points_on_boundary");
        std::ofstream ofs_q_points;
        ofs_q_points.open ("plot_q_points_on_boundary", std::ofstream::out | std::ofstream::app);

        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            for (unsigned int j = 0; j <my_segment.q_points .size(); ++j)
                ofs_q_points <<  my_segment.q_points[j] << std::endl;
        }

        ofs_q_points.close();
    }

    std::vector<dealii::Point<2>> get_q_points(){
        std::vector<dealii::Point<2>> q_points;
        for (unsigned int i = 0; i <segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];
            for (unsigned int j = 0; j <my_segment.q_points.size(); ++j)
                q_points.insert(q_points.end(), my_segment.q_points[j]);
        }

        return q_points;
    }

    std::vector<int> get_segment_indices_inside_cell(dealii::DoFHandler<2>::cell_iterator cell) const
    {
        std::vector<int> segment_indices;
        for (unsigned int i = 0; i < segment_list.size(); ++i)
        {
            segment my_segment = segment_list[i];

            if (cell->vertex(0)[0] <= std::min(my_segment.beginPoint[0], my_segment.endPoint[0]) && cell->vertex(0)[1] <= std::min(my_segment.beginPoint[1], my_segment.endPoint[1]) && cell->vertex(3)[0] >= std::max(my_segment.beginPoint[0], my_segment.endPoint[0]) && cell->vertex(3)[1] >= std::max(my_segment.beginPoint[1], my_segment.endPoint[1]))

                segment_indices.insert(segment_indices.end(), i);
        }
        return segment_indices;
    }


private:

    std::vector<dealii::Point<2>> calculate_q_points(const segment my_segment) const
    {
        dealii::Point<2> q_point;
        std::vector<dealii::Point<2>> q_points;

        for (unsigned int i = 0; i < boundary_quadrature.size(); ++i)
        {
            q_point = my_segment.beginPoint + ((my_segment.endPoint-my_segment.beginPoint) * boundary_quadrature.get_points()[i][0]);
            q_points.insert(q_points.end(), q_point);
        }

        return q_points;
    }

    std::vector<double> calculate_q_weights() const
    {
        std::vector<double> q_weights;
        for (unsigned int i = 0; i < boundary_quadrature.size(); ++i)
        {
            q_weights.insert(q_weights.end(), boundary_quadrature.get_weights()[i]);
        }
        return q_weights;
    }

    dealii::Point<2> calculate_normals (const segment my_segment) const // for polygon defined clockwise
    {
        double dx, dy;
        dx = (my_segment.endPoint[0] - my_segment.beginPoint[0]) / my_segment.length;
        dy = (my_segment.endPoint[1] - my_segment.beginPoint[1]) / my_segment.length;

        dealii::Point<2> normalVector = {-dy, dx};

        Assert(std::isfinite(normalVector[0]), dealii::ExcNumberNotFinite(std::complex<double>(normalVector[0])));
        Assert(std::isfinite(normalVector[1]), dealii::ExcNumberNotFinite(std::complex<double>(normalVector[1])));

        return normalVector;
    }

    double calculate_distance(const segment my_segment, const dealii::Point<2> p) const
    {
        double distance = std::abs(my_segment.beginPoint.distance(p)) +  std::abs(my_segment.endPoint.distance(p));
        return distance;
    }

};



