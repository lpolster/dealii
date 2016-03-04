using namespace dealii;

class SolutionBase
{
protected:
    const Point<2>   source_center  = {0.0, 0.0};
    const double       width = 0.25;
};

template <int dim>
class Solution : public Function<dim>,
        protected SolutionBase
{
public:
    Solution () : Function<dim>() {}
    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
    virtual Tensor<1,dim> gradient (const Point<dim>   &p,
                                    const unsigned int  component = 0) const;
};

template <int dim>
double Solution<dim>::value (const Point<dim>   &p,
                             const unsigned int) const
{
    double return_value = 0;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    return_value = std::exp(-x_minus_xi.norm_square() /
                            (width * width));

    return return_value;
}

template <int dim>
Tensor<1,dim> Solution<dim>::gradient (const Point<dim>   &p,
                                       const unsigned int) const
{
    Tensor<1,dim> return_value;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    return_value = (-2 / (width * width) *
                    std::exp(-x_minus_xi.norm_square() /
                             (width * width)) *
                    x_minus_xi);

    return return_value;
}

template <int dim>
class RightHandSide : public Function<dim>,
        protected SolutionBase
{
public:
    RightHandSide () : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
};

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int) const
{
    double return_value = 0;

    const Tensor<1,dim> x_minus_xi = p - source_center;

    // The Laplacian:
    return_value = ((2*2 - 4*x_minus_xi.norm_square()/
                     (width * width)) /
                    (width * width) *
                    std::exp(-x_minus_xi.norm_square() /
                             (width * width)));
    return return_value;
}
