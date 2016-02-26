// from Patrick Esser

#include <deal.II/grid/tria.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/qprojector.h>

/**
 * Child index of a child cell.
 */
  template <class IT>
unsigned int get_child_nr(const IT child)
{
  unsigned int targets_child_nr = (unsigned int) -1;
  for(unsigned int child_nr = 0;
      child_nr < child->parent()->n_children();
      ++child_nr)
  {
    if(child->parent()->child_index(child_nr) == child->index())
    {
      targets_child_nr = child_nr;
      break;
    }
  }
  Assert(targets_child_nr != (unsigned int) -1, dealii::ExcInternalError());
  return targets_child_nr;
}

/**
 * Calculate the path from a root node to a given target cell.
 */
  template <class IT>
void get_path(std::vector<unsigned int>& path, IT target)
{
  while(target->level() > 0)
  {
    path[target->level()] = get_child_nr<IT>(target);
    target = target->parent();
  }
  path[0] = target->index();
}

/**
 * Walk a given path as long as possible.
 */
template <int dim>
  typename dealii::Triangulation<dim>::cell_iterator
walk_path(std::vector<unsigned int>& path, const dealii::Triangulation<dim>& tria)
{
  typename dealii::Triangulation<dim>::cell_iterator node(
      &tria,
      0,
      path[0],
      0);
  for(unsigned int step = 0; step < path.size() - 1; ++step)
  {
    node = node->child(path[step + 1]);
    if(!node->has_children())
      break;
  }
  return node;
}


/**
 * Given a cell iterator and a triangulation (which is not necessarily the
 * same as the cell iterator is pointing to), return a cell iterator
 * pointing to a cell in the given triangulation which follows the same path
 * as the given cell iterator as long as possible in the given
 * triangulation.
 * The resulting cell iterator either
 * - is logically a parent of the given cell iterator (if the given
 *   triangulation is locally coarser than the one the given cell iterator
 *   is pointing to) or
 * - is logically the same cell iterator as the given one
 */
  template <int dim>
typename dealii::Triangulation<dim>::cell_iterator
topological_equivalent(
    typename dealii::Triangulation<dim>::cell_iterator cell,
    const dealii::Triangulation<dim>& tria)
{
  if(&tria == &(cell->get_triangulation()))
  {
    // same triangulation, just copy the given iterator
    return cell;
  }
  // calculate path to root
  unsigned int level = cell->level();
  std::vector<unsigned int> local_child_indices(level + 1);
  get_path(local_child_indices, cell);
  // walk the path in the given triangulation
  return walk_path(local_child_indices, tria);
}
