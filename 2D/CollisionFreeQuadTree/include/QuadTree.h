#ifndef QUADTREE_QUADTREE_H
#define QUADTREE_QUADTREE_H

#include "Rectangle.h"
#include "Particle.h"

#include <any>
#include <vector>
#include <tuple>
#include <sstream>
#include <algorithm>
#include <SFML/Graphics/RenderTarget.hpp>
#include <SFML/Graphics/CircleShape.hpp>
#include <SFML/Graphics/RectangleShape.hpp>
#include <SFML/Graphics/Text.hpp>

class Particle;

class QuadTree {

private:
    bool      isLeaf = true; //! bool leaf or not leaf
    unsigned  level  = 0;    //! current level/depth
    unsigned  capacity;      //! capacity (maximum amount of objects, before subdividing)
    unsigned  maxLevel;      //! maximum level/depth of QuadTree

    Rectangle bounds;                //! boundaries as Rectangle instance
    QuadTree* parent = nullptr;      //! parent QuadTree
    QuadTree* children[4] = { nullptr, nullptr, nullptr, nullptr }; //! chilren QuadTrees

    sf::Text  text;                        //! text for graphical interface
    sf::RectangleShape	   shape;          //! Shape (representation) for graphical interface
    std::vector<Particle*> objects;        //! objects within the QuadTree
    std::vector<Particle*> foundObjects;   //! found objects within the QuadTree

    /**!
     * Subdivide the QuadTree (since capacity is reached)
     */
    void subdivide();

    /**!
     * Delete empty nodes (with empty leaf)
     */
    void deleteEmptyNodes();

    /**!
     * Get children of QuadTree within boundaries
     * @param bound boundaries as Rectangle instance
     *
     * @return QuadTree
     */
    inline QuadTree *getChild(const Rectangle &bound) const noexcept;

    inline QuadTree *getChild(const double x, const double y) const noexcept;

public:
    /**!
     * Constructor for QuadTree using a Rectangle, capacity and maximum level
     *
     * @param _bound Rectangle instance
     * @param _capacity capacity for each node before subdivision
     * @param _maxLevel maximum level of nodes
     */
    QuadTree(const Rectangle &_bound, unsigned _capacity, unsigned _maxLevel);

    /**!
     * Constructor for QuadTree using another QuadTree instance
     */
    QuadTree(const QuadTree&);

    /**!
     * Default constructor for QuadTree
     */
    QuadTree();

    std::tuple<float, float, float> getCenterOfMass();

    void drawCOM(sf::RenderTarget &canvas) noexcept;

    /**!
     * Insert a object/particle into the the QuadTree
     *
     * @param obj is a particle instance
     * @return bool, whether particle is inserted
     */
    bool insert(Particle *obj);

    /**!
     * Remove a object/particle from the QuadTree.
     *
     * @param obj is a particle instance
     * @return bool, whether particle is removed
     */
    bool remove(Particle *obj);

    /**!
     * Removes and re-inserts object into QuadTree
     *
     * @param obj is a particle instance
     * @return bool, whether particle is (re)-inserted
     */
    bool update(Particle *obj);

    /**!
     * Search QuadTree for objects/particles within the provided boundary and return them in vector
     *
     * @param bound Rectangle instance to provide a boundary to search within
     * @return std::vector of objects/particles within the boundaries
     */
    std::vector<Particle*> &getObjectsInBound_unchecked(const Rectangle &bound);

    /**!
     * Total number of children for this QuadTree
     *
     * @return unsigned int, total number of children
     */
    unsigned totalNumberOfChildren() const noexcept;

    /**!
     * Total number of objects/particles for this QuadTree
     *
     * @return unsigned int, total number of objects/particles
     */
    unsigned totalNumberOfObjects() const noexcept;

    /**!
     * Set font for graphical representation
     *
     * @param font sf::Font instance
     */
    void setFont(const sf::Font &font) noexcept;

    /**!
     * Make the graphical representation
     *
     * @param canvas
     */
    void draw(sf::RenderTarget &canvas) noexcept;

    /**!
     * Clear the graphical representation
     */
    void clear() noexcept;

    /**!
     * get Leaf of QuadTree
     *
     * @param bound
     * @return QuadTree
     */
    QuadTree *getLeaf(const Rectangle &bound);

    /**!
     * Destructor for the QuadTree, in order to delete the children recursively
     */
    ~QuadTree();

};


#endif //QUADTREE_QUADTREE_H
