#include "../include/QuadTree.h"

QuadTree::QuadTree() : QuadTree({}, 0, 0) { }

QuadTree::QuadTree(const QuadTree &otherQuadtree) :
                    QuadTree(otherQuadtree.bounds, otherQuadtree.capacity, otherQuadtree.maxLevel) { }

QuadTree::QuadTree(const Rectangle &_bound, unsigned _capacity, unsigned _maxLevel) :
    bounds(_bound), capacity(_capacity), maxLevel(_maxLevel) {
        objects.reserve(_capacity);
        foundObjects.reserve(_capacity);
        shape.setOutlineThickness(1.f);
        shape.setSize(sf::Vector2f((float)bounds.width - shape.getOutlineThickness(), (float)bounds.height));
        shape.setOutlineColor(sf::Color::Black);
        shape.setPosition((float)_bound.x, (float)_bound.y);
        shape.setFillColor(sf::Color(242, 242, 242));
        text.setFillColor(sf::Color(128, 128, 128));
}

// Inserts an object into this quadtree
bool QuadTree::insert(Particle *obj) {

    if (obj->qt != nullptr) {
        return false;
    }

    if (!isLeaf) {
        // insert object into leaf
        if (QuadTree *child = getChild(obj->x, obj->y))
            return child->insert(obj);
    }
    objects.push_back(obj);
    obj->qt = this;

    // Subdivide if required
    if (isLeaf && level < maxLevel && objects.size() >= capacity) {
        subdivide();
        update(obj);
    }
    return true;
}

// Removes an object from this quadtree
bool QuadTree::remove(Particle *obj) {

    if (obj->qt == nullptr) {
        return false; // Cannot exist in vector
    }
    if (obj->qt != this) {
        return obj->qt->remove(obj);
    }

    objects.erase(std::find(objects.begin(), objects.end(), obj));
    obj->qt = nullptr;
    deleteEmptyNodes();
    return true;
}

// Removes and re-inserts object into quadtree (for objects that move)
bool QuadTree::update(Particle *obj) {
    if (!remove(obj)) {
        return false;
    }
    // Not contained in this node -- insert into parent
    if (parent != nullptr && !bounds.contains(obj->x, obj->y)) {
        return parent->insert(obj);
    }
    if (!isLeaf) {
        // Still within current node -- insert into leaf
        if (QuadTree *child = getChild(obj->x, obj->y))
            return child->insert(obj);
    }
    return insert(obj);
}

// Searches quadtree for objects within the provided boundary and returns them in vector
std::vector<Particle*> &QuadTree::getObjectsInBound_unchecked(const Rectangle &bound) {
    foundObjects.clear();
    for (const auto &obj : objects) {
        // Only check for intersection with OTHER boundaries
        foundObjects.push_back(obj);
    }
    if (!isLeaf) {
        // Get objects from leaves
        if (QuadTree *child = getChild(bound)) {
            child->getObjectsInBound_unchecked(bound);
            foundObjects.insert(foundObjects.end(), child->foundObjects.begin(), child->foundObjects.end());
        } else for (QuadTree *leaf : children) {
                if (leaf->bounds.intersects(bound)) {
                    leaf->getObjectsInBound_unchecked(bound);
                    foundObjects.insert(foundObjects.end(), leaf->foundObjects.begin(), leaf->foundObjects.end());
                }
            }
    }
    return foundObjects;
}

std::tuple<float, float, float> QuadTree::getCenterOfMass() {
    float total_mass = 0.0f;
    float m_r_x = 0.0f;
    float m_r_y = 0.0f;
    for (auto&& found : this->getObjectsInBound_unchecked(bounds)) {
        total_mass += found->m;
        m_r_x += found->m * found->x;
        m_r_y += found->m * found->y;
    }
    return std::make_tuple(m_r_x/total_mass, m_r_y/total_mass, total_mass);
}

// Returns total children count for this quadtree
unsigned QuadTree::totalNumberOfChildren() const noexcept {
    unsigned total = 0;
    if (isLeaf) return total;
    for (QuadTree *child : children)
        total += child->totalNumberOfChildren();
    return 4 + total;
}

// Returns total object count for this quadtree
unsigned QuadTree::totalNumberOfObjects() const noexcept {
    unsigned total = (unsigned)objects.size();
    if (!isLeaf) {
        for (QuadTree *child : children)
            total += child->totalNumberOfObjects();
    }
    return total;
}

void QuadTree::setFont(const sf::Font &font) noexcept {
    text.setFont(font);
    text.setCharacterSize(40 / (level? level : 1));
    text.setPosition(
            (float)bounds.getRight() - (text.getLocalBounds().width  * 0.5f),
            (float)bounds.getTop()-1 - (text.getLocalBounds().height * 0.5f)
    );
    if (isLeaf) return;
    for (QuadTree *child : children)
        child->setFont(font);
}

void QuadTree::draw(sf::RenderTarget &canvas) noexcept {
    setFont(*text.getFont());

    if (!objects.empty())
        shape.setFillColor(sf::Color::White);
    else
        shape.setFillColor(sf::Color(242, 242, 242));

    canvas.draw(shape);
    if (!isLeaf) {
        for (QuadTree *child : children)
            child->draw(canvas);
    } else {
        std::stringstream ss;
        ss << level;
        text.setString(ss.str());
        canvas.draw(text);
    }
}

void QuadTree::drawCOM(sf::RenderTarget &canvas) noexcept {
    float radius = 5;
    if (isLeaf) {
        auto com = getCenterOfMass();
        sf::CircleShape comShape;
        comShape.setRadius(radius);
        //comShape.setPosition((float)std::get<0>com, (float)std::get<1>com);
        comShape.setPosition(std::get<0>(com) - radius/2.0f, std::get<1>(com) - radius/2.0f);
        comShape.setFillColor(sf::Color::Black);
        canvas.draw(comShape);
    }
    else {
        for (QuadTree *child : children) {
            child->drawCOM(canvas);
        }
    }
}

// Removes all objects and children from this quadtree
void QuadTree::clear() noexcept {
    if (!objects.empty()) {
        for (auto&& obj : objects)
            obj->qt = nullptr;
        objects.clear();
    }
    if (!isLeaf) {
        for (QuadTree *child : children)
            child->clear();
        isLeaf = true;
    }
}

// Subdivides into four quadrants
void QuadTree::subdivide() {
    double width  = bounds.width  * 0.5f;
    double height = bounds.height * 0.5f;
    double x = 0, y = 0;
    for (unsigned i = 0; i < 4; ++i) {
        switch (i) {
            case 0: x = bounds.x + width; y = bounds.y; break; // Top right
            case 1: x = bounds.x;         y = bounds.y; break; // Top left
            case 2: x = bounds.x;         y = bounds.y + height; break; // Bottom left
            case 3: x = bounds.x + width; y = bounds.y + height; break; // Bottom right
        }
        children[i] = new QuadTree({ x, y, width, height }, capacity, maxLevel);
        children[i]->level  = level + 1;
        children[i]->parent = this;
    }
    isLeaf = false;
}

// Discards buckets if all children are leaves and contain no objects
void QuadTree::deleteEmptyNodes() {
    if (!objects.empty()) return;
    if (!isLeaf) {
        for (QuadTree *child : children)
            if (!child->isLeaf || !child->objects.empty())
                return;
    }
    if (clear(), parent != nullptr)
        parent->deleteEmptyNodes();
}

QuadTree *QuadTree::getLeaf(const Rectangle &bound) {
    QuadTree *leaf = this;
    if (!isLeaf) {
        if (QuadTree *child = getChild(bound))
            leaf = child->getLeaf(bound);
    }
    return leaf;
}

QuadTree *QuadTree::getChild(const Rectangle &bound) const noexcept {
    bool left  = bound.x + bound.width < bounds.getRight();
    bool right = bound.x > bounds.getRight();

    if (bound.y + bound.height < bounds.getTop()) {
        if (left)  return children[1]; // Top left
        if (right) return children[0]; // Top right
    } else if (bound.y > bounds.getTop()) {
        if (left)  return children[2]; // Bottom left
        if (right) return children[3]; // Bottom right
    }
    return nullptr; // Cannot contain boundary -- too large
}

// Returns child that contains the provided boundary
QuadTree *QuadTree::getChild(const double x, const double y) const noexcept {
    bool left  = x < bounds.getRight();
    bool right = x > bounds.getRight();

    if (y < bounds.getTop()) {
        if (left)  return children[1]; // Top left
        if (right) return children[0]; // Top right
    } else if (y > bounds.getTop()) {
        if (left)  return children[2]; // Bottom left
        if (right) return children[3]; // Bottom right
    }
    return nullptr; // Cannot contain boundary -- too large
}


/**
//calculate force directly
for (auto&& found : this->getObjectsInBound_unchecked(bounds)) {
    if (!obj->identical(found)) {
        obj->calculateForce(found);
    }
}
 **/

void QuadTree::calculateForce(Particle *obj) {

    if (isLeaf) {
        auto com = getCenterOfMass();
        Particle COMParticle = Particle(std::get<0>(com),
                                        std::get<1>(com),
                                        std::get<2>(com));
        if (!obj->identical(COMParticle)) {
            obj->calculateForce(COMParticle);
        }
    } else {
        // Get distances
        auto com = getCenterOfMass();
        Particle COMParticle = Particle(std::get<0>(com),
                                        std::get<1>(com),
                                        std::get<2>(com));

        float distance = obj->getDistance(COMParticle);

        if ((bounds.width) / distance > 0.5f) {
            // Go deeper in the tree
            if (QuadTree *child = getChild(obj->x, obj->y)) {
                child->calculateForce(obj);
            }
        } else {
            if (!obj->identical(COMParticle)) {
                std::cout << "Not identical!!!" << std::endl;
                std::cout << COMParticle.x << " " << COMParticle.y << std::endl;
                obj->calculateForce(COMParticle);
            }
        }
    }
}

QuadTree::~QuadTree() {

    clear();
    if (children[0]) {
        delete children[0];
    }
    if (children[1]) {
        delete children[1];
    }
    if (children[2]) {
        delete children[2];
    }
    if (children[3]) {
        delete children[3];
    }
}

