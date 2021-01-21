#include "../include/Rectangle.h"
#include "../include/Particle.h"
#include "../include/QuadTree.h"

#include <time.h>
#include <iostream>
#include <SFML/Graphics/RenderWindow.hpp>
#include <SFML/Graphics/Font.hpp>
#include <SFML/Window/Event.hpp>

/** Settings **/
Rectangle WINDOW_BOUNDS = { 0, 0, 1280, 720 };
Rectangle MAP_BOUNDS = {0, 0, WINDOW_BOUNDS.width - 560, WINDOW_BOUNDS.height};
float RADIUS = 10;
unsigned CAPACITY = 8;
unsigned MAX_LEVEL = 5;

class Object {

public:

    // velocity
    double dx;
    double dy;
    //sf::RectangleShape shape;
    sf::CircleShape shape;
    Particle item;

    Object(double _x, double _y, double _width, double _height) {
        item = Particle({ _x, _y, _width, _height }, this);
        shape.setPosition((float)item.bound.x, (float)item.bound.y);
        shape.setRadius(RADIUS);
        dx = (rand() % 201 - 100) * 0.05f; //0.05f;
        dy = (rand() % 201 - 100) * 0.05f; //0.05f
    }

    void move() {
        if (item.bound.x + dx < 0 || item.bound.x + item.bound.width + dx > MAP_BOUNDS.width)
            dx = -dx;
        if (item.bound.y + dy < 0 || item.bound.y + item.bound.height + dy > MAP_BOUNDS.height)
            dy = -dy;
        item.bound.x += dx;
        item.bound.y += dy;
        shape.setPosition((float)item.bound.x, (float)item.bound.y);
    }
};

int main() {
    // initialize random number generator
    srand((unsigned)time(NULL));

    // initialize window for graphical representation
    sf::RenderWindow window(sf::VideoMode((unsigned)WINDOW_BOUNDS.width, (unsigned)WINDOW_BOUNDS.height), "QuadTree");
    // set frame rate (limit)
    window.setFramerateLimit(60);
    // set mouse cursor invisible
    window.setMouseCursorVisible(false);

    // create QuadTree instance
    QuadTree map = QuadTree(MAP_BOUNDS, CAPACITY, MAX_LEVEL);
    std::vector<Object*> objects;

    // font for graphical representation
    sf::Font font;
    font.loadFromFile("UbuntuMono-R.ttf");
    map.setFont(font);

    // info text (box)
    sf::Text info("Info", font);
    info.setCharacterSize(15);
    info.setStyle(sf::Text::Bold);
    info.setFillColor(sf::Color::White);
    info.setPosition(720, 0);

    // event handling
    sf::Event event;

    // mouse handling
    sf::CircleShape mouseHandler;
    mouseHandler.setOutlineThickness(3.0f);
    mouseHandler.setFillColor(sf::Color(127, 0, 255, 0));
    mouseHandler.setOutlineColor(sf::Color::Magenta);
    Rectangle mouseBoundary = { 0, 0, 20, 20 };

    // moving objects or frozen objects
    bool freezeObjects = false;

    /** GUI mainloop **/

    while (window.isOpen()) {
        // Update controls
        while (window.pollEvent(event)) {
            // Key events
            if (event.type == sf::Event::KeyPressed) {
                switch (event.key.code) {
                    // Esc = exit
                    case sf::Keyboard::Escape:
                        window.close();
                        break;
                        // F = freeze all objects
                    case sf::Keyboard::F:
                        freezeObjects = !freezeObjects;
                        break;
                        // C = clear quadtree and remove all objects
                    case sf::Keyboard::C:
                        map.clear();
                        for (auto &&obj : objects)
                            delete obj;
                        objects.clear();
                        break;
                        // Up = increase size of mouse box
                    case sf::Keyboard::Up:
                        mouseBoundary.width += 2;
                        mouseBoundary.height += 2;
                        break;
                        // Down = decrease size of mouse box
                    case sf::Keyboard::Down:
                        mouseBoundary.width -= 2;
                        mouseBoundary.height -= 2;
                        break;
                    default:
                        std::cout << "Not implemented" << std::endl;
                }
            }
            // Window closed?
            else if (event.type == sf::Event::Closed) {
                window.close();
            }
            // ...
            else {
                // if event.type is not a KeyPressed nor Closed event
            }
        }
        //clear the window
        window.clear();
        // draw the map (QuadTree)
        map.draw(window);

        /** Collisions **/

        std::vector<Object*> mouseCollisions;
        unsigned long long collisions = 0;
        unsigned long long qtCollisionChecks = 0;
        unsigned long long bfCollisionChecks = 0;
        for (auto&& obj : objects) {
            obj->shape.setFillColor(sf::Color::Blue);

            if (mouseBoundary.intersects(obj->item.bound)) {
                obj->shape.setFillColor(sf::Color::Red);
                mouseCollisions.push_back(obj);
                ++collisions;
            }
            for (auto&& otherObj : objects)
                ++bfCollisionChecks;
            for (auto&& found : map.getObjectsInBound_unchecked(obj->item.bound)) {
                ++qtCollisionChecks;
                if (&obj->item != found && found->bound.intersects(obj->item.bound)) {
                    ++collisions;
                    obj->shape.setFillColor(sf::Color::Red);
                }
            }
            if (!freezeObjects) {
                obj->move();
                map.update(&obj->item);
            }
            window.draw(obj->shape);
        }
        // Update mouse box
        mouseBoundary.x = sf::Mouse::getPosition(window).x;
        mouseBoundary.y = sf::Mouse::getPosition(window).y;
        mouseHandler.setRadius(RADIUS);
        mouseHandler.setPosition((float)mouseBoundary.x, (float)mouseBoundary.y);

        // Add objects on left click
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && MAP_BOUNDS.contains(mouseBoundary)) {
            objects.push_back(new Object(mouseBoundary.getRight(), mouseBoundary.getTop(), rand() % 20 + 4, rand() % 20 + 4));
            map.insert(&objects.back()->item);
        }
        // Remove objects on right click
        if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
            for (auto&& obj : mouseCollisions) {
                objects.erase(std::find(objects.begin(), objects.end(), obj));
                map.remove(&obj->item);
                delete obj;
            }
        }
        // Display quadtree debug info
        std::stringstream ss;
        ss <<   "Total number of Children:     " << map.totalNumberOfChildren()
           << "\nTotal number of Objects:      " << map.totalNumberOfObjects()
           << "\nTotal number of Collisions:   " << collisions
           << "\nQuadTree collision checks:    " << qtCollisionChecks
           << "\nBrute force collision checks: " << bfCollisionChecks
           << "\nCollisions with mouse:        " << mouseCollisions.size()
           << "\nObjects in this quad:         " << map.getLeaf(mouseBoundary)->totalNumberOfObjects();
        info.setString(ss.str());
        window.draw(info);

        if (MAP_BOUNDS.contains(mouseBoundary)) {
            window.draw(mouseHandler);
        }

        window.display();
    }

    /** GUI was terminated
     * cleanup (memory)
     * **/

    // delete map/QuadTree
    map.clear();
    // delete objects
    for (auto&& obj : objects)
        delete obj;
    objects.clear();
}

