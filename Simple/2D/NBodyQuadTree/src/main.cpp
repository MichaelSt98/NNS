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
float RADIUS = 3;
unsigned CAPACITY = 8;
unsigned MAX_LEVEL = 5;

class Object {

public:

    //sf::RectangleShape shape;
    sf::CircleShape shape;
    Particle item;

    Object(double _x, double _y) {

        double dx = (rand() % 201 - 100) * 0.05f; //0.05f;
        double dy = (rand() % 201 - 100) * 0.05f; //0.05f
        double mass = 1;

        item = Particle( _x, _y, dx, dy, mass);

        shape.setPosition((float)item.x - RADIUS/2.0f, (float)item.y - RADIUS/2.0f);
        shape.setRadius(RADIUS);

    }

    void move() {
        item.move(MAP_BOUNDS);
        shape.setPosition((float)item.x - RADIUS/2.0f, (float)item.y - RADIUS/2.0f);
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
    window.setMouseCursorVisible(true);

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

    // mouse boundary for choosing quad in GUI
    Rectangle mouseBoundary = { 0, 0, 20, 20 };

    // moving objects or frozen objects
    bool freezeObjects = false;
    bool centerOfMass = true;

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
                        // M = center of mass
                    case sf::Keyboard::M:
                        centerOfMass = !centerOfMass;
                        break;
                        // C = clear quadtree and remove all objects
                    case sf::Keyboard::C:
                        map.clear();
                        for (auto &&obj : objects)
                            delete obj;
                        objects.clear();
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

        if (centerOfMass) {
            map.drawCOM(window);
        }

        for (auto&& obj : objects) {
            obj->shape.setFillColor(sf::Color::Blue);

            if (!freezeObjects) {
                //obj->item.move(MAP_BOUNDS);
                obj->move();
                map.update(&obj->item);
            }
            window.draw(obj->shape);
        }
        // Update mouse box
        mouseBoundary.x = sf::Mouse::getPosition(window).x;
        mouseBoundary.y = sf::Mouse::getPosition(window).y;

        // Add objects on left click
        if (sf::Mouse::isButtonPressed(sf::Mouse::Left) && MAP_BOUNDS.contains(mouseBoundary)) {
            objects.push_back(new Object(mouseBoundary.getRight(), mouseBoundary.getTop()));
            map.insert(&objects.back()->item);
        }

        // Display quadtree debug info
        std::stringstream ss;
        ss <<   "Total number of Children:     " << map.totalNumberOfChildren()
           << "\nTotal number of Objects:      " << map.totalNumberOfObjects()
           << "\nObjects in this quad:         " << map.getLeaf(mouseBoundary)->totalNumberOfObjects();
        info.setString(ss.str());
        window.draw(info);

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

