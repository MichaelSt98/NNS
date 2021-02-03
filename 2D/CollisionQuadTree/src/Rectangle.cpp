#include "../include/Rectangle.h"

Rectangle::Rectangle(const Rectangle &otherRectangle)
                    : Rectangle(otherRectangle.x, otherRectangle.y, otherRectangle.width, otherRectangle.height) { }

Rectangle::Rectangle(double _x, double _y, double _width, double _height) :
        x { _x }, y { _y }, width { _width }, height { _height} { }

bool Rectangle::contains(const Rectangle &otherRectangle) const noexcept {
    if (x > otherRectangle.x) {
        return false;
    }
    if (y > otherRectangle.y) {
        return false;
    }
    if (x + width  < otherRectangle.x + otherRectangle.width) {
        return false;
    }
    if (y + height < otherRectangle.y + otherRectangle.height) {
        return false;
    }
    // within boundaries
    return true;
}

bool Rectangle::intersects(const Rectangle &otherRectangle) const noexcept {
    if (x > otherRectangle.x + otherRectangle.width) {
        return false;
    }
    if (x + width < otherRectangle.x) {
        return false;
    }
    if (y > otherRectangle.y + otherRectangle.height) {
        return false;
    }
    if (y + height < otherRectangle.y) {
        return false;
    }
    // intersects
    return true;
}

double Rectangle::getLeft() const noexcept {
    return x - (width  * 0.5f);
}

double Rectangle::getTop() const noexcept {
    return y + (height * 0.5f);
}

double Rectangle::getRight() const noexcept {
    return x + (width  * 0.5f);
}

double Rectangle::getBottom() const noexcept {
    return y - (height * 0.5f);
}