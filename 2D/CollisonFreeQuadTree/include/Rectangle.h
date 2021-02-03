#ifndef QUADTREE_RECTANGLE_H
#define QUADTREE_RECTANGLE_H


class Rectangle {

public:
    double x;         //! x-coordinate
    double y;         //! y-coordinate
    double width;     //! width of the rectangle
    double height;    //! height of the rectangle

    /**!
     * Check if rectangle contains another rectangle
     *
     * @param otherRectangle other rectangle instance
     * @return bool, true if rectangle contains the other rectangle or false if not
     */
    bool contains(const Rectangle &otherRectangle) const noexcept;

    bool contains(const double x_pos, const double y_pos) const noexcept;

    /**!
     * Check if rectangle intersects another rectangle
     *
     * @param otherRectangle other rectangle instance
     * @return true if rectangle intersects the other rectangle or false if not
     */
    bool intersects(const Rectangle &otherRectangle) const noexcept;

    /**!
     * Get the left border of the rectangle
     *
     * @return double, the left border of the rectangle
     */
    double getLeft() const noexcept;

    /**!
     * Get the top border of the rectangle
     *
     * @return double, the top border of the rectangle
     */
    double getTop() const noexcept;

    /**!
     * get the right border of the rectangle
     *
     * @return double, the right border of the rectangle
     */
    double getRight() const noexcept;

    /**!
     * get the bottom border of the rectangle
     *
     * @return double, the bottom border of the rectangle
     */
    double getBottom() const noexcept;

    /**!
     * Constructor taking another rectangle instance.
     */
    Rectangle(const Rectangle&);

    /**!
     * Constructor, for creating a new rectangle instance.
     *
     * @param _x double, the x-coordinate
     * @param _y double, the y-coordinate
     * @param _width double, the width of the rectangle
     * @param _height double, the height of the rectangle
     */
    Rectangle(double _x = 0, double _y = 0, double _width = 0, double _height = 0);
};


#endif //QUADTREE_RECTANGLE_H
