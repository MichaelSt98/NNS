//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Interaction.h"

Interaction::Interaction() : frictionEnabled { true } { }

Interaction::Interaction(bool _frictionEnabled) : frictionEnabled { _frictionEnabled } { }

void Interaction::singleInteraction(Body* body1, Body* body2, bool single) {
    Vector3D positionDifference;
    positionDifference.x = (body1->position.x - body2->position.x) * TO_METERS;
    positionDifference.y = (body1->position.y - body2->position.y) * TO_METERS;
    positionDifference.z = (body1->position.z - body2->position.z) * TO_METERS;
    double distance = positionDifference.magnitude();

    if (distance == 0) {
        return;
    }

    double force = TIME_STEP * (G * body1->mass * body2->mass) /
                    ((distance*distance + SOFTENING*SOFTENING) * distance);

    body1->acceleration.x -= force * positionDifference.x / body1->mass;
    body1->acceleration.y -= force * positionDifference.y / body1->mass;
    body1->acceleration.z -= force * positionDifference.z / body1->mass;

#if ENABLE_FRICTION
        if (single)
        {
            double friction = 0.5/pow(2.0, FRICTION_FACTOR * (((distance + SOFTENING)) /
                                            (TO_METERS)));

            if (friction > 0.0001 && ENABLE_FRICTION)
            {
                body1->acceleration.x += friction * (body2->velocity.x - body1->velocity.x) / 2;
                body1->acceleration.y += friction * (body2->velocity.y - body1->velocity.y) / 2;
                body1->acceleration.z += friction * (body2->velocity.z - body1->velocity.z) / 2;
            }
        }
#else
    (void)single;
#endif

}


void Interaction::treeInteraction(Tree *tree, Body *body) {

    if (tree->isExternal()) {
        Body *treeBody = &tree->centerOfMass;
        singleInteraction(body, treeBody, true);
    }
    else if (tree->getOctant().getLength() /
            Vector3D::magnitude(tree->centerOfMass.position.x - body->position.x,
                                tree->centerOfMass.position.y - body->position.y,
                                tree->centerOfMass.position.z - body->position.z) < MAX_DISTANCE) {

        Body *treeBody = &tree->centerOfMass;
        singleInteraction(body, treeBody, false);
    }
    else {
        if (tree->UNW != NULL) {
            treeInteraction(tree->UNW, body);
        }
        if (tree->UNE != NULL) {
            treeInteraction(tree->UNE, body);
        }
        if (tree->USW != NULL) {
            treeInteraction(tree->USW, body);
        }
        if (tree->USE != NULL) {
            treeInteraction(tree->USE, body);
        }
        if (tree->LNW != NULL) {
            treeInteraction(tree->LNW, body);
        }
        if (tree->LNE != NULL) {
            treeInteraction(tree->LNE, body);
        }
        if (tree->LSW != NULL) {
            treeInteraction(tree->LSW, body);
        }
        if (tree->LSE != NULL) {
            treeInteraction(tree->LSE, body);
        }
    }
}
