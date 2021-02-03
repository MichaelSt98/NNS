//
// Created by Michael Staneker on 25.01.21.
//

#include "../include/Tree.h"

Tree::Tree(Octant&& o) : octant(std::move(o)) {
    UNW = NULL;
    UNE = NULL;
    USW = NULL;
    USE = NULL;
    LNW = NULL;
    LNE = NULL;
    LSW = NULL;
    LSE = NULL;
}

Tree::Tree(Octant& o) : octant(o){
    UNW = NULL;
    UNE = NULL;
    USW = NULL;
    USE = NULL;
    LNW = NULL;
    LNE = NULL;
    LSW = NULL;
    LSE = NULL;
}

const Octant& Tree::getOctant () const {
    return octant;
}

Tree::~Tree() {
    if (UNW != NULL) {
        delete UNW;
    }
    if (UNE != NULL) {
        delete UNE;
    }
    if (USW != NULL) {
        delete USW;
    }
    if (USE != NULL) {
        delete USE;
    }
    if (LNW != NULL) {
        delete LNW;
    }
    if (LNE != NULL) {
        delete LNE;
    }
    if (LSW != NULL) {
        delete LSW;
    }
    if (LSE != NULL) {
        delete LSE;
    }
}

bool Tree::isExternal() {
    return (UNW == NULL &&
            UNE == NULL &&
            USW == NULL &&
            USE == NULL &&
            LNW == NULL &&
            LNE == NULL &&
            LSW == NULL &&
            LSE == NULL);
}

void Tree::insert(Body *body) {

    int version = 3;

    if (centerOfMass.mass == 0.0) {
        centerOfMass = *body;
    }
    else {
        bool isExtern = isExternal();
        Body *updatedBody;
        if (!isExtern) {

            centerOfMass.position.x = (body->position.x * body->mass + centerOfMass.position.x * centerOfMass.mass) /
                                            (body->mass + centerOfMass.mass);
            centerOfMass.position.y = (body->position.y * body->mass + centerOfMass.position.y * centerOfMass.mass) /
                                            (body->mass + centerOfMass.mass);
            centerOfMass.position.z = (body->position.z * body->mass + centerOfMass.position.z * centerOfMass.mass) /
                                            (body->mass + centerOfMass.mass);

            centerOfMass.mass += body->mass;
            updatedBody = body;
        }
        else {

            updatedBody = &centerOfMass;
        }

        if (version == 1) {

            Octant &&unw = octant.getUNW();
            if (unw.contains(updatedBody->position)) {
                if (UNW == NULL) { UNW = new Tree(std::move(unw)); }
                UNW->insert(updatedBody);
            } else {
                Octant &&une = octant.getUNE();
                if (une.contains(updatedBody->position)) {
                    if (UNE == NULL) { UNE = new Tree(std::move(une)); }
                    UNE->insert(updatedBody);
                } else {
                    Octant &&usw = octant.getUSW();
                    if (usw.contains(updatedBody->position)) {
                        if (USW == NULL) { USW = new Tree(std::move(usw)); }
                        USW->insert(updatedBody);
                    } else {
                        Octant &&use = octant.getUSE();
                        if (use.contains(updatedBody->position)) {
                            if (USE == NULL) { USE = new Tree(std::move(use)); }
                            USE->insert(updatedBody);
                        } else {
                            Octant &&lnw = octant.getLNW();
                            if (lnw.contains(updatedBody->position)) {
                                if (LNW == NULL) { LNW = new Tree(std::move(lnw)); }
                                LNW->insert(updatedBody);
                            } else {
                                Octant &&lne = octant.getLNE();
                                if (lne.contains(updatedBody->position)) {
                                    if (LNE == NULL) { LNE = new Tree(std::move(lne)); }
                                    LNE->insert(updatedBody);
                                } else {
                                    Octant &&lsw = octant.getLSW();
                                    if (lsw.contains(updatedBody->position)) {
                                        if (LSW == NULL) { LSW = new Tree(lsw); }
                                        LSW->insert(updatedBody);
                                    } else {
                                        Octant &&lse = octant.getLSE();
                                        if (lse.contains(updatedBody->position)) {
                                            if (LSE == NULL) { LSE = new Tree(std::move(lse)); }
                                            LSE->insert(updatedBody);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else if (version == 2) {

            #pragma omp parallel
            {
                #pragma omp single
                {
                    #pragma omp task
                    {
                        Octant &&unw = octant.getUNW();
                        if (unw.contains(updatedBody->position)) {
                            if (UNW == NULL) { UNW = new Tree(std::move(unw)); }
                            UNW->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&une = octant.getUNE();
                        if (une.contains(updatedBody->position)) {
                            if (UNE == NULL) { UNE = new Tree(std::move(une)); }
                            UNE->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&usw = octant.getUSW();
                        if (usw.contains(updatedBody->position)) {
                            if (USW == NULL) { USW = new Tree(std::move(usw)); }
                            USW->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&use = octant.getUSE();
                        if (use.contains(updatedBody->position)) {
                            if (USE == NULL) { USE = new Tree(std::move(use)); }
                            USE->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&lnw = octant.getLNW();
                        if (lnw.contains(updatedBody->position)) {
                            if (LNW == NULL) { LNW = new Tree(std::move(lnw)); }
                            LNW->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&lne = octant.getLNE();
                        if (lne.contains(updatedBody->position)) {
                            if (LNE == NULL) { LNE = new Tree(std::move(lne)); }
                            LNE->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&lsw = octant.getLSW();
                        if (lsw.contains(updatedBody->position)) {
                            if (LSW == NULL) { LSW = new Tree(lsw); }
                            LSW->insert(updatedBody);
                        }
                    }

                    #pragma omp task
                    {
                        Octant &&lse = octant.getLSE();
                        if (lse.contains(updatedBody->position)) {
                            if (LSE == NULL) { LSE = new Tree(std::move(lse)); }
                            LSE->insert(updatedBody);
                        }
                    }
                } // end omp single
            } // end omp parallel

        }
        else if (version == 3) {
            int subOctant = octant.getSubOctant(updatedBody->position);
            //std::cout << "subOctant = " << subOctant << std::endl;
            switch (subOctant) {
                // UNW; //0 // UNE; //1 // USW; //2 // USE; //3
                // LNW; //4 // LNE; //5 // LSW; //6 // LSE; //7
                case 0:
                    if (UNW == NULL) { UNW = new Tree(octant.getUNW()); }
                    UNW->insert(updatedBody);
                    break;
                case 1:
                    if (UNE == NULL) { UNE = new Tree(octant.getUNE()); }
                    UNE->insert(updatedBody);
                    break;
                case 2:
                    if (USW == NULL) { USW = new Tree(octant.getUSW()); }
                    USW->insert(updatedBody);
                    break;
                case 3:
                    if (USE == NULL) { USE = new Tree(octant.getUSE()); }
                    USE->insert(updatedBody);
                    break;
                case 4:
                    if (LNW == NULL) { LNW = new Tree(octant.getLNW()); }
                    LNW->insert(updatedBody);
                    break;
                case 5:
                    if (LNE == NULL) { LNE = new Tree(octant.getLNE()); }
                    LNE->insert(updatedBody);
                    break;
                case 6:
                    if (LSW == NULL) { LSW = new Tree(octant.getLSW()); }
                    LSW->insert(updatedBody);
                    break;
                case 7:
                    if (LSE == NULL) { LSE = new Tree(octant.getLSE()); }
                    LSE->insert(updatedBody);
                    break;
                default:
                    std::cout << "Nothing to do ..." << std::endl;
            }
        }
        else {
            std::cout << "Version: " << version << " not implemented!" << std::endl;
        }
        if (isExtern) {
            insert(body);
        }
    }
}
