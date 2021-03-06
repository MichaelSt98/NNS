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

int Tree::getMaxDepth() {
    if (isExternal()) {
        return -1;
    }
    else
    {
        // UNW; //0 // UNE; //1 // USW; //2 // USE; //3
        // LNW; //4 // LNE; //5 // LSW; //6 // LSE; //7
        int depths [8] = {};

        if (UNW != NULL) {
            depths[0] = UNW->getMaxDepth();
        }
        if (UNE != NULL) {
            depths[1] = UNE->getMaxDepth();
        }
        if (USW != NULL) {
            depths[2] = USW->getMaxDepth();
        }
        if (USE != NULL) {
            depths[3] = USE->getMaxDepth();
        }
        if (LNW != NULL) {
            depths[4] = LNW->getMaxDepth();
        }
        if (LNE != NULL) {
            depths[5] = LNE->getMaxDepth();
        }
        if (LSW != NULL) {
            depths[6] = LSW->getMaxDepth();
        }
        if (LSE != NULL) {
            depths[7] = LSE->getMaxDepth();
        }

        int max_depth = 0;

        for (int i=0; i<8; i++) {
            if (depths[i] > max_depth) {
                max_depth = depths[i];
            }
        }
        return max_depth + 1;

    }
}

void Tree::getDepth(Body *body, int &depth) {
    if (isExternal()) {
        return;
    }
    else {
        depth++;
        int subOctant = octant.getSubOctant(body->position);
        switch (subOctant) {
            // UNW; //0 // UNE; //1 // USW; //2 // USE; //3
            // LNW; //4 // LNE; //5 // LSW; //6 // LSE; //7
            case 0:
                if (UNW == NULL) { return; }
                UNW->getDepth(body, depth);
                break;
            case 1:
                if (UNE == NULL) { return; }
                UNE->getDepth(body, depth);
                break;
            case 2:
                if (USW == NULL) { return; }
                USW->getDepth(body, depth);
                break;
            case 3:
                if (USE == NULL) { return; }
                USE->getDepth(body, depth);
                break;
            case 4:
                if (LNW == NULL) { return; }
                LNW->getDepth(body, depth);
                break;
            case 5:
                if (LNE == NULL) { return; }
                LNE->getDepth(body, depth);
                break;
            case 6:
                if (LSW == NULL) { return; }
                LSW->getDepth(body, depth);
                break;
            case 7:
                if (LSE == NULL) { return; }
                LSE->getDepth(body, depth);
                break;
            default:
                return;
        }
    }
}

void Tree::insert(Body *body) {

    int version = 1;

    if (centerOfMass.mass == 0.0) {
        centerOfMass = *body;
    }
    else {
        bool isExtern = isExternal();
        Body *updatedBody;
        if (!isExtern) {

            //std::cout << "NotExtern" << std::endl;

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
            //std::cout << "Else ..." << std::endl;
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
            Octant unw = octant.getUNW();
            if (unw.contains(updatedBody->position)) {
                if (UNW == NULL) { UNW = new Tree(unw); }
                UNW->insert(updatedBody);
            } else {
                Octant une = octant.getUNE();
                if (une.contains(updatedBody->position)) {
                    if (UNE == NULL) { UNE = new Tree(une); }
                    UNE->insert(updatedBody);
                } else {
                    Octant usw = octant.getUSW();
                    if (usw.contains(updatedBody->position)) {
                        if (USW == NULL) { USW = new Tree(usw); }
                        USW->insert(updatedBody);
                    } else {
                        Octant use = octant.getUSE();
                        if (use.contains(updatedBody->position)) {
                            if (USE == NULL) { USE = new Tree(use); }
                            USE->insert(updatedBody);
                        } else {
                            Octant lnw = octant.getLNW();
                            if (lnw.contains(updatedBody->position)) {
                                if (LNW == NULL) { LNW = new Tree(lnw); }
                                LNW->insert(updatedBody);
                            } else {
                                Octant lne = octant.getLNE();
                                if (lne.contains(updatedBody->position)) {
                                    if (LNE == NULL) { LNE = new Tree(lne); }
                                    LNE->insert(updatedBody);
                                } else {
                                    Octant lsw = octant.getLSW();
                                    if (lsw.contains(updatedBody->position)) {
                                        if (LSW == NULL) { LSW = new Tree(std::move(lsw)); }
                                        LSW->insert(updatedBody);
                                    } else {
                                        Octant lse = octant.getLSE();
                                        if (lse.contains(updatedBody->position)) {
                                            if (LSE == NULL) { LSE = new Tree(lse); }
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
