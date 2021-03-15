//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box)); //TODO: ?
    output_particles(root);
    while (t < t_end) {
        t += delta_t;
        std::cout << "t = " << t << std::endl;
        output_particles(root);
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        compX_BH(root, delta_t);
        compF_BH(root, root, getSystemSize(&box));
        compV_BH(root, delta_t);
        repairTree(root);
    }
}
