//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

//TODO: adapt to parallel method (use SubDomainKeyTree)
void timeIntegration_BH(float t, float delta_t, float t_end, float diam, TreeNode *root, SubDomainKeyTree *s) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box));
    //output_particles(root);
    while (t < t_end) {
        std::cout << "t = " << t << std::endl;
        //output_particles(root);
        t += delta_t; // update timestep
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        compX_BH(root, delta_t);
        //compF_BH(root, root, getSystemSize(&box), s);
        compF_BHpar(root, diam, s);

        compV_BH(root, delta_t);

        //compF_BHpar(root, diam, s);

        sendParticles(root, s);
        //repairTree(root);
    }
    std::cout << "t = " << t << ", FINAL RESULT:" << std::endl;
    output_particles(root);
}
