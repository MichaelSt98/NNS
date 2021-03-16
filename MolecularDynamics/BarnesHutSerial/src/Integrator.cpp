//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box,
                        Renderer *renderer, char *image, double *hdImage) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box)); //TODO: ?
    //output_particles(root);

    int step = 0;

    while (t < t_end) {
        std::cout << "t = " << t << std::endl;
        output_particles(root);
        ++step;
        // rendering
        if (step%renderer->getRenderInterval()==0)
        {
            Particle prtcls[renderer->getNumParticles()];
            get_particle_array(root, prtcls);
            renderer->createFrame(image, hdImage, prtcls, step);
        }

        t += delta_t; // update timestep
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        compX_BH(root, delta_t);
        compF_BH(root, root, getSystemSize(&box));
        compV_BH(root, delta_t);
        repairTree(root);
    }
    std::cout << "t = " << t << ", FINAL RESULT:" << std::endl;
    output_particles(root);

}
