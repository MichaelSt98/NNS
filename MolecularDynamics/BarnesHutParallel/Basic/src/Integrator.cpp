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
        Logger(INFO) << "t = " << t;
        //output_particles(root);
        ++step;
        // rendering
        if (step%renderer->getRenderInterval()==0)
        {
            //Particle prtcls[renderer->getNumParticles()];
            Particle *prtcls;
            int N = get_particle_array(root, prtcls);
            Logger(DEBUG) << "Rendering timestep #" << step << ": N = " << N;
            renderer->createFrame(image, hdImage, prtcls, step);
        }

        t += delta_t; // update timestep
        compX_BH(root, delta_t);
        //compF_BH(root, root, getSystemSize(&box));
        compV_BH(root, delta_t);
        repairTree(root);
    }
    Logger(INFO) << "t = " << t << ", DONE.";
    //output_particles(root);
}


void timeIntegration_BH_par(float t, float delta_t, float t_end, float diam, TreeNode *root, SubDomainKeyTree *s,
                            Renderer *renderer, char *image, double *hdImage) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box));
    //output_particles(root);

    int step = 0;

    while (t < t_end) {
        Logger(DEBUG) << " ";
        Logger(DEBUG) << "t = " << t;
        Logger(DEBUG) << "--------------------------";
        //output_particles(root);

        // rendering
        if (step%renderer->getRenderInterval()==0)
        {
            //Particle prtcls[renderer->getNumParticles()];
            Particle *prtcls;
            int N = gatherParticles(root, s, prtcls);
            if (s->myrank == 0) {
                Logger(DEBUG) << "Rendering timestep #" << step << ": N = " << N;
                renderer->setNumParticles(N);
                renderer->createFrame(image, hdImage, prtcls, step);
                delete [] prtcls;
            }
        }
        ++step;

        t += delta_t; // update timestep
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        compF_BHpar(root, diam, s);
        repairTree(root); // cleanup local tree by removing symbolicForce-particles

        compX_BH(root, delta_t);

        compV_BH(root, delta_t);

        moveParticles_BH(root);

        //Logger(DEBUG) << "BEFORE sendParticles()";
        //output_tree(root, true);

        sendParticles(root, s);
        compLocalPseudoParticlespar(root);

        output_tree(root, true, true);
        //Logger(DEBUG) << "AFTER sendParticles()";
        //repairTree(root);
    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
    //output_particles(root);
}
