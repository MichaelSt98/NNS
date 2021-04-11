//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void finalize(TreeNode *root) {
    output_tree(root);
    MPI_Finalize();
}

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
            renderer->createFrame(image, hdImage, prtcls, step, &root->box);
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
                            Renderer *renderer, char *image, double *hdImage, bool processColoring) {
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
            int *prtN;
            int N;
            //int N = gatherParticles(root, s, prtcls);
            if (processColoring) {
                N = gatherParticles(root, s, prtcls, prtN);
            }
            else {
                N = gatherParticles(root, s, prtcls);
            }
            //for (int i=0; i<N; i++) {
                //if (prtN[i] < 0) {
                    //Logger(INFO) << "Process number: " << prtN[i];
                //}
            //}
            if (s->myrank == 0) {
                Logger(INFO) << "Rendering timestep #" << step << ": N = " << N;
                renderer->setNumParticles(N);
                //renderer->createFrame(image, hdImage, prtcls, step, &root->box);
                if (processColoring) {
                    renderer->createFrame(image, hdImage, prtcls, prtN, s->numprocs, step, &root->box);
                    delete[] prtN;
                }
                else {
                    renderer->createFrame(image, hdImage, prtcls, step, &root->box);
                }
                delete [] prtcls;
            }
            output_tree(root, false);
            /*if (s->myrank == 0) {
                if (N != 100) {
                    output_tree(root, true, false);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                    //exit(0);
                }
            }*/
        }
        ++step;

        t += delta_t; // update timestep
        //std::cout << "\nroot->p.x = (" << root->p.x[0] << ", " << root->p.x[1] << ", " << root->p.x[2] << ")" << std::endl;
        //compF_BHpar(root, diam, s);
        //repairTree(root); // cleanup local tree by removing symbolicForce-particles

        compF_BHpar(root, diam, s);
        repairTree(root); // cleanup local tree by removing symbolicForce-particles

        compX_BH(root, delta_t);

        compV_BH(root, delta_t);

        moveParticles_BH(root);

        //output_tree(root, true, false);
        //setFlags(root);
        //output_tree(root, "log/" + std::to_string(step-1) + "flag" + std::to_string(s->myrank), true, false);
        //moveLeaf(root, root);
        //output_tree(root, "log/" + std::to_string(step-1) + "leaf" + std::to_string(s->myrank),true, false);
        //output_tree(root, true, false);

        //repairTree(root);
        //output_tree(root, "log/" + std::to_string(step-1) + "repa" + std::to_string(s->myrank),true, false);

        //Logger(DEBUG) << "BEFORE sendParticles()";
        //output_tree(root, false);

        sendParticles(root, s);

        //compLocalPseudoParticlespar(root);
        compPseudoParticlespar(root, s);

        output_tree(root, true, false);

        //output_tree(root, false, false);
        //Logger(DEBUG) << "AFTER sendParticles()";
        //repairTree(root);
    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
    //output_particles(root);
}
