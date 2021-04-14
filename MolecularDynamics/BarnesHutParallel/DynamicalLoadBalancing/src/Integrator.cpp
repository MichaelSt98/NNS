//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void finalize(TreeNode *root) {
    output_tree(root);
    MPI_Finalize();
}

void timeIntegration_BH_par(float t, float delta_t, float t_end, float diam, TreeNode *root, SubDomainKeyTree *s,
                            Renderer *renderer, char *image, double *hdImage, bool render, bool processColoring) {

    int step = 0;

    while (t <= t_end) {
        Logger(DEBUG) << " ";
        Logger(DEBUG) << "t = " << t;
        Logger(DEBUG) << "============================";

        Logger(DEBUG) << "--------------------------";
        Logger(DEBUG) << "Load balancing ... ";

        newLoadDistribution(root, s); // calculate new load distribution

        // update tree with new ranges
        clearDomainList(root);

        createDomainList(root, 0, 0, s);

        sendParticles(root, s);

        compPseudoParticlespar(root, s);

        //output_tree(root, "log/balanced_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);


        Logger(DEBUG) << "... done.";
        Logger(DEBUG) << "--------------------------";

        // rendering
        if (render && step % renderer->getRenderInterval()==0)
        {
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
            //output_tree(root, false);
        }

        if (t == t_end){
            break; // done
        }

        ++step;

        t += delta_t; // update timestep

        //output_tree(root, "log/before_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);

        compF_BHpar(root, diam, s);
        repairTree(root); // cleanup local tree by removing symbolicForce-particles

        compX_BH(root, delta_t);

        compV_BH(root, delta_t);

        //output_tree(root, "log/before_move" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        setFlags(root);
        moveLeaf(root, root);

        //output_tree(root, "log/after_move" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        repairTree(root);

        //output_tree(root, "log/after_repair" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        //moveParticles_BH(root);

        sendParticles(root, s);

        compPseudoParticlespar(root, s);

        output_tree(root, false, false);

        Logger(DEBUG) << "============================";
    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
}
