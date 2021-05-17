//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void finalize(TreeNode *root) {
    outputTree(root);
    MPI_Finalize();
}

void timeIntegration_BH_par(float t, float delta_t, float t_end, float diam, TreeNode *root, SubDomainKeyTree *s,
                            Renderer *renderer, char *image, double *hdImage, bool render, bool processColoring,
                            bool h5Dump, int h5DumpEachTimeSteps) {

    int step = 0;

    while (t <= t_end) {
        Logger(DEBUG) << " ";
        Logger(DEBUG) << "t = " << t;
        Logger(DEBUG) << "============================";

        Logger(DEBUG) << "--------------------------";
        Logger(DEBUG) << "Load balancing ... ";

        //Logger(DEBUG) << "OLD Ranges:";
        //for (int i=0; i<=s->numprocs; i++){
        //    Logger(DEBUG) << "range[" << i << "] = " << s->range[i];
        //}

        //outputTree(root, "log/beforeLBproc" + std::to_string(s->myrank), true, false);

        newLoadDistribution(root, s); // calculate new load distribution

        //outputTree(root, "log/afterNLDproc" + std::to_string(s->myrank), true, false);

        // update tree with new ranges
        clearDomainList(root);

        createDomainList(root, 0, 0UL, s);

        sendParticles(root, s);

        compPseudoParticlesPar(root, s);

        Logger(DEBUG) << "NEW Ranges:";
        for (int i=0; i<=s->numprocs; i++){
            Logger(DEBUG) << "range[" << i << "] = " << s->range[i];
        }

        //outputTree(root, "log/balanced_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);
        //outputTree(root, "log/endTSproc" + std::to_string(s->myrank), true, false);

        outputTree(root, false, false);

        Logger(DEBUG) << "... done.";

        Logger(DEBUG) << "--------------------------";

        if (h5Dump && step % h5DumpEachTimeSteps==0){
            Logger(DEBUG) << "Dump particles to h5 file ...";

            std::stringstream filename;
            filename << std::setw(6) << std::setfill('0') << step;

            // open a new file with the MPI IO driver for parallel Read/Write
            HighFive::File h5File("output/ts" + filename.str() + ".h5",
                                  HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                                  HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

            // get global particle count
            long Nproc = countParticles(root);
            long N = 0;
            MPI_Allreduce(&Nproc, &N, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

            // as this is directly after load balancing, the particle count per process is known
            const int ppp = (N % s->numprocs != 0) ? N/s->numprocs+1 : N/s->numprocs; // particles per process
            std::vector<size_t> dataSpaceDims(2);
            dataSpaceDims[0] = std::size_t(N); // number of particles
            dataSpaceDims[1] = DIM;

            //TODO: write ranges to file to recover process of particle
            //HighFive::DataSet ranges = ;

            // create data sets to be filled with particle data
            HighFive::DataSet pos = h5File.createDataSet<double>("/x", HighFive::DataSpace(dataSpaceDims));
            HighFive::DataSet vel = h5File.createDataSet<double>("/v", HighFive::DataSpace(dataSpaceDims));
            HighFive::DataSet key = h5File.createDataSet<unsigned long>("/hilbertKey", HighFive::DataSpace(N));

            particles2file(root, &pos, &vel, &key, s);

            Logger(DEBUG) << "NUMBER OF PARTICLES = " << N;

            Logger(DEBUG) << "...done";
            Logger(DEBUG) << "--------------------------";
        }

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
            //outputTree(root, false);
            Logger(DEBUG) << "--------------------------";
        }

        if (t == t_end){
            /*if (outFile != ""){
                particles2file(root, outFile, s);
            }*/
            break; // done after rendering of last step
        }

        ++step;

        t += delta_t; // update timestep

        //outputTree(root, "log/before_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);

        compF_BHpar(root, diam, s);
        repairTree(root); // cleanup local tree by removing symbolicForce-particles

        compX_BH(root, delta_t);

        compV_BH(root, delta_t);

        //outputTree(root, "log/before_move" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        setFlags(root);
        moveLeaf(root, root);

        //outputTree(root, "log/after_move" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        repairTree(root);

        //outputTree(root, "log/after_repair" + std::to_string(step) + "proc" + std::to_string(s->myrank),
        //            true, false);

        //moveParticles_BH(root);

        sendParticles(root, s);

        compPseudoParticlesPar(root, s);

        outputTree(root, false, false);

        Logger(DEBUG) << "============================";
    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
}
