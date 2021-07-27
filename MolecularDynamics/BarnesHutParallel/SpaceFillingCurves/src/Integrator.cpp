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
                            bool h5Dump, int h5DumpEachTimeSteps, int loadBalancingInterval) {

    int step = 0;
    H5Profiler &profiler = H5Profiler::getInstance();

    double t1, t2; // timing variables

    while (t <= t_end) {
        Logger(INFO) << " ";
        Logger(INFO) << "t = " << t << ", step = " << step;
        Logger(DEBUG) << "============================";

        std::stringstream stepss;
        stepss << std::setw(6) << std::setfill('0') << step;

        profiler.setStep(step);

        if (step % loadBalancingInterval==0) {

            Logger(DEBUG) << "--------------------------";
            t1 = MPI_Wtime();
            Logger(DEBUG) << "Load balancing ... ";

            profiler.setStep(step/loadBalancingInterval);
            profiler.time();

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

            //profiler.disableWrite();
            sendParticles(root, s);
            //profiler.enableWrite();

            compPseudoParticlesPar(root, s);

            profiler.time2file("/loadBalancing/totalTime", s->myrank);
            profiler.setStep(step);

            Logger(DEBUG) << "NEW Ranges:";
            for (int i = 0; i <= s->numprocs; i++) {
                Logger(DEBUG) << "range[" << i << "] = " << s->range[i];
            }

            //outputTree(root, "log/balanced_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);
            //outputTree(root, "log/endTSproc" + std::to_string(s->myrank), true, false);

            Logger(DEBUG) << "... done.";
            t2 = MPI_Wtime();
            Logger(INFO) << "++++++++++++++++++++++++++++ Load balancing: " << t2-t1 << "s";
            Logger(DEBUG) << "--------------------------";
        }

        outputTree(root, false, false);
        profiler.value2file("/general/numberOfParticles", s->myrank, countParticles(root));

        if (h5Dump && step % h5DumpEachTimeSteps==0){
            Logger(DEBUG) << "Dump particles to h5 file ...";

            // open a new file with the MPI IO driver for parallel Read/Write
            HighFive::File h5File("output/ts" + stepss.str() + ".h5",
                                  HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate,
                                  HighFive::MPIOFileDriver(MPI_COMM_WORLD, MPI_INFO_NULL));

            // get global particle count
            long Nproc = countParticles(root);
            long N = 0;
            MPI_Allreduce(&Nproc, &N, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

            // as this is directly after load balancing, the particle count per process is known
            std::vector<size_t> dataSpaceDims(2);
            dataSpaceDims[0] = std::size_t(N); // number of particles
            dataSpaceDims[1] = DIM;

            //write ranges to file to recover process of particle
            HighFive::DataSet ranges = h5File.createDataSet<unsigned long>("/hilbertRanges",
                                                                           HighFive::DataSpace(s->numprocs+1));
            ranges.write(s->range);

            // create data sets to be filled with particle data
            HighFive::DataSet pos = h5File.createDataSet<double>("/x", HighFive::DataSpace(dataSpaceDims));
            HighFive::DataSet vel = h5File.createDataSet<double>("/v", HighFive::DataSpace(dataSpaceDims));
            HighFive::DataSet key = h5File.createDataSet<unsigned long>("/hilbertKey", HighFive::DataSpace(N));

            particles2file(root, &pos, &vel, &key, s);

            Logger(DEBUG) << "NUMBER OF PARTICLES = " << N;

            Logger(DEBUG) << "...done";
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
        }

        if (t >= t_end){
            /*if (outFile != ""){
                particles2file(root, outFile, s);
            }*/
            break; // done after rendering of last step
        }
        Logger(DEBUG) << "--------------------------";
        Logger(DEBUG) << "Computing time step ...";

        ++step;

        t += delta_t; // update timestep

        //outputTree(root, "log/before_step" + std::to_string(step) + "proc" + std::to_string(s->myrank), true, false);

        Logger(DEBUG) << "... force calculation ...";
        t1 = MPI_Wtime();

        profiler.time();
        compF_BHpar(root, diam, s);
        repairTree(root); // cleanup local tree by removing symbolicForce-particles
        profiler.time2file("/forceComputation/totalTime", s->myrank);

        t2 = MPI_Wtime();
        Logger(INFO) << "+++++++++++++++++++++++++ Force calculation: " << t2-t1 << "s";

        Logger(DEBUG) << "... updating positions and velocities ...";
        t1 = MPI_Wtime();

        profiler.time();
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

        profiler.disableWrite();
        sendParticles(root, s);
        profiler.enableWrite();

        compPseudoParticlesPar(root, s);
        profiler.time2file("/updatePosVel/totalTime", s->myrank);

        t2 = MPI_Wtime();
        Logger(INFO) << "++++++++++++++ Position and velocity update: " << t2-t1 << "s";
        Logger(DEBUG) << "... done.";

        //outputTree(root, false, false);

        Logger(DEBUG) << "============================";
    }
    Logger(DEBUG) << "t = " << t << ", FINISHED";
}
