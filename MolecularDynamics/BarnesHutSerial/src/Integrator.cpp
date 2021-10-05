//
// Created by Michael Staneker on 15.03.21.
//

#include "../include/Integrator.h"

void timeIntegration_BH(float t, float delta_t, float t_end, TreeNode *root, Box box,
                        Renderer *renderer, char *image, double *hdImage,
                        bool render, bool h5Dump, int h5DumpEachTimeSteps) {
    //compF_basis(p, N);
    //compF_BH(root, root, getSystemSize(&box)); //TODO: ?
    //output_particles(root);

    int step = 0;

    compF_BH(root, root, getSystemSize(&box));

    while (t <= t_end) {
        Logger(INFO) << "t = " << t;
        //output_particles(root);
        ++step;

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
        if (render && step%renderer->getRenderInterval()==0)
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
    Logger(INFO) << "t = " << t << ", DONE.";
    //output_particles(root);
}
