//
// Created by Michael Staneker on 22.02.21.
//

#ifndef CUDA_PARAMETERS_H
#define CUDA_PARAMETERS_H

typedef enum Model {

    disk_model,
    colliding_disk_model,
    plummer_model

} Model;


typedef struct SimulationParameters {

    Model model;
    bool opengl;
    bool debug;
    bool benchmark;
    bool fullscreen;
    float iterations;
    float timestep;
    float gravity;
    float dampening;

} SimulationParameters;

#endif //CUDA_PARAMETERS_H
