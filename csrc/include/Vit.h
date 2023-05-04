#ifndef VIT_H_
#define VIT_H_

#include <mpi.h>

#include "transformer_cube.h"
#include "transformer_tmp_cube.h"
#include "transformer_openmp.h"
#include "cube.h"
#include "sequential.h"
#include "utils.h"

class VisionTransformer
{

private:
    StartupOptions options;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int num_classes = 10;
    int patch_size = 16;
    int num_channels = 3;
    int hidden_dim = 768;

    // for pipeline
    int num_stages = -1;
    int num_micro = -1;
    int pid = -1;
    bool first_stage = false;
    bool last_stage = false;
    Sequential my_stage;
public:
    VisionTransformer(StartupOptions options, int num_hidden_layers, int num_attention_heads, int num_classes, int patch_size, int num_channels, int hidden_dim)
        : options(options), num_hidden_layers(num_hidden_layers), num_attention_heads(num_attention_heads), num_classes(num_classes), patch_size(patch_size), num_channels(num_channels), hidden_dim(hidden_dim)
    {
        int nproc = 0;
        if (options.usingPip || options.usingTMP) {
            MPI_Comm_rank(MPI_COMM_WORLD, &pid);
            MPI_Comm_size(MPI_COMM_WORLD, &nproc);
        }
        if (options.usingPip) {
            num_micro = options.numMicroBatch;
            int blocks_per_stage = 0;
            if (!options.usingTMP) {
                // only using pipeline
                num_stages = nproc;
                if (num_hidden_layers % num_stages != 0 || num_hidden_layers < num_stages)
                    throw std::runtime_error("Only support num_hidden_layers is a multiple of num_stages.");
                blocks_per_stage = num_hidden_layers / num_stages;
                // Wrap the transformer blocks into stages
                my_stage.layers.reserve(blocks_per_stage);
                for (int i = 0; i < blocks_per_stage; i++) {
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() add layer %d\n", pid, i);
#endif    
                    TransformerEncoderLayer_cube* layer = new TransformerEncoderLayer_cube(hidden_dim, hidden_dim, num_attention_heads);
                    my_stage.layers.push_back(layer);
                }
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() finish adding layer\n", pid);
#endif
                if (pid == 0)
                    first_stage = true;
                if (pid == num_stages - 1)
                    last_stage = true;
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() first_stage=%d last_stage=%d\n", pid, first_stage, last_stage);
#endif
            }
            else
            {
                // Using pipeline and tensor
                printf("============Pipeline and OpenMP TMP============\n");
                num_stages = nproc;
                if (num_hidden_layers % num_stages != 0 || num_hidden_layers < num_stages)
                    throw std::runtime_error("Only support num_hidden_layers is a multiple of num_stages.");
                blocks_per_stage = num_hidden_layers / num_stages;
#ifdef DEBUG
                printf("num_stages=%d blocks_per_stage=%d\n", num_stages, blocks_per_stage);
#endif
                // Wrap the transformer blocks into stages
                my_stage.layers.reserve(blocks_per_stage);
                for (int i = 0; i < blocks_per_stage; i++) {
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() add layer %d\n", pid, i);
#endif    
                    int num_workers = 12;
                    TransformerEncoderLayer_openmp* layer = new TransformerEncoderLayer_openmp(hidden_dim, hidden_dim, num_attention_heads, num_workers);
                    my_stage.layers.push_back(layer);
                }
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() finish adding layer\n", pid);
#endif
                if (pid == 0)
                    first_stage = true;
                if (pid == num_stages - 1)
                    last_stage = true;
#ifdef DEBUG
                printf("[Worker %d] VisionTransformer() first_stage=%d last_stage=%d\n", pid, first_stage, last_stage);
#endif
            }
        }
        else
        {
            if (!options.usingTMP)
            {
                //not using pipeline and tensor
                for (int i = 0; i < num_hidden_layers; ++i)
                {
                    TransformerEncoderLayer_cube* layer = new TransformerEncoderLayer_cube(hidden_dim, hidden_dim, num_attention_heads);
                    my_stage.layers.push_back(layer);
                }
            }
            else
            {
                // TODO: only using tensor
                for (int i = 0; i < num_hidden_layers; ++i)
                {
                    TransformerEncoderLayerTMP_CUBE* layer = new TransformerEncoderLayerTMP_CUBE(hidden_dim, hidden_dim, num_attention_heads, pid, nproc);
                    my_stage.layers.push_back(layer);
                }
            }
        }
    }
    ~VisionTransformer()
    {
        for (int i = 0; i < my_stage.layers.size(); ++i)
        {
            delete my_stage.layers[i];
        }
    }
    Cube forward(const Cube &input)
    {
        Cube output(input.batch_size, input.rows, input.cols);

        if (options.usingPip)
        {
            float communication_time = 0.f;

            int total_time_slices = num_stages + num_micro - 1;
            Cube my_input(input.batch_size / num_micro, input.rows, input.cols);
            MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
            if (pid == 0){
                printf("[MASTER] VisionTransformer() total_time_slices=%d\n", total_time_slices);
                fflush(stdout);
            }
#endif
            for (int t = 0; t < total_time_slices; t++)
            { // all worker work total_time_slices time slices
#ifdef DEBUG
                if (pid == 0){
                    printf("[MASTER] VisionTransformer() start pipeline on %d time slice\n", t);
                    fflush(stdout);
                }
                MPI_Barrier(MPI_COMM_WORLD);
#endif
                if (pid <= t && t < pid + num_micro)
                { // each worker works num_micro time slices
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() start to work\n", pid);
                    fflush(stdout);
#endif
                    // Step 1: get my input data
                    auto recv_start = std::chrono::system_clock::now();
                    if (first_stage)
                    {
                        // get micro batch from input batch
                        int micro_batch_low = (t - pid) * my_input.batch_size;
                        int micro_batch_high = (t - pid + 1) * my_input.batch_size;
                        std::copy(
                            input.data+micro_batch_low * input.rows * input.cols,
                            input.data+micro_batch_high * input.rows * input.cols,
                            my_input.data);
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() finish getting input data from input\n", pid);
                    fflush(stdout);
#endif
                    }
                    else
                    {
                        // get micro batch from prev worker
                        MPI_Recv(my_input.data, my_input.batch_size * my_input.rows * my_input.cols, MPI_FLOAT, pid - 1, 0, MPI_COMM_WORLD, nullptr);
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() finish getting input data from %d\n", pid, pid - 1);
                    fflush(stdout);
#endif
                    }
                    auto recv_end = std::chrono::system_clock::now();
                    communication_time += std::chrono::duration<float>(recv_end - recv_start).count();
                    // printf("[Worker %d] recv_input time:\t%.6fs\n", pid, std::chrono::duration<float>(recv_end - recv_start).count());

                    // Step 2: forward my input data
                    Cube my_output = my_stage.forward(my_input);
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() finish forwarding\n", pid);
                    fflush(stdout);
#endif
                    // Step 3: send output
                    auto send_output_start = std::chrono::system_clock::now();
                    if (last_stage)
                    {
                        // generate output
                        int micro_batch_low = (t - pid) * my_output.batch_size;
                        std::copy(
                            my_output.data,
                            my_output.data +my_output.batch_size * my_output.rows * my_output.cols,
                            output.data + micro_batch_low * output.rows * output.cols);
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() finish generating output data\n", pid);
                    fflush(stdout);
#endif
                    }
                    else
                    {
                        // send micro batch to next worker
                        MPI_Request request;
                        MPI_Isend(my_output.data, my_output.batch_size * my_output.rows * my_output.cols, MPI_FLOAT, pid + 1, 0, MPI_COMM_WORLD, &request);
#ifdef DEBUG
                    printf("[Worker %d] VisionTransformer() finish sending output data to %d\n", pid, pid+1);
                    fflush(stdout);
#endif
                    }
                    auto send_output_end = std::chrono::system_clock::now();
                    // printf("[Worker %d] send_output time:\t%.6fs\n", pid, std::chrono::duration<float>(send_output_end - send_output_start).count());
                    communication_time += std::chrono::duration<float>(send_output_end - send_output_start).count();
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
            // result is stored in last_stage worker, FIXME: broadcast to all or just to pid 0?
            MPI_Bcast(output.data, output.batch_size * output.rows * output.cols, MPI_FLOAT, num_stages - 1, MPI_COMM_WORLD);
            printf("[Worker %d] ViT communication time:\t%.6fs\n", pid, communication_time);
            fflush(stdout);
        }
        else
        {
            if (!options.usingTMP)
            {
                output = my_stage.forward(input);
            }
            else
            {
                // only using tensor
                output = my_stage.forward(input);
            }
        }
        return output;
    }

    void backward(const Cube &targets);
};

#endif
