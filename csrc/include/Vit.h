#ifndef VIT_H_
#define VIT_H_

#include <mpi.h>

#include "transformer_cube.h"
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
    //     Linear embedding;
    //     Linear classification;
    //     LayerNormalization layer_norm;
    //     GlobalAveragePooling pooling;
public:
    VisionTransformer(StartupOptions options, int num_hidden_layers, int num_attention_heads, int num_classes, int patch_size, int num_channels, int hidden_dim)
        : options(options), num_hidden_layers(num_hidden_layers), num_attention_heads(num_attention_heads), num_classes(num_classes), patch_size(patch_size), num_channels(num_channels), hidden_dim(hidden_dim)
    //   embedding(hidden_dim, patch_size * patch_size * num_channels),
    //   classification(num_classes, hidden_dim),
    //   layer_norm(hidden_dim)
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
                    my_stage.layers.emplace_back(hidden_dim, hidden_dim, num_attention_heads);
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
                // TODO: using pipeline and tensor
            }
        }
        else
        {
            if (!options.usingTMP)
            {
                for (int i = 0; i < num_hidden_layers; ++i)
                {
                    my_stage.layers.emplace_back(hidden_dim, hidden_dim, num_attention_heads);
                }
            }
            else
            {
                // TODO: only using tensor
            }
        }
    }
    Cube forward(const Cube &input)
    {
        // // Tokenization
        // Cube tokens = input.tokenize(patch_size);

        // // Embedding
        // Cube embedded_tokens = embedding(tokens);

        // // Add positional encodings (assuming you have a custom function to generate positional encodings)
        // Cube positional_encodings = generate_positional_encodings(embedded_tokens.shape());
        // embedded_tokens += positional_encodings;

        Cube output(input.batch_size, input.rows, input.cols);

        if (options.usingPip)
        {
            if (!options.usingTMP)
            {
                int total_time_slices = num_stages + num_micro - 1;
                Cube my_input(input.batch_size / num_micro, input.rows, input.cols);
                MPI_Barrier(MPI_COMM_WORLD);
                for (int t = 0; t < total_time_slices; t++)
                { // all worker work total_time_slices time slices
                    if (pid <= t && t < pid + num_micro)
                    { // each worker works num_micro time slices
                        // Step 1: get my input data
                        if (first_stage)
                        {
                            // get micro batch from input batch
                            int micro_batch_low = (t - pid) * num_micro;
                            int micro_batch_high = (t - pid + 1) * num_micro;
                            std::copy(input.data+micro_batch_low * input.rows * input.cols, input.data+micro_batch_high * input.rows * input.cols, my_input.data);
                        }
                        else
                        {
                            // get micro batch from prev worker
                            MPI_Recv(my_input.data, my_input.batch_size * my_input.rows * my_input.cols, MPI_FLOAT, pid - 1, 0, MPI_COMM_WORLD, nullptr);
                        }
                        // Step 2: forward my input data
                        Cube my_output = my_stage.forward(my_input);
                        // Step 3: send output
                        if (last_stage)
                        {
                            // generate output
                            int micro_batch_low = (t - pid) * num_micro;
                            std::copy(my_output.data, my_output.data +my_output.batch_size * my_output.rows * my_output.cols, output.data + micro_batch_low * output.rows * output.cols);
                        }
                        else
                        {
                            // send micro batch to next worker
                            MPI_Request request;
                            MPI_Isend(my_output.data, my_output.batch_size * my_output.rows * my_output.cols, MPI_FLOAT, pid + 1, 0, MPI_COMM_WORLD, &request);
                        }
                    }
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                // result is stored in last_stage worker, FIXME: broadcast to all or just to pid 0?
                MPI_Bcast(output.data, output.batch_size * output.rows * output.cols, MPI_FLOAT, num_stages - 1, MPI_COMM_WORLD);
            }
            else
            {
                // TODO: using pipeline and tensor
            }
        }
        else
        {
            if (!options.usingTMP)
            {
                output = my_stage.forward(input);
            }
            else
            {
                // TODO: only using tensor
            }
        }

        // // Layer normalization
        // Cube normalized_tokens = layer_norm(embedded_tokens);

        // // Global average pooling
        // Cube pooled = pooling(normalized_tokens);

        // // Classification
        // Cube logits = classification(pooled);
        return output;
    }

    void backward(const Cube &targets);
};

#endif