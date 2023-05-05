#ifndef MY_UTILS_H_
#define MY_UTILS_H_

#include <string.h>
#include <stdlib.h>

struct StartupOptions {
    bool usingTMP;
    bool usingPip;
    int numMicroBatch;
    int numBatchSize;
    std::string inputFile;
    std::string outputFile;
};

std::string removeQuote(std::string input) {
  if (input.length() > 0 && input.front() == '\"')
    return input.substr(1, input.length() - 2);
  return input;
}

StartupOptions parseOptions(int argc, char **argv) {
    StartupOptions rs;
    rs.usingPip = false;
    rs.usingTMP = false;
    rs.outputFile = "";
    rs.inputFile = "";
    rs.numMicroBatch = 4;
    rs.numBatchSize = 32;

    for (int i = 1; i < argc; i++) {
        if (i < argc - 1) {
            if (strcmp(argv[i], "-micro") == 0)
                rs.numMicroBatch = atoi(argv[i + 1]);
            else if (strcmp(argv[i], "-batch") == 0)
                rs.numBatchSize = atoi(argv[i + 1]);
            // else if (strcmp(argv[i], "-c") == 0)
            //   rs.checkCorrectness = true;
            else if (strcmp(argv[i], "-in") == 0)
              rs.inputFile = removeQuote(argv[i + 1]);
            // else if (strcmp(argv[i], "-n") == 0)
            //   rs.numParticles = atoi(argv[i + 1]);
            // else if (strcmp(argv[i], "-v") == 0)
            //   rs.viewportRadius = (float)atof(argv[i + 1]);
            else if (strcmp(argv[i], "-out") == 0)
              rs.outputFile = argv[i + 1];
            // else if (strcmp(argv[i], "-fo") == 0) {
            //   rs.bitmapOutputDir = removeQuote(argv[i + 1]);
            //   rs.frameOutputStyle = FrameOutputStyle::AllFrames;
            // } else if (strcmp(argv[i], "-ref") == 0)
            //   rs.referenceAnswerDir = removeQuote(argv[i + 1]);
        }
        if (strcmp(argv[i], "-tmp") == 0) {
            rs.usingTMP = true;
        } else if (strcmp(argv[i], "-pip") == 0) {
            rs.usingPip = true;
        }
    }
    // check if available
    if (rs.usingPip) {
        if (rs.numMicroBatch <= 0)
            throw std::runtime_error("numMicroBatch <= 0 when using pipeline model parallelism.");
        if (rs.numBatchSize % rs.numMicroBatch != 0)
            throw std::runtime_error("numBatchSize should be a multiple of numMicroBatch when using pipeline model parallelism.");
    }

    return rs;
}

#endif