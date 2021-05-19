#include "../include/H5Renderer.h"

H5Renderer::H5Renderer(std::string _h5folder, double _systemSize, int _imgHeight, bool _processColoring) :
h5folder { _h5folder }, systemSize { _systemSize }, imgHeight { _imgHeight }, processColoring { _processColoring },
h5files { std::vector<fs::path>() }
{
    // gather files found at h5folder
    fs::path h5path ( h5folder );
    if( !fs::exists(h5path) ){
        Logger(ERROR) << "Bad provided path for 'h5folder/*.h5': '" << h5folder << "' doesn't exist.";
    } else if ( !fs::is_directory(h5path) ){
        Logger(ERROR) << "Bad provided path for 'h5folder/*.h5': '" << h5folder << "' is not a directory.";
    } else {
        Logger(INFO) << "Collecting h5 files from '" << h5folder << "' ...";
        fs::directory_iterator endDirIt; // empty iterator serves as end
        fs::directory_iterator dirIt(h5path);
        while (dirIt != endDirIt){
            // TODO: also allow .hdf, .hdf5 etc.
            if(fs::extension(dirIt->path()) == ".h5"){
                // collect h5files in h5folder dir in container
                h5files.push_back(dirIt->path());
                Logger(INFO) << "Found " << dirIt->path().filename();
            }
            ++dirIt;
        }
        Logger(INFO) << "... done.";
    }

    // initialize pixelspace
    psSize = 2*imgHeight*imgHeight;
    pixelSpace = new ColorRGB[psSize];
}

// public functions

H5Renderer::~H5Renderer(){
    delete [] pixelSpace;
}

void H5Renderer::createImages(std::string outDir){

    // loop through all found h5 files
    for(auto const &path: h5files){

        Logger(INFO) << "Reading " << path.filename() << " ...";

        // opening file
        HighFive::File file(path.string(), HighFive::File::ReadOnly);

        // reading process ranges
        HighFive::DataSet rng = file.getDataSet("/hilbertRanges");
        std::vector<unsigned long> ranges;
        rng.read(ranges);

        // reading particle keys
        HighFive::DataSet key = file.getDataSet("/hilbertKey");
        std::vector<unsigned long> k;
        key.read(k);

        // reading particle positions
        HighFive::DataSet pos = file.getDataSet("/x");
        std::vector<std::vector<double>> x; // container for particle positions
        pos.read(x);

        Logger(INFO) << "... looping through particles ...";

        // looping through particles
        for (int i=0; i<x.size(); ++i){
            // process coloring
            ColorRGB color = procColor(k[i], ranges);
            particle2Pixel(x[i][0], x[i][1], x[i][2], color);
        }

        std::string outFile = outDir + "/" + path.stem().string() + ".ppm";
        Logger(INFO) << "... writing to file '" << outFile << "' ...";
        // writing pixelSpace to png file
        pixelSpace2File(outFile);

        // clear pixel space
        clearPixelSpace();
        Logger(INFO) << "... done.";
    }
}

// private functions
ColorRGB H5Renderer::procColor(unsigned long k, const std::vector<unsigned long> &ranges){
    for(int proc=0; proc < ranges.size()-1; ++proc){
        if (k > ranges[proc] && k < ranges[proc+1]){
            // particle belongs to process proc
            return COLORS[proc];
        }
    }
    return ColorRGB(); // black
}

void H5Renderer::clearPixelSpace(){
    for(int px=0; px < psSize; ++px){
        pixelSpace[px] = ColorRGB();
    }
}

int H5Renderer::pos2pixel(double pos){
    return round(imgHeight/2. * (1. + pos/systemSize*SCALE2FIT));
}

void H5Renderer::particle2Pixel(double x, double y, double z, const ColorRGB &color){
    // convert to pixel space
    int xPx = pos2pixel(x);
    int yPx = pos2pixel(y);
    int zPx = pos2pixel(z);

    // draw in x-y plane
    pixelSpace[xPx+2*imgHeight*yPx] = color;
    // draw in x-z plane
    pixelSpace[xPx+2*imgHeight*zPx+imgHeight] = color;
}

void H5Renderer::pixelSpace2File(const std::string &outFile){
    // using *.ppm
    // https://en.wikipedia.org/wiki/Netpbm#File_formats

    std::ofstream file (outFile, std::ofstream::binary);

    // flatten ColorRGB struct
    char pxData[3*psSize];
    for (int px=0; px < 3*psSize; px+=3){
        pxData[px] = pixelSpace[px/3].r;
        pxData[px+1] = pixelSpace[px/3].g;
        pxData[px+2] = pixelSpace[px/3].b;
    }

    if (file.is_open()){
        file << "P6\n" << 2*imgHeight << " " << imgHeight << "\n" << "255\n";
        file.write(pxData, psSize*3);
        file.close();
    }
}







