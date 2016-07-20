#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <abhsf/utils/colors.h>

using map_t = std::map<std::pair<uint32_t, uint32_t>, uintmax_t>;

struct stat_t 
{
    map_t map;
    uintmax_t min;
};

bool is_directory(const std::string& path)
{
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0)
        return 0;
    return S_ISDIR(statbuf.st_mode);
}

int main(int argc, char* argv[])
{
    std::cout << "Scheme: " << cyan;
    int scheme = atoi(argv[1]);
    std::cout << ((scheme == 0) ? "min-fixed" : "adaptive");
    std::cout << reset << std::endl;

    std::cout << "Precision " << yellow;
    int precision = atoi(argv[2]);
    std::cout << ((precision == 0) ? "single" : "double");
    std::cout << reset << std::endl;

    int n = atoi(argv[3]);
    std::cout << "Number of matrices: " << magenta << n << reset << std::endl;

    std::vector<stat_t> stats;

    DIR* dir;
    if ((dir = opendir(".")) == nullptr)
        throw std::runtime_error("Error running opendir().");

    struct dirent* ent;
    while ((ent = readdir(dir)) != nullptr) {
        std::string path(ent->d_name);
        if (is_directory(path) && path != "." && path != "..") {
            std::ifstream f;

            if (precision == 0)
                f.open(path + "/mmf-single");
            else
                f.open(path + "/mmf-double");

            map_t map;
            uintmax_t min = 0;

            for (int k = 0; k < 64; k++) {
                uintmax_t h, w, coo32, csr32, coo, csr, bitmap, dense, adaptive, adaptive134;
                f >> h >> w >> coo32 >> csr32 >> coo >> csr >> bitmap >> dense >> adaptive >> adaptive134;

                uintmax_t minfixed = std::min(std::min(coo, csr), std::min(bitmap, dense)) + 2;

                if (min == 0)
                    min = std::min(minfixed - 2, adaptive);
                else 
                    min = std::min(min, std::min(minfixed - 2, adaptive));

                auto key = std::make_pair(h, w);
                if (scheme == 0)
                    map[key] = minfixed;
                else 
                    map[key] = adaptive;
            }

            stat_t stat;
            stat.map = std::move(map);
            stat.min = min;
            stats.push_back(stat);

            f.close();
        }
    }

    closedir(dir);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(stats.begin(), stats.end(), g);

    double avg_64 = 0.0;
    double max_64 = 0.0;
    double avg_8 = 0.0;
    double max_8 = 0.0;

    for (int k = 0; k < n; k++) {
        auto min = stats[k].min;

        // B_64:
        uintmax_t tmin = 0;
        for (int h = 2; h <= 256; h *= 2) {
            for (int w = 2; w <= 256; w *= 2) {
                auto key = std::make_pair(h, w);
                tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
            }
        }
        double delta = double(tmin - min) / double(min) * 100.0;
        max_64 = std::max(max_64, delta);
        avg_64 += delta;

        // B_8:
        tmin = 0;
        for (int h = 2; h <= 256; h *= 2) {
            int w = h;
            auto key = std::make_pair(h, w);
            tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
        }
        delta = double(tmin - min) / double(min) * 100.0;
        max_8 = std::max(max_8, delta);
        avg_8 += delta;
    }

    avg_64 /= double(n);
    std::cout << "B_64: Average = " << green << avg_64 << reset
        << ", Maximum = " << red << max_64 << reset << std::endl;

    avg_8 /= double(n);
    std::cout << "B_8: Average = " << green << avg_8 << reset
        << ", Maximum = " << red << max_8 << reset << std::endl;
}
