#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
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

    std::set<std::pair<uintmax_t, uintmax_t>> B_64, B_20, B_14, B_8;

    for (int h = 2; h <= 256; h *= 2) 
        B_8.emplace(h, h);
    assert(B_8.size() == 8);

    B_14 = B_8;
    for (int h = 4; h <= 16; h *= 2) 
        for (int w = 4; w <= 16; w *= 2) 
            B_14.emplace(h, w);
    assert(B_14.size() == 14);

    B_20 = B_8;
    for (int h = 4; h <= 32; h *= 2) 
        for (int w = 4; w <= 32; w *= 2) 
            B_20.emplace(h, w);
    assert(B_20.size() == 20);

    for (int h = 2; h <= 256; h *= 2) 
        for (int w = 2; w <= 256; w *= 2) 
            B_64.emplace(h, w);
    assert(B_64.size() == 64);

    double avg_64 = 0.0;
    double max_64 = 0.0;
    double avg_20 = 0.0;
    double max_20 = 0.0;
    double avg_14 = 0.0;
    double max_14 = 0.0;
    double avg_8 = 0.0;
    double max_8 = 0.0;

    for (int k = 0; k < n; k++) {
        auto min = stats[k].min;

        // B_64:
        uintmax_t tmin = 0;
        for (auto key : B_64) 
            tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
        double delta = double(tmin - min) / double(min) * 100.0;
        max_64 = std::max(max_64, delta);
        avg_64 += delta;

        // B_20:
        tmin = 0;
        for (auto key : B_20) 
            tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
        delta = double(tmin - min) / double(min) * 100.0;
        max_20 = std::max(max_20, delta);
        avg_20 += delta;

        // B_14:
        tmin = 0;
        for (auto key : B_14) 
            tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
        delta = double(tmin - min) / double(min) * 100.0;
        max_14 = std::max(max_14, delta);
        avg_14 += delta;

        // B_8:
        tmin = 0;
        for (auto key : B_8) 
            tmin = (tmin == 0) ?  tmin = stats[k].map[key] : tmin = std::min(tmin, stats[k].map[key]);
        delta = double(tmin - min) / double(min) * 100.0;
        max_8 = std::max(max_8, delta);
        avg_8 += delta;
    }

    avg_64 /= double(n);
    std::cout << "B_64: Average = " << green << avg_64 << reset
        << ", Maximum = " << red << max_64 << reset << std::endl;

    avg_20 /= double(n);
    std::cout << "B_20: Average = " << green << avg_20 << reset
        << ", Maximum = " << red << max_20 << reset << std::endl;

    avg_14 /= double(n);
    std::cout << "B_14: Average = " << green << avg_14 << reset
        << ", Maximum = " << red << max_14 << reset << std::endl;

    avg_8 /= double(n);
    std::cout << "B_8: Average = " << green << avg_8 << reset
        << ", Maximum = " << red << max_8 << reset << std::endl;
}
