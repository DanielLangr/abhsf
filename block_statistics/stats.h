#ifndef ABHSF_STATS_H
#define ABHSF_STATS_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <utility>

#include <abhsf/utils/matrix_properties.h>

class matrix_block_stats
{
    public:
        using block_stats_t = std::map<uint32_t, uint32_t>; // block nnz -> number of blocks
        using stats_t = std::map<std::pair<uint32_t, uint32_t>, block_stats_t>; // block size -> block statistics

        void read()
        {
            stats_.clear();

            // read matrix properties
            std::ifstream f("props");
            int type, symmetry;
            uintmax_t nnz_mtx, nnz_stored, nnz_all;
            f >> props_.m >> props_.n >> nnz_mtx >> nnz_stored >> nnz_all >> type >> symmetry;
            props_.nnz = nnz_stored; // ! stored nnz taken
            props_.type = static_cast<matrix_type_t>(type);
            props_.symmetry = static_cast<matrix_symmetry_t>(symmetry);
            f.close();

            // read block statistics
            for (int k = 1; k <= 8; k++) {
                for (int l = 1; l <= 8; l++) {
                    const uint32_t r = 1 << k; // block height
                    const uint32_t s = 1 << l; // block width
                    auto block_size = std::make_pair(r, s);

                    stats_.emplace(block_size, block_stats_t());

                    std::ifstream f(std::to_string(r) + "x" + std::to_string(s) + ".bstats");
                    while (true) {
                        uint32_t nnz, count;
                        f >> nnz >> count;
                        if (!f) 
                            break;
                        stats_[block_size].emplace(nnz, count);
                    }
                    f.close();
                }
            }
        }

        const matrix_properties& props() const { return props_; }
        const stats_t& stats() const { return stats_; }

        bool check()
        {
            for (const auto& stat : stats_) {
                uintmax_t nnz = 0;
                for (const auto& entry : stat.second) 
                    nnz += entry.first * entry.second;
                if (nnz != props_.nnz)
                    return false;
            }
            return true;
        }

    private:
        matrix_properties props_;
        stats_t stats_;
};

#endif
