#ifndef ABHSF_STATS_H
#define ABHSF_STATS_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>

#include <abhsf/utils/matrix_properties.h>

class matrix_block_stats
{
    public:
        using block_stats_t = std::map<uint32_t, uint32_t>; // block nnz -> number of blocks
        using stats_t = std::map<uint32_t, block_stats_t>; // block size -> block statistics

        void read()
        {
            stats_.clear();

            // read matrix properties
            std::ifstream f("props");
            int type, symmetry;
            f >> props_.m >> props_.n >> props_.nnz >> type >> symmetry;
            props_.type = static_cast<matrix_type_t>(type);
            props_.symmetry = static_cast<matrix_symmetry_t>(symmetry);
            f.close();

            // read block statistics
            for (int k = 1; k <= 10; k++) {
                const uint32_t s = 1 << k; // block size
                stats_.emplace(s, block_stats_t());

                std::ifstream f(std::to_string(s) + ".bstats");
                while (true) {
                    uint32_t nnz, count;
                    f >> nnz >> count;
                    if (!f) 
                        break;
                    stats_[s].emplace(nnz, count);
                }
                f.close();
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
