#ifndef ABHSF_MMF_H
#define ABHSF_MMF_H

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

bool is_directory(const std::string& path)
{
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0)
        return 0;
    return S_ISDIR(statbuf.st_mode);
}

class mmf_t
{
    public:
        using bs_record_t = std::array<uintmax_t, 8>;
        using block_size_t = std::pair<uintmax_t, uintmax_t>;
        using matrix_entry_t = std::map<block_size_t, bs_record_t>;
        using map_t = std::map<std::string, matrix_entry_t>;

        enum class precision_t
        {
            SINGLE,
            DOUBLE
        };

        void load(precision_t precision)
        {
            DIR* dir;
            if ((dir = opendir(".")) == nullptr)
                throw std::runtime_error("Error running opendir().");

            struct dirent* ent;
            while ((ent = readdir(dir)) != nullptr) {
                std::string path(ent->d_name);
                if (is_directory(path) && path != "." && path != "..") {
                    // process matrix data
                    matrix_entry_t matrix_entry;

                    std::ifstream f;

                    if (precision == precision_t::SINGLE)
                        f.open(path + "/mmf-single");
                    else
                        f.open(path + "/mmf-double");

                    for (int k = 0; k < 64; k++) {
                        uintmax_t h, w, coo32, csr32, coo, csr, bitmap, dense, adaptive, adaptive134;
                        f >> h >> w >> coo32 >> csr32 >> coo >> csr >> bitmap >> dense >> adaptive >> adaptive134;

                        block_size_t block_size = std::make_pair(h, w);
                        bs_record_t bs_record = { coo32, csr32, coo, csr, bitmap, dense, adaptive, adaptive134 };
                        matrix_entry[block_size] = bs_record;
                    }
                    map_[path] = matrix_entry;

                    f.close();
                }
            }
        }

        const map_t& map() const { return map_; }

        uintmax_t block_opt(const std::string& matrix,
                block_size_t* block_size = nullptr, uintmax_t* index = nullptr) const 
        {
            auto iter = map_.find(matrix);
            if (iter == map_.end())
                std::runtime_error("Requered matrix does not exists in the database!");
            const auto& matrix_entry = iter->second;

            uintmax_t opt = std::numeric_limits<uintmax_t>::max();

            for (auto const& bs_records : matrix_entry) {
                for (int k = 2; k <= 6; k++) {
                    auto actual = (bs_records.second)[k];
                    if (actual < opt) {
                        opt = actual;

                        if (block_size)
                            *block_size = bs_records.first;
                        if (index)
                            *index = k;
                    }
                }
            }

            return opt;
        }

    private:
        map_t map_;

};

#endif
