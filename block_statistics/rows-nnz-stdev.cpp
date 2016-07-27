#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <abhsf/utils/colors.h>

double stdev(const std::string& filename, uintmax_t m) 
{
    std::vector<uintmax_t> rows_nnz(m);
    std::ifstream f(filename);
    for (uintmax_t k = 0; k < m; k++)
        f>> rows_nnz[k];
    f.close();

    // average
    uintmax_t nnz = std::accumulate(rows_nnz.cbegin(), rows_nnz.cend(), uintmax_t(0));
    double avg = double(nnz) / double(m);

    double stdev = 0.0;
    for (uintmax_t k = 0; k < m; k++)
        stdev += pow(double(rows_nnz[k]) - avg, 2.0);
    stdev /= double(m);
    stdev = sqrt(stdev);
    
    return stdev;
}

int main()
{
    std::ifstream f("props");
    uintmax_t m, n;
    f >> m >> n;
    f.close();
    
    double stdev_stored = stdev("rows_nnz_stored", m);
    double stdev_all    = stdev("rows_nnz_all",    m);

    std::cout << stdev_stored << " " << stdev_all << " " 
        << stdev_stored / double(n) * 100.0 << " " 
        << stdev_all / double(n) * 100.0 << std::endl;
}
