#ifndef UTILS_TIMER_H
#define UTILS_TIMER_H

#include <chrono>
#include <cstdint>

#include <sys/time.h>

template <typename Clock = std::chrono::high_resolution_clock>
class chrono_timer
{
    public:
        using clock_type = Clock;
        using time_point_type = std::chrono::time_point<Clock>;

        enum start_options { start_now, do_not_start_now };

        explicit chrono_timer(start_options should_start = do_not_start_now) 
        {
            if (should_start == start_now)
                start();
        }

        void start()
        {
            start_ = clock_type::now();
        }

        uintmax_t stop()
        {
            auto diff = clock_type::now() - start_;
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(diff);
            nsec_ = duration.count();
            return nsec_;
        }

        uintmax_t nanoseconds() const { return nsec_; }

        double seconds() const { return double(nsec_) / 1.0e9; }
        double miliseconds() const { return double(nsec_) / 1.0e6; }
        double microseconds() const { return double(nsec_) / 1.0e3; }

    private:
        time_point_type start_;
        uintmax_t nsec_;
};

class gettimeofday_timer
{
    public:
        enum start_options { start_now, do_not_start_now };

        explicit gettimeofday_timer(start_options should_start = do_not_start_now)
        {
            if (should_start == start_now)
                start();
        }

        void start()
        {
            gettimeofday(&t0_, NULL);
        }

        uintmax_t stop()
        {
            gettimeofday(&t1_, NULL);
            usec_ = (t1_.tv_sec - t0_.tv_sec) * 1000000UL + (t1_.tv_usec - t0_.tv_usec);
            return usec_ * 1000; // returns nanoseconds
        }

        uintmax_t nanoseconds() const { return usec_ * 1000; }

        double seconds() const { return double(usec_) / 1.0e6; }
        double miliseconds() const { return double(usec_) / 1.0e3; }
        double microseconds() const { return double(usec_); }

    private:
        timeval t0_, t1_;
        uintmax_t usec_;
};
#endif
