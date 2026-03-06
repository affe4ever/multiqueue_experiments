#pragma once

#include "util.hpp"
#include "../tools/ranked_btree.hpp"

#include <mutex>
#include <random>
#include <optional>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <vector>

#ifdef DR_PQ_USE_BUFFERS
  #ifndef DR_PQ_BUFFER_SIZE
    #define DR_PQ_BUFFER_SIZE 64
  #endif
#endif

namespace wrapper::dr_pq {

template <bool Min, typename Key = unsigned long, typename T = Key>
class DR_PQ {
    public:
        using key_type    = Key;
        using mapped_type = T;
        using value_type  = std::pair<key_type, mapped_type>;

    private:
        using comparator = std::conditional_t<Min,
                                              std::less<value_type>,
                                              std::greater<value_type>>;

        using pq_type = ranked_btree::tree<value_type, comparator>;

        struct Guard {
            std::mutex* m_;
            explicit Guard(std::mutex* m) : m_(m) { if (m_) m_->lock(); }
            ~Guard()                               { if (m_) m_->unlock(); }

            Guard(const Guard&)            = delete;
            Guard& operator=(const Guard&) = delete;
        };

        const bool     threaded_;
        mutable std::mutex lock_;

        Guard make_guard() const { return Guard(threaded_ ? &lock_ : nullptr); }

        pq_type pq_;

        double mean_{0.0};
        double stddev_{1.0};
        double percentile_{0.0};

        std::mt19937                     gen_{std::random_device{}()};
        mutable std::normal_distribution<double> dist_;

#ifdef DR_PQ_USE_BUFFERS
        void flush(std::vector<value_type>& buf) {
            for (auto& v : buf)
                pq_.insert(v);
            buf.clear();
        }
#endif

    public:
        using settings_type = util::EmptySettings;

        class handle_type {
            friend DR_PQ;
            DR_PQ* pq_;

#ifdef DR_PQ_USE_BUFFERS
            std::vector<value_type>          buf_;
            std::mt19937                     gen_;
            std::normal_distribution<double> dist_;

            explicit handle_type(DR_PQ& pq)
                : pq_(&pq)
                , gen_(std::random_device{}())
                , dist_(pq.dist_)
            {}
#else
            explicit handle_type(DR_PQ& pq) : pq_(&pq) {}
#endif

           public:
            bool push(value_type const& value) {
#ifdef DR_PQ_USE_BUFFERS
                buf_.push_back(value);
                if (buf_.size() >= DR_PQ_BUFFER_SIZE) {
                    auto guard = pq_->make_guard();
                    pq_->flush(buf_);
                }
#else
                pq_->push(value);
#endif
                return true;
            }

            std::optional<value_type> try_pop() {
#ifdef DR_PQ_USE_BUFFERS
                return pq_->try_pop_buffered(buf_, gen_, dist_);
#else
                return pq_->try_pop();
#endif
            }

#ifdef DR_PQ_USE_BUFFERS
            ~handle_type() {
                if (!buf_.empty()) {
                    auto guard = pq_->make_guard();
                    pq_->flush(buf_);
                }
            }
#endif
        };

        static double inverse_normal_cdf(double p) {
            if (p <= 0.0 || p >= 1.0)
                throw std::invalid_argument("Percentile must be in (0,1)");

            static const double a1 = -3.969683028665376e+01;
            static const double a2 =  2.209460984245205e+02;
            static const double a3 = -2.759285104469687e+02;
            static const double a4 =  1.383577518672690e+02;
            static const double a5 = -3.066479806614716e+01;
            static const double a6 =  2.506628277459239e+00;

            static const double b1 = -5.447609879822406e+01;
            static const double b2 =  1.615858368580409e+02;
            static const double b3 = -1.556989798598866e+02;
            static const double b4 =  6.680131188771972e+01;
            static const double b5 = -1.328068155288572e+01;

            static const double c1 = -7.784894002430293e-03;
            static const double c2 = -3.223964580411365e-01;
            static const double c3 = -2.400758277161838e+00;
            static const double c4 = -2.549732539343734e+00;
            static const double c5 =  4.374664141464968e+00;
            static const double c6 =  2.938163982698783e+00;

            static const double d1 =  7.784695709041462e-03;
            static const double d2 =  3.224671290700398e-01;
            static const double d3 =  2.445134137142996e+00;
            static const double d4 =  3.754408661907416e+00;

            const double plow  = 0.02425;
            const double phigh = 1.0 - plow;

            double q, r;
            if (p < plow) {
                q = std::sqrt(-2.0 * std::log(p));
                return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                       ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
            }
            if (phigh < p) {
                q = std::sqrt(-2.0 * std::log(1.0 - p));
                return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                         ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
            }
            q = p - 0.5;
            r = q * q;
            return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
                   (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
        }

        DR_PQ(int num_threads,
              std::size_t /*initial_capacity*/,
              settings_type const& /*unused*/,
              std::optional<double> mean       = std::nullopt,
              std::optional<double> stddev     = std::nullopt,
              std::optional<double> percentile = std::nullopt)
            : threaded_(num_threads != 0)
        {
            bool m = mean.has_value();
            bool s = stddev.has_value();
            bool p = percentile.has_value();

            if ((m + s + p) != 2)
                throw std::invalid_argument(
                    "Exactly two of mean, stddev, percentile must be given");

            double computed_mean   = 0.0;
            double computed_stddev = 1.0;

            if (s && p) {
                if (*percentile <= 0.0 || *percentile >= 0.5)
                    throw std::invalid_argument("Percentile must be in (0, 0.5)");
                if (*stddev <= 0.0)
                    throw std::invalid_argument("Stddev must be positive");
                double z        = inverse_normal_cdf(*percentile);
                computed_mean   = std::abs(-z * (*stddev));
                computed_stddev = *stddev;
            }
            else if (m && p) {
                double z        = inverse_normal_cdf(*percentile);
                computed_mean   = *mean;
                computed_stddev = std::abs((0.0 - *mean) / z);
            }
            else {
                computed_mean   = *mean;
                computed_stddev = *stddev;
            }

            mean_   = computed_mean;
            stddev_ = computed_stddev;

            double z    = (0.0 - mean_) / stddev_;
            percentile_ = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));

            dist_ = std::normal_distribution<double>(mean_, stddev_);

            std::clog << "DR_PQ settings\n"
                      << "  Mean:                 " << mean_       << '\n'
                      << "  Stddev:               " << stddev_     << '\n'
                      << "  Percentile (index 0): " << percentile_ << "\n\n";
        }

        void push(value_type const& value) {
            auto guard = make_guard();
            pq_.insert(value);
        }

        std::optional<value_type> try_pop() {
            auto guard = make_guard();
            if (pq_.empty()) return std::nullopt;
            int index = static_cast<int>(std::round(dist_(gen_)));
            index = std::clamp(index, 0, static_cast<int>(pq_.size() - 1));
            value_type value = *pq_.select_by_rank(static_cast<std::size_t>(index));
            pq_.erase_by_rank(static_cast<std::size_t>(index));
            return value;
        }

#ifdef DR_PQ_USE_BUFFERS
        std::optional<value_type> try_pop_buffered(std::vector<value_type>& buf,
                                                   std::mt19937& gen,
                                                   std::normal_distribution<double>& dist) {
            auto guard = make_guard();
            flush(buf);
            if (pq_.empty()) return std::nullopt;
            int index = static_cast<int>(std::round(dist(gen)));
            index = std::clamp(index, 0, static_cast<int>(pq_.size() - 1));
            value_type value = *pq_.select_by_rank(static_cast<std::size_t>(index));
            pq_.erase_by_rank(static_cast<std::size_t>(index));
            return value;
        }
#endif

        bool empty() const {
            auto guard = make_guard();
            return pq_.empty();
        }

        std::size_t size() const {
            auto guard = make_guard();
            return pq_.size();
        }

        static void write_human_readable(std::ostream& out) {
#ifdef DR_PQ_USE_BUFFERS
            out << "DR_PQ (ranked_btree, buffered, flush=" << DR_PQ_BUFFER_SIZE << ")\n";
#else
            out << "DR_PQ (ranked_btree, mutex)\n";
#endif
        }

        handle_type get_handle() { return handle_type{*this}; }
};

} // namespace wrapper::dr_pq