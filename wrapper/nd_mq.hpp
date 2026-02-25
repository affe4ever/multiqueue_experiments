#pragma once

#include "util.hpp"
#include "../tools/replay_tree.hpp"

#include <mutex>
#include <random>
#include <optional>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace wrapper::mq_pq {

template <bool Min, typename Key = unsigned long, typename T = Key>
class ND_MQ {
    public:
        using key_type    = Key;
        using mapped_type = T;
        using value_type  = std::pair<key_type, mapped_type>;

    private:
        struct KeyOfValue {
            static const value_type& get(const value_type& v) { return v; }
        };

        using comparator = std::conditional_t<Min,
                                              std::less<value_type>,
                                              std::greater<value_type>>;

        using pq_type = ReplayTree<value_type,
                                   value_type,
                                   KeyOfValue,
                                   comparator>;

        std::mutex lock_{};
        pq_type pq_{};

        double mean_{0.0};
        double stddev_{1.0};
        double percentile_{0.0};

        std::mt19937 gen_{std::random_device{}()};
        std::normal_distribution<double> dist_{0.0, 1.0};

    public:
        using handle_type   = util::SelfHandle<ND_MQ>;
        using settings_type = util::EmptySettings;

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
            const double phigh = 1 - plow;

            double q, r;

            if (p < plow) {
                q = std::sqrt(-2 * std::log(p));
                return (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                       ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
            }

            if (phigh < p) {
                q = std::sqrt(-2 * std::log(1 - p));
                return -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
                        ((((d1*q + d2)*q + d3)*q + d4)*q + 1);
            }

            q = p - 0.5;
            r = q * q;
            return (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
                   (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1);
        }

        ND_MQ(int /*unused*/,
              std::size_t /*initial_capacity*/,
              settings_type const& /*unused*/,
              std::optional<double> mean = std::nullopt,
              std::optional<double> stddev = std::nullopt,
              std::optional<double> percentile = std::nullopt)
        {
            bool m = mean.has_value();
            bool s = stddev.has_value();
            bool p = percentile.has_value();

            if ((m + s + p) != 2)
                throw std::invalid_argument("Exactly two of mean, stddev, percentile must be given");

            if (s && p) {
                if (*percentile <= 0.0 || *percentile >= 0.5)
                    throw std::invalid_argument("Percentile must be in (0,0.5)");
                if (*stddev <= 0.0)
                    throw std::invalid_argument("Stddev must be positive");

                double z = inverse_normal_cdf(*percentile);
                double mu = -z * (*stddev);
                if (mu < 0.0) mu = -mu;

                mean_ = mu;
                stddev_ = *stddev;
            }
            else if (m && p) {
                double z = inverse_normal_cdf(*percentile);
                double sigma = (0.0 - *mean) / z;
                if (sigma < 0.0) sigma = -sigma;

                mean_ = *mean;
                stddev_ = sigma;
            }
            else if (m && s) {
                mean_ = *mean;
                stddev_ = *stddev;
            }

            double z = (0.0 - mean_) / stddev_;
            percentile_ = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));

            dist_ = std::normal_distribution<double>(mean_, stddev_);

            std::clog << "ND_MQ settings\n";
            std::clog << "Mean: " << mean_ << '\n';
            std::clog << "Stddev: " << stddev_ << '\n';
            std::clog << "Percentile (index 0): " << percentile_ << "\n\n";
        }

        void push(value_type const& value) {
            std::scoped_lock guard(lock_);
            pq_.insert(value);
        }

        std::optional<value_type> try_pop() {
            std::scoped_lock guard(lock_);

            if (pq_.empty())
                return std::nullopt;

            int index = static_cast<int>(std::round(dist_(gen_)));
            index = std::clamp(index,
                               0,
                               static_cast<int>(pq_.size() - 1));

            auto it = pq_.begin();
            std::advance(it, index);

            value_type value = *it;

            auto [success, rank, delay] = pq_.erase_val(value);

            if (!success)
                throw std::runtime_error("ND_MQ: erase_val failed");

            return value;
        }

        bool empty() const {
            std::scoped_lock guard(lock_);
            return pq_.empty();
        }

        std::size_t size() const {
            std::scoped_lock guard(lock_);
            return pq_.size();
        }

        static void write_human_readable(std::ostream& out) {
            out << "ND_MQ (ReplayTree-based Gaussian MQ)\n";
        }

        handle_type get_handle() {
            return handle_type{*this};
        }
    };

} // namespace wrapper::nd_mq