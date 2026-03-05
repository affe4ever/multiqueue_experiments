#pragma once
namespace ranked_btree { template<typename,typename,typename> class tree; }

#define TLX_BTREE_FRIENDS \
    friend class btree_friend; \
    template<typename,typename,typename> friend class ::ranked_btree::tree

#include "tlx_btree.hpp"
#include <cstddef>
#include <iterator>
#include <stdexcept>

namespace tlx {

template <typename Key>
struct set_key_of_value {
    static const Key& get(const Key& v) { return v; }
};

template <typename Key>
struct btree_default_set_traits : public btree_default_traits<Key, Key> {};

template <
    typename Key,
    typename Compare   = std::less<Key>,
    typename Traits    = btree_default_set_traits<Key>,
    typename Allocator = std::allocator<Key>>
class btree_set
    : public BTree<Key, Key, set_key_of_value<Key>, Compare, Traits, false, Allocator>
{
    using base = BTree<Key, Key, set_key_of_value<Key>, Compare, Traits, false, Allocator>;
public:
    explicit btree_set(const Allocator& a = Allocator()) : base(a) {}
    explicit btree_set(const Compare& c, const Allocator& a = Allocator()) : base(c, a) {}
};

struct btree_friend {
    template <typename BT>
    static typename BT::iterator select(BT& bt, std::size_t k) {
        auto it = bt.begin();
        std::advance(it, static_cast<std::ptrdiff_t>(k));
        return it;
    }
    template <typename BT>
    static typename BT::const_iterator cselect(const BT& bt, std::size_t k) {
        auto it = bt.begin();
        std::advance(it, static_cast<std::ptrdiff_t>(k));
        return it;
    }
};

}

namespace ranked_btree {

template <
    typename Key,
    typename Compare = std::less<Key>,
    typename Traits  = tlx::btree_default_set_traits<Key>>
class tree : public tlx::btree_set<Key, Compare, Traits>
{
    using base    = tlx::btree_set<Key, Compare, Traits>;
    using btree_t = typename base::Self;

    btree_t&       bt()       { return static_cast<btree_t&>(*this); }
    const btree_t& bt() const { return static_cast<const btree_t&>(*this); }

public:
    using typename base::iterator;
    using typename base::const_iterator;

    iterator select_by_rank(std::size_t k) {
        if (k >= this->size())
            throw std::out_of_range("ranked_btree::select_by_rank: out of range");
        return tlx::btree_friend::select(bt(), k);
    }

    const_iterator select_by_rank(std::size_t k) const {
        if (k >= this->size())
            throw std::out_of_range("ranked_btree::select_by_rank: out of range");
        return tlx::btree_friend::cselect(bt(), k);
    }

    void erase_by_rank(std::size_t k) {
        if (k >= this->size())
            throw std::out_of_range("ranked_btree::erase_by_rank: out of range");
        this->erase(tlx::btree_friend::select(bt(), k));
    }
};

}