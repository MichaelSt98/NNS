#ifndef OOP_KEYTYPE_H
#define OOP_KEYTYPE_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <boost/mpi.hpp>

typedef unsigned long keyInteger;

class KeyType {

public:
    keyInteger key;
    int maxLevel;

    KeyType();
    KeyType(keyInteger key_);

    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version) {
        ar & key;
        ar & maxLevel;
    }

    int getMaxLevel();

    friend std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

    friend KeyType operator<<(KeyType key2Shift, std::size_t n);
    friend KeyType operator>>(KeyType key2Shift, std::size_t n);
    friend KeyType operator|(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator&(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator+(KeyType lhsKey, KeyType rhsKey);
    friend bool operator<(KeyType lhsKey, KeyType rhsKey);
    friend bool operator<=(KeyType lhsKey, KeyType rhsKey);
    friend bool operator>(KeyType lhsKey, KeyType rhsKey);
    friend bool operator>=(KeyType lhsKey, KeyType rhsKey);
};

std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

KeyType operator<<(KeyType key2Shift, std::size_t n);
KeyType operator>>(KeyType key2Shift, std::size_t n);
KeyType operator|(KeyType lhsKey, KeyType rhsKey);
KeyType operator&(KeyType lhsKey, KeyType rhsKey);
KeyType operator+(KeyType lhsKey, KeyType rhsKey);
bool operator<(KeyType lhsKey, KeyType rhsKey);
bool operator<=(KeyType lhsKey, KeyType rhsKey);
bool operator>(KeyType lhsKey, KeyType rhsKey);
bool operator>=(KeyType lhsKey, KeyType rhsKey);

#endif //OOP_KEYTYPE_H
