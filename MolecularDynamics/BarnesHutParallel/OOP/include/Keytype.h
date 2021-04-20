#ifndef OOP_KEYTYPE_H
#define OOP_KEYTYPE_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>
#include <boost/mpi.hpp>

class KeyType;

typedef unsigned long keyInteger;
//typedef std::vector<KeyType> KeyList;

class KeyType {

public:
    keyInteger key;
    int maxLevel;

    KeyType();
    KeyType(keyInteger key_);
    template<typename I>KeyType(I key);

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

template<typename I>KeyType::KeyType(I key) {
    this->key = (keyInteger)key;
}

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
