#ifndef OOP_KEYTYPE_H
#define OOP_KEYTYPE_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>


typedef unsigned long keyInteger;

class KeyType {

public:
    keyInteger key;
    int maxLevel;

    KeyType();
    KeyType(keyInteger key_);

    int getMaxLevel();

    friend std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

    friend KeyType operator<<(KeyType key2Shift, std::size_t n);
    friend KeyType operator>>(KeyType key2Shift, std::size_t n);
    friend KeyType operator|(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator&(KeyType lhsKey, KeyType rhsKey);
    friend KeyType operator+(KeyType lhsKey, KeyType rhsKey);
};

std::ostream &operator<<(std::ostream &os, const KeyType &key2print);

KeyType operator<<(KeyType key2Shift, std::size_t n);
KeyType operator>>(KeyType key2Shift, std::size_t n);
KeyType operator|(KeyType lhsKey, KeyType rhsKey);
KeyType operator&(KeyType lhsKey, KeyType rhsKey);
KeyType operator+(KeyType lhsKey, KeyType rhsKey);

#endif //OOP_KEYTYPE_H
