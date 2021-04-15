#include "../include/Keytype.h"

KeyType::KeyType() {
    key = (keyInteger)0;
    maxLevel = getMaxLevel();
}

KeyType::KeyType(keyInteger key_) : KeyType() {
    key = key_;
}

int KeyType::getMaxLevel() {
    return (int)(sizeof(keyInteger)*CHAR_BIT/3);
}

std::ostream &operator<<(std::ostream &os, const KeyType &key2print) {
    //std::cout << key2print.key << std::endl;
    int level[key2print.maxLevel];
    for (int i=0; i<key2print.maxLevel; i++) {
        level[i] = key2print.key >> (key2print.maxLevel*3 - 3*(i+1)) & (int)7;
    }
    for (int i=0; i<key2print.maxLevel; i++) {
        os << std::to_string(level[i]);
        os <<  "|";
    }
    return os;
}

KeyType operator<<(KeyType key2Shift, std::size_t n) {
    return KeyType(key2Shift.key << n);
}

KeyType operator>>(KeyType key2Shift, std::size_t n) {
    return KeyType(key2Shift.key >> n);
}

KeyType operator|(KeyType lhsKey, KeyType rhsKey) {
    return KeyType(lhsKey.key | rhsKey.key);
}

KeyType operator&(KeyType lhsKey, KeyType rhsKey) {
    return KeyType(lhsKey.key & rhsKey.key);
}

KeyType operator+(KeyType lhsKey, KeyType rhsKey) {
    return KeyType(lhsKey.key + rhsKey.key);
}

bool operator<(KeyType lhsKey, KeyType rhsKey) {
    return (lhsKey.key < rhsKey.key);
}

bool operator<=(KeyType lhsKey, KeyType rhsKey) {
    return (lhsKey.key <= rhsKey.key);
}

bool operator>(KeyType lhsKey, KeyType rhsKey) {
    return (lhsKey.key > rhsKey.key);
}

bool operator>=(KeyType lhsKey, KeyType rhsKey) {
    return (lhsKey.key >= rhsKey.key);
}