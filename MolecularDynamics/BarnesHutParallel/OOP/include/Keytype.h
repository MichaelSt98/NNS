#ifndef OOP_KEYTYPE_H
#define OOP_KEYTYPE_H

#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

//always use unsigned
typedef unsigned int standardType;
typedef unsigned int extensionType;

class KeyType {

public:
    int sizeOfStandardType;
    int sizeOfExtensionType;
    int standardOverhead;
    int extensionOverhead;
    standardType maxStandard;
    extensionType maxExtension;
    void getTypeSizes();

    bool extended;

    standardType keyStandard;
    std::vector<extensionType> keyExtension;

public:
    KeyType();
    template<typename I>KeyType(I key);

    int getMaxLevel() const;

    friend std::ostream &operator << (std::ostream &os, const KeyType &key);
};

template<typename I>KeyType::KeyType(I key) {
    getTypeSizes();
    if (sizeof(key)*CHAR_BIT <= sizeOfStandardType) {
        keyStandard = (unsigned long)key & maxStandard;
    }
    else {
        keyStandard = key & maxStandard;
        key = key >> (sizeOfStandardType - standardOverhead);
        if (key > 0) {
            extended = true;
        }
        while(key > 0) {
            std::cout << "Inserting: " << (extensionType)(key & maxExtension) << std::endl;
            keyExtension.push_back((extensionType)(key & maxExtension));
            key = key >> (sizeOfExtensionType - extensionOverhead);
            std::cout << "Key: " << key << std::endl;
        }
    }
}

inline std::ostream &operator<<(std::ostream &os, const KeyType &key)
{
    int level = key.getMaxLevel();
    int levels[level];

    int keyPart = 0;
    int keyExtendedIndex = 0;
    int j = 0;

    for (int i = 0; i<level; i++) {
        if (i < (key.sizeOfStandardType-key.standardOverhead)/3) {
            std::cout << "i = " << i << std::endl;
            keyPart = key.keyStandard >> (key.sizeOfStandardType - 3*(i + 1) - key.standardOverhead) & (int)7;
            //levels[i] = keyPart;
        }
        else {
            if (key.extended) {
                if (j > (key.sizeOfExtensionType-key.extensionOverhead)) {
                    j = 0;
                    keyExtendedIndex++;
                }
                keyPart = key.keyExtension[keyExtendedIndex]
                                  >> (key.sizeOfExtensionType - 3 * (j + 1) - key.extensionOverhead) & (int)7;
                j++;
                //levels[i] = keyPart;
            }
        }
        levels[i] = keyPart;
        keyPart = -1;
    }

    for (int i = 0; i<level; i++) {
        os << std::to_string(levels[i]);
        os <<  "|";
    }
    return os;
}

#endif //OOP_KEYTYPE_H
