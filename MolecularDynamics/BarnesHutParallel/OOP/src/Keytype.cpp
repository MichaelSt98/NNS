#include "../include/Keytype.h"


KeyType::KeyType() {
    getTypeSizes();
    keyStandard = 0UL;
    extended = false;
}

void KeyType::getTypeSizes() {
    sizeOfStandardType = sizeof(standardType)*CHAR_BIT;
    sizeOfExtensionType = sizeof(extensionType)*CHAR_BIT;
    standardOverhead = sizeOfStandardType % 3;
    extensionOverhead = sizeOfExtensionType % 3;
    maxStandard = std::numeric_limits<standardType>::max() >> standardOverhead; // - (standardOverhead << (sizeOfStandardType-1));
    std::cout << "maxStandard: " << maxStandard << std::endl;

    maxExtension = std::numeric_limits<extensionType>::max() >> extensionOverhead; // - (extensionOverhead << (sizeOfExtensionType-1));
    std::cout << "maxExtension: " << maxExtension << std::endl;

}

int KeyType::getMaxLevel() const {
    if (extended) {
        int standardMaxLevel = (sizeOfStandardType-1)/3;
        std::cout << "keyExtension.size(): " << keyExtension.size() << std::endl;
        int extensionLevels = (keyExtension.size()*sizeOfExtensionType-1)/3;
        return standardMaxLevel + extensionLevels;
    }
    else {
        return (sizeOfStandardType-1)/3;
    }
}
