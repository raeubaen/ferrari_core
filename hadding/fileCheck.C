#include <TFile.h>
#include <TKey.h>
#include <TObject.h>
#include <iostream>

void fileCheck(const char* filename)
{
    TFile* f = TFile::Open(filename);

    if (!f || f->IsZombie()) {
        std::cout << "FILE_ZOMBIE" << std::endl;
        return;
    }

    TIter next(f->GetListOfKeys());
    TKey* key = nullptr;

    while ((key = (TKey*)next())) {

        TObject* obj = key->ReadObj();

        //std::cout << "read: " << key->GetName() << " OK!" << std::endl;
        if (!obj) {
            std::cout << "BROKEN_OBJECT " << key->GetName() << std::endl;
            f->Close();
            return;
        }

        delete obj;
    }

    std::cout << "FILE_OK" << std::endl;

    f->Close();
}
