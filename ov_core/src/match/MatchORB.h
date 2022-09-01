//
// Created by zzq on 2021/9/11.
//

#ifndef SRC_MATCHORB_H
#define SRC_MATCHORB_H
#include "match/MatchBase.h"

namespace ov_core{

    class MatchORB: public MatchBase {

    public:
        MatchORB(MatchBaseOptions &options): MatchBase(options)
        {}

    private:

    void loadVocabulary(string voc_path) override{}

    void loadPriorMap(string map_path) override{}

    void ExtractFeatureAndDescriptor(Keyframe& kf) override{};

    bool DetectLoop(Keyframe& kf) override{};

    void MatchingWithLoop(Keyframe& kf) override{};


    ORBDatabase* _db;

    };
}



#endif //SRC_MATCHORB_H
