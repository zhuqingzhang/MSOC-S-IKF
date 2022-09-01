//
// Created by zzq on 2020/10/11.
//

#ifndef SRC_MATCHBRIEF_H
#define SRC_MATCHBRIEF_H
#include "match/MatchBase.h"

namespace ov_core{

    class BriefExtractor
    {
      public:
      virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const;
      BriefExtractor(const std::string &pattern_file);
      BriefExtractor(){};
      DVision::BRIEF m_brief;
    };



    class MatchBrief: public MatchBase {

    public:
        MatchBrief(MatchBaseOptions &options): MatchBase(options)
        {
          loadVocabulary(_options.voc_file);

          loadPriorMap(_options.map_file);

          extractor = BriefExtractor(_options.brief_pattern_filename);
        }


    void loadVocabulary(string voc_file) override;

    void loadPriorMap(string map_file) override;

    void ExtractFeatureAndDescriptor(Keyframe& kf) override;

    bool DetectLoop(Keyframe& kf) override;

    void MatchingWithLoop(Keyframe& kf) override;



    private:
    
    BriefDatabase _db;

    BriefVocabulary* voc;

    BriefExtractor extractor;

    };
}



#endif //SRC_MATCHBRIEF_H
