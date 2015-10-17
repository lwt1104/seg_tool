#include <pcl/point_types.h>
#include <pcl/features/vfh.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/filter.h>
#include <map>
// #include "column_match_lib.cpp"
#include "flann_template_match_helper.cpp"
#include <fstream>

typedef pcl::PointXYZRGB PointType;
typedef pcl::Normal NormalType;
typedef std::map<int, std::vector<pcl::PointCloud<PointType>::Ptr> > Map;

double pos_prior;
std::string output_dir;
ofstream outfile;

float get_depth(const pcl::PointCloud<PointType>::Ptr& cloud) {
  float zsum = 0;
  int num = 0;
  for (size_t i = 0; i < cloud->points.size(); i++) {
    PointType &p = cloud->points[i];
    if (!pcl::isFinite(p)) {
      continue;
    }
    zsum += p.z;
    num++;
  }
  float distance = zsum / num;
  return distance;
}

int get_distance_key(float d, int scale) {
  float step = (float)scale / 100.0;
  int level = d / step;
  return level;
}

std::string get_folder_name(float d, int scale) 
{
  return "s";
};


int loadToVector(const boost::filesystem::path &base_dir, int step, std::map<int, std::vector<pcl::PointCloud<PointType>::Ptr> >& samples) {
 if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir)) {
    PCL_ERROR ("Couldn't find the sub directory %s.\n", base_dir.string().c_str());
    return -1;
  }
  for (boost::filesystem::directory_iterator it(base_dir); it != boost::filesystem::directory_iterator (); ++it) {
    if (boost::filesystem::is_directory(it->status ()))
    {
      pcl::console::print_highlight ("Visit %s.\n", it->path().filename().string().c_str());
      loadToVector(it->path (), step, samples);
    }
    if (boost::filesystem::is_regular_file (it->status ()) && boost::filesystem::extension (it->path ()) == ".pcd")
    {
      // std::cout << it->path().string() << std::endl;
      pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
      if (pcl::io::loadPCDFile<PointType> (it->path().string(), *cloud) == -1) {
        PCL_ERROR ("Couldn't read file %s.pcd \n", it->path().string().c_str());
        continue;
      }
      float depth = get_depth(cloud);
      int key = get_distance_key(depth, step);
      if (key < 3) {
        continue;
      }
      std::map<int, std::vector<pcl::PointCloud<PointType>::Ptr> >::iterator it = samples.find(key);
      if (it == samples.end()) {
        std::vector<pcl::PointCloud<PointType>::Ptr> temp;
        samples[key] =  temp; 
      }
      samples[key].push_back(cloud);
    }

  }

  return 0;
}

int loadData(const boost::filesystem::path &base_dir, int step, int positive, std::map<int, std::vector<pcl::PointCloud<PointType>::Ptr> >& samples) {
 if (!boost::filesystem::exists (base_dir) && !boost::filesystem::is_directory (base_dir)) {
    PCL_ERROR ("Couldn't find the directory %s.\n", base_dir.string().c_str());
    return -1;
  }
  
  if (positive > 0) {
    loadToVector(base_dir.string() + "/" + "positive", step, samples);
  } else {
    loadToVector(base_dir.string() + "/" + "negative", step, samples);
  }
  return 0;
}

int findFirstLarger(std::vector<double> arr, double needle) {
  int l = 0, r = arr.size() - 1;
  int m;
  while (l <= r) {
    m = (l + r) / 2;
    if (arr[m] == needle) {
      return m;
    }
    if (arr[m] < needle) {
      l = m + 1;
    } else {
      r = m - 1;
    }
  }
  return l;

  // 1 3 5 9 12
}

int getLeq(std::vector<double> arr, double thresh) {
  int i = findFirstLarger(arr, thresh);
  if (i == arr.size()) {
    return arr.size();
  }
  if (arr[i] == thresh) {
    return i + 1;
  } else {
    return i;
  }
}
double getError(std::vector<double> pos_scores, std::vector<double> neg_scores, double thresh, int silent = 1) {
   int pos_c_num = getLeq(pos_scores, thresh);
   int neg_e_num = getLeq(neg_scores, thresh);
   double pos_e = 1 - (double) pos_c_num / pos_scores.size();
   double neg_e = (double) neg_e_num / neg_scores.size();

   double pos_ratio = (pos_prior > 0) ? pos_prior : (double) pos_scores.size() / (pos_scores.size() + neg_scores.size());
   double neg_ratio = 1 - pos_ratio; 
   double error_rate = pos_ratio * pos_e + neg_ratio * neg_e;

   if (!silent) {
     std::cout << "threshold: " << thresh;
     std::cout << "\tPos ratio: " << pos_ratio <<" Pos error: " << pos_e;
     std::cout << "\tNeg ratio: " << neg_ratio << " Neg error: " << neg_e;
     std::cout << "\tTotal: "<< error_rate << "\n";
   }
   return error_rate;
}

void getDetectionFalsePos(std::vector<double> pos_scores, std::vector<double> neg_scores, double thresh, 
  double& pos_c, double& neg_e, double& error, double& precision, double& neg_predict) {
   int pos_c_num = getLeq(pos_scores, thresh);
   int neg_e_num = getLeq(neg_scores, thresh);
   double pos_e = 1 - (double) pos_c_num / pos_scores.size();
   pos_c = 1 - pos_e;
   neg_e = (double) neg_e_num / neg_scores.size();
   double pos_ratio = (pos_prior > 0) ? pos_prior : (double) pos_scores.size() / (pos_scores.size() + neg_scores.size());
   double neg_ratio = 1 - pos_ratio; 
   error = pos_ratio * pos_e + neg_ratio * neg_e;

   precision = (pos_ratio * pos_c) / ((pos_ratio * pos_c) + (1 - pos_ratio) * neg_e);
   neg_predict = ((1 - pos_ratio) * (1 - neg_e)) / (pos_ratio * (1 - pos_c) + (1 - pos_ratio) * (1 - neg_e));  

}

void process(std::vector<pcl::PointCloud<PointType>::Ptr> pos, std::vector<pcl::PointCloud<PointType>::Ptr> neg) {
  std::vector<double> pos_scores;
  std::vector<double> neg_scores;
  
  std::cout << "\tPositive:\n";
  for(int i = 0; i < pos.size(); i++) {
    pos_scores.push_back(match(pos[i]));

  }
  std::sort(pos_scores.begin(), pos_scores.end());
  for (int i = 0; i < pos_scores.size(); i++) {
    std::cout << pos_scores[i] << "  ";
  }
  std::cout << "\n";

  std::cout << "\tNegative:\n";
  for(int i = 0; i < neg.size(); i++) {
    neg_scores.push_back(match(neg[i]));
  }

  std::sort(neg_scores.begin(), neg_scores.end());
  for (int i = 0; i < neg_scores.size(); i++) {
    std::cout << neg_scores[i] << "  ";
  }
  std::cout << "\n";

  int pi = findFirstLarger(pos_scores, neg_scores[0]);
  if (pi == pos_scores.size()) {
    std::cout << "\t All small\n"; 
  } else {
    std::cout << "\t" << pos_scores[pi] << std::endl;
  }

  int ni = 0;
  double threshold = neg_scores[ni];
  double min_error = 100;
  double best_thresh = threshold - 0.0001;
  while (pi < pos_scores.size() && ni < neg_scores.size()) {
    if (neg_scores[ni] <= pos_scores[pi]) {
       threshold = neg_scores[ni] - 0.0001;
       ni++;
    } else {
      threshold = pos_scores[pi];
      pi++;
    }
    double e = getError(pos_scores, neg_scores, threshold, 0);
    if (e < min_error) {
      min_error = e;
      best_thresh = threshold;
    }
  }
  
  double pos_c, neg_e, err, precision, neg_predict;
  getDetectionFalsePos(pos_scores, neg_scores, best_thresh, pos_c, neg_e, err, precision, neg_predict);
  PCL_ERROR("Best Threshold: %f, error: %f .\n", best_thresh, min_error);
  PCL_ERROR("Deteciton rate: %f, false positive: %f, error rate: %f.\n", pos_c, neg_e, err);
  PCL_ERROR("precision: %f, negative predict: %f \n", precision, neg_predict);
  if (outfile.is_open()) {
    outfile << precision << std::endl;
    outfile << neg_predict << std::endl;
  }

}

void learnCDP(Map& pos_samples, Map& neg_samples) {
 
  Map::iterator it;
  for(it = pos_samples.begin(); it != pos_samples.end(); it++) {
    // std::cout << it->first << "  " << std::endl;
    pcl::console::print_highlight ("%d .\n", it->first);
    Map::iterator it2 = neg_samples.find(it->first);
    if (it2 == neg_samples.end()) {
      continue;
    }
    if (outfile.is_open()) {
      outfile << it->first << std::endl;
    }
    process(it->second, it2->second);
  }
}

int
main(int argc, char** argv) {
  if (argc < 3) {
    PCL_ERROR ("Need at least two parameters! Syntax is: %s [data] [model_dir]\n", argv[0]);
    return (-1);
  }

  int step = 20;
  pcl::console::parse_argument (argc, argv, "-step", step);

  pos_prior = -1;
  pcl::console::parse_argument (argc, argv, "-pos_prior", pos_prior);

  output_dir = "";
  pcl::console::parse_argument (argc, argv, "-o", output_dir);

  std::cout << output_dir.size() << std::endl;

  if (output_dir.size() && pos_prior > 0) {
    outfile.open(output_dir.c_str(), std::ios::out | std::ios::trunc);
    outfile << pos_prior << std::endl;
  }

  Map pos_samples;
  Map neg_samples;

  loadData(argv[1], step, 1, pos_samples);
  loadData(argv[1], step, -1, neg_samples);

  loadFlannVFH(argv[2]);

  // std::map<int, std::vector<pcl::PointCloud<PointType>::Ptr> >::iterator it;
  // for(it = pos_samples.begin(); it != pos_samples.end(); it++) {
  //   std::cout << it->first << "  " << std::endl;
  //   for (int i = 0; i < it->second.size(); i++) {
  //     std::cout << "\t" << it->second[i]->points.size() << std::endl;
  //   }
  // }

  learnCDP(pos_samples, neg_samples);
  if (outfile.is_open()) {
    outfile.close();
  }

  return 0;
}