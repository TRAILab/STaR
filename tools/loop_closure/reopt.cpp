// Offline batch re-optimization of a KISS-Matcher graph dump (local fix 4 in kiss/kiss_matcher_8b58b11_local_fixes.patch) under chosen
// noise models, to compare odometry noise, per-keyframe gravity priors, robust loop kernels and
// incremental (KISS iSAM2) vs batch (LM) solutions on fixed measurements.
// usage: reopt GRAPH_G2O RAW_TUM OUT_TUM ODOM_ROT_DEG ODOM_TRANS_M LOOP_ROT_DEG LOOP_TRANS_M ATT_DEG
//              KERNEL(none|huber|cauchy) INIT(raw|g2o)
// ATT_DEG 0 disables the gravity prior. KISS defaults: 0.573 0.1 0.573 0.1 0 (kf0 prior is added here).
#include <gtsam/geometry/Pose3.h>
#include <gtsam/navigation/AttitudeFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace gtsam;

int main(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "usage: reopt GRAPH_G2O RAW_TUM OUT_TUM ODOM_ROT_DEG ODOM_TRANS_M LOOP_ROT_DEG "
                 "LOOP_TRANS_M ATT_DEG KERNEL INIT\n";
    return 2;
  }
  const std::string kernel = argv[9], init = argv[10];
  const double d2r = M_PI / 180.0;
  const double odr = std::atof(argv[4]) * d2r, odt = std::atof(argv[5]);
  const double lpr = std::atof(argv[6]) * d2r, lpt = std::atof(argv[7]);
  const double att = std::atof(argv[8]) * d2r;

  std::vector<double> stamps;
  Values raw;
  {
    std::ifstream f(argv[2]);
    std::string line;
    size_t k = 0;
    while (std::getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ss(line);
      double t, x, y, z, qx, qy, qz, qw;
      ss >> t >> x >> y >> z >> qx >> qy >> qz >> qw;
      stamps.push_back(t);
      raw.insert(k++, Pose3(Rot3::Quaternion(qw, qx, qy, qz), Point3(x, y, z)));
    }
  }
  auto graph_and_values = readG2o(argv[1], true);
  const NonlinearFactorGraph &dumped = *graph_and_values.first;
  const Values &isam_values = *graph_and_values.second;

  auto diag = [](double r, double t) {
    return noiseModel::Diagonal::Sigmas((Vector(6) << r, r, r, t, t, t).finished());
  };
  SharedNoiseModel odom_noise = diag(odr, odt), loop_noise = diag(lpr, lpt);
  if (kernel == "huber") {
    loop_noise = noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(1.345), loop_noise);
  } else if (kernel == "cauchy") {
    loop_noise = noiseModel::Robust::Create(noiseModel::mEstimator::Cauchy::Create(1.0), loop_noise);
  }

  NonlinearFactorGraph g;
  std::vector<size_t> odom_idx, loop_idx, att_idx;
  for (const auto &factor : dumped) {
    const auto *b = dynamic_cast<const BetweenFactor<Pose3> *>(factor.get());
    if (!b) continue;
    const Key a = b->key1(), c = b->key2();
    const bool odom = (c == a + 1) || (a == c + 1);
    (odom ? odom_idx : loop_idx).push_back(g.size());
    g.add(BetweenFactor<Pose3>(a, c, b->measured(), odom ? odom_noise : loop_noise));
  }
  g.add(PriorFactor<Pose3>(0, raw.at<Pose3>(0), diag(1e-4, 1e-3)));
  if (att > 0) {
    for (size_t i = 0; i < stamps.size(); ++i) {
      // Holds body z, in the world frame, to the odometry's. On sloped ground this also resists
      // world-frame yaw changes, unlike a prior on world up in the body frame.
      const Point3 up = raw.at<Pose3>(i).rotation().rotate(Point3(0, 0, 1));
      att_idx.push_back(g.size());
      g.add(Pose3AttitudeFactor(i, Unit3(up), noiseModel::Isotropic::Sigma(2, att), Unit3(0, 0, 1)));
    }
  }
  if (isam_values.size() != stamps.size()) {
    std::cerr << "keyframe count mismatch: g2o " << isam_values.size() << " raw " << stamps.size() << "\n";
    return 3;
  }

  auto breakdown = [&](const Values &v) {
    double eo = 0, el = 0, ea = 0;
    for (auto i : odom_idx) eo += g.at(i)->error(v);
    for (auto i : loop_idx) el += g.at(i)->error(v);
    for (auto i : att_idx) ea += g.at(i)->error(v);
    std::ostringstream s;
    s << std::setprecision(4) << "total " << g.error(v) << " (odom " << eo << ", loop " << el << ", gravity " << ea << ")";
    return s.str();
  };
  const Values &initial = (init == "g2o") ? isam_values : raw;
  LevenbergMarquardtParams params;
  params.setMaxIterations(500);
  params.setRelativeErrorTol(1e-10);
  params.setAbsoluteErrorTol(1e-10);
  LevenbergMarquardtOptimizer optimizer(g, initial, params);
  const Values result = optimizer.optimize();

  std::cout << "factors: odom " << odom_idx.size() << ", loop " << loop_idx.size() << ", gravity "
            << att_idx.size() << "; keyframes " << stamps.size() << "\n";
  std::cout << "error at raw        : " << breakdown(raw) << "\n";
  std::cout << "error at KISS iSAM2 : " << breakdown(isam_values) << "\n";
  std::cout << "error at batch LM   : " << breakdown(result) << "  iterations " << optimizer.iterations() << "\n";

  std::ofstream out(argv[3]);
  out << "#timestamp x y z qx qy qz qw\n" << std::fixed << std::setprecision(9);
  for (size_t i = 0; i < stamps.size(); ++i) {
    const Pose3 p = result.at<Pose3>(i);
    const auto q = p.rotation().toQuaternion();
    out << stamps[i] << " " << p.x() << " " << p.y() << " " << p.z() << " " << q.x() << " " << q.y() << " "
        << q.z() << " " << q.w() << "\n";
  }
  return 0;
}
