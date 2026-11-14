// SPDX-License-Identifier: MIT
// C++ integration against NVIDIA cuVS 26.10 GMM public API.
// Reads float32 row-major train/eval features. Outputs weights/means/cov
// as a flat float32 file and one JSON line containing measured timings.
#include <cuvs/cluster/gmm.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
std::string arg(int argc, char** argv, std::string key) {
  for (int i=1; i+1<argc; ++i) if (key == argv[i]) return argv[i+1];
  throw std::invalid_argument("Missing parameter "+key);
}
std::vector<float> read_f32(std::string path, size_t length) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("Cannot read "+path);
  if (f.tellg() != static_cast<std::streamoff>(length*sizeof(float)))
    throw std::invalid_argument("Input byte length mismatch: "+path);
  std::vector<float> a(length);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(a.data()), length*sizeof(float));
  if (!f) throw std::runtime_error("Incomplete data in "+path);
  if (std::any_of(a.begin(), a.end(), [](float v) {return !std::isfinite(v);}))
    throw std::invalid_argument("Non-finite input");
  return a;
}
size_t cov_len(std::string type, size_t K, size_t d) {
  if (type=="diag") return K*d;
  if (type=="spherical") return K;
  if (type=="tied") return d*d;
  if (type=="full") return K*d*d;
  throw std::invalid_argument("Unknown covariance");
}
cuvs::cluster::gmm::covariance_type to_cov(std::string type) {
  using T = cuvs::cluster::gmm::covariance_type;
  if(type=="diag") return T::DIAG;
  if(type=="spherical") return T::SPHERICAL;
  if(type=="tied") return T::TIED;
  if(type=="full") return T::FULL;
  throw std::invalid_argument("Unknown covariance");
}
void write_f32(std::string path, std::vector<float> const& x) {
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Cannot create "+path);
  f.write(reinterpret_cast<char const*>(x.data()), x.size()*sizeof(float));
  if (!f) throw std::runtime_error("Write failed "+path);
}
}
int main(int argc, char** argv) {
  try {
    using namespace cuvs::cluster;
    int64_t N=std::stoll(arg(argc,argv,"--n"));
    int64_t D=std::stoll(arg(argc,argv,"--d"));
    int64_t NV=std::stoll(arg(argc,argv,"--eval-n"));
    int K=std::stoi(arg(argc,argv,"--k"));
    int iters=std::stoi(arg(argc,argv,"--iterations"));
    auto reg=std::stod(arg(argc,argv,"--reg"));
    auto seed=std::stoull(arg(argc,argv,"--seed"));
    std::string type=arg(argc,argv,"--covariance");
    std::string init=arg(argc,argv,"--init");
    if (N<K || D<1 || NV<1 || K<1 || iters<1 || reg<=0)
      throw std::invalid_argument("Invalid dimensions or EM hyperparameters");
    size_t cn=cov_len(type,K,D);
    auto host_x=read_f32(arg(argc,argv,"--train"), N*D);
    auto host_v=read_f32(arg(argc,argv,"--eval"), NV*D);

    raft::resources res;
    cudaStream_t stream=raft::resource::get_cuda_stream(res).get();
    auto x=raft::make_device_matrix<float,int64_t>(res,N,D);
    auto v=raft::make_device_matrix<float,int64_t>(res,NV,D);
    raft::update_device(x.data_handle(),host_x.data(),N*D,stream);
    raft::update_device(v.data_handle(),host_v.data(),NV*D,stream);
    auto weights=raft::make_device_vector<float,int64_t>(res,K);
    auto means=raft::make_device_matrix<float,int64_t>(res,K,D);
    auto cov=raft::make_device_vector<float,int64_t>(res,cn);
    auto pchol=raft::make_device_vector<float,int64_t>(res,cn);
    auto prec=raft::make_device_vector<float,int64_t>(res,cn);
    auto labels=raft::make_device_vector<int,int64_t>(res,N);
    auto logp=raft::make_device_vector<float,int64_t>(res,NV);
    raft::resource::sync_stream(res,stream);
    gmm::params cfg;
    cfg.n_components=K;
    cfg.cov_type=to_cov(type);
    cfg.max_iter=iters;
    cfg.seed=seed;
    cfg.tol=1e-10;
    cfg.reg_covar=reg;
    cfg.n_init=1;
    if (init=="kmeans") cfg.init=gmm::init_method::KMeans;
    else if (init=="random") cfg.init=gmm::init_method::Random;
    else throw std::invalid_argument("Invalid init");
    float lower=0;
    int n_iter=0;
    bool converged=false;
    auto start=std::chrono::steady_clock::now();
    gmm::fit(res,cfg,raft::make_const_mdspan(x.view()),
             weights.view(),means.view(),cov.view(),pchol.view(),
             prec.view(),labels.view(),raft::make_host_scalar_view(&lower),
             raft::make_host_scalar_view(&n_iter),raft::make_host_scalar_view(&converged));
    raft::resource::sync_stream(res,stream);
    auto fitted=std::chrono::steady_clock::now();
    auto score=[&] {
      gmm::score_samples(res,cfg,raft::make_const_mdspan(v.view()),
        raft::make_const_mdspan(weights.view()),raft::make_const_mdspan(means.view()),
        raft::make_const_mdspan(pchol.view()),logp.view());
    };
    score(); // warm up
    raft::resource::sync_stream(res,stream);
    auto s0=std::chrono::steady_clock::now();
    score();
    raft::resource::sync_stream(res,stream);
    auto s1=std::chrono::steady_clock::now();
    std::vector<float> hw(K),hm(K*D),hc(cn),hl(NV);
    raft::update_host(hw.data(),weights.data_handle(),K,stream);
    raft::update_host(hm.data(),means.data_handle(),K*D,stream);
    raft::update_host(hc.data(),cov.data_handle(),cn,stream);
    raft::update_host(hl.data(),logp.data_handle(),NV,stream);
    raft::resource::sync_stream(res,stream);
    std::vector<float> packed;
    packed.insert(packed.end(),hw.begin(),hw.end());
    packed.insert(packed.end(),hm.begin(),hm.end());
    packed.insert(packed.end(),hc.begin(),hc.end());
    write_f32(arg(argc,argv,"--model"),packed);
    auto nonfinite=std::count_if(hl.begin(),hl.end(),[](float a){return !std::isfinite(a);});
    double mean=std::accumulate(hl.begin(),hl.end(),0.0)/NV;
    double var=0;
    for (auto y:hl) var+=(y-mean)*(y-mean);
    size_t free_mem=0,total_mem=0;
    cudaMemGetInfo(&free_mem,&total_mem);
    double fitsec=std::chrono::duration<double>(fitted-start).count();
    double scoresec=std::chrono::duration<double>(s1-s0).count();
    std::cout << std::setprecision(12)
      << "{\"backend\":\"cuvs\",\"fit_seconds\":"<<fitsec
      << ",\"score_seconds\":"<<scoresec
      << ",\"score_frames\":"<<NV
      << ",\"log_likelihood_mean\":"<<mean
      << ",\"log_likelihood_std\":"<<std::sqrt(var/NV)
      << ",\"sum_weights\":"<<std::accumulate(hw.begin(),hw.end(),0.0)
      << ",\"nonfinite_heldout_logprobs\":"<<nonfinite
      << ",\"iterations_reported\":"<<n_iter
      << ",\"converged\":"<<(converged?"true":"false")
      << ",\"lower_bound\":"<<lower
      << ",\"gpu_memory_free_end_bytes\":"<<free_mem
      << ",\"gpu_memory_total_bytes\":"<<total_mem
      << ",\"gpu_peak_bytes\":null}"<<std::endl;
    return nonfinite?3:0;
  } catch (std::exception const& ex) {
    std::cerr << "cuVS bonus failed: " << ex.what() << std::endl;
    return 2;
  }
}
